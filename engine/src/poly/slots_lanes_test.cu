// src/poly/slots_lanes_test.cu - the cached experts' slots launches of native_mmvq.cu's formats with MmvqSlots::lanes
// (Inkling 2, CUDA: native_mmvq_slots_lanes_kernel, a warp a row) against the same launches without it, to the bit:
// random blocks of IQ4_NL, Q8_0, Q2_0, MXFP4, Q4_0, Q5_0, Q2_K..Q6_K and IQ4_XS, 1 to 32 matrices a launch, row counts
// that leave a block's last warps short, n_in from 1 block to 4096 values (both of the block kernel's row layouts),
// through the host's list (MmvqSlots) and the card's picks (MmvqIds); every slot's output is followed by canaries
// that must stay, and every row must be written.  STRATA_SLOTS_WARP=1 compares against the warp
// kernel where it takes a launch.  Q5_0, Q3_K, Q5_K and Q6_K keep the block kernel with lanes set (kSlotsLanesFaster in
// native_mmvq.cu), so for them both launches are the old kernel's.  --bench also times Inkling 2's 8-expert launches with and without lanes.
//
// usage: poly_slots_lanes_test [--bench | --quick]   (exit 0: all agree)
#include "strata/kernels/native_mmvq.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace {

namespace k = strata::kernels;

int failures = 0, checks = 0;
std::mt19937 rng(4321);

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::printf("CUDA error at %s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

template <typename T> T* up(const std::vector<T>& v) {
    T* d = nullptr;
    ck(cudaMalloc(&d, std::max<size_t>(1, v.size()) * sizeof(T)), "malloc");
    if (!v.empty()) ck(cudaMemcpy(d, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice), "upload");
    return d;
}

// a format: its values a block, and where its fp16 scales (or MXFP4's exponent byte) sit in a block
struct Kind {
    int type;
    const char* name;
    int qk;
    std::vector<int> f16;   // byte offsets of fp16 scales
    int e8 = -1;            // byte offset of an e8m0 exponent
};
const std::vector<Kind>& kinds() {
    static const std::vector<Kind> v = {
        {20, "IQ4_NL", 32, {0}},   {8, "Q8_0", 32, {0}},     {42, "Q2_0", 64, {0}},     {39, "MXFP4", 32, {}, 0},
        {2, "Q4_0", 32, {0}},      {6, "Q5_0", 32, {0}},     {10, "Q2_K", 256, {80, 82}}, {11, "Q3_K", 256, {108}},
        {12, "Q4_K", 256, {0, 2}}, {13, "Q5_K", 256, {0, 2}}, {14, "Q6_K", 256, {208}},   {23, "IQ4_XS", 256, {0}},
    };
    return v;
}

// random bytes with sane scales in every block
void* random_weights(const Kind& kd, int n_in, int n_out) {
    const size_t blk = k::native_mmvq_weight_bytes(kd.type, kd.qk, 1);
    std::uniform_int_distribution<int> byte(0, 255), expo(118, 126);
    std::uniform_real_distribution<float> scale(1e-3f, 1e-2f);
    std::vector<uint8_t> h(k::native_mmvq_weight_bytes(kd.type, n_in, n_out));
    for (uint8_t& c : h) c = (uint8_t) byte(rng);
    for (size_t b = 0; b < h.size() / blk; ++b) {
        for (int off : kd.f16) {
            const __half d = __float2half(scale(rng));
            std::memcpy(h.data() + b * blk + off, &d, 2);
        }
        if (kd.e8 >= 0) h[b * blk + kd.e8] = (uint8_t) expo(rng);
    }
    return up(h);
}

// n_in values of each of `rows` tokens in Q8_1 on the card
void* random_q8_1(int n_in, int rows, cudaStream_t st) {
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> h(size_t(n_in) * rows);
    for (float& v : h) v = nd(rng);
    float* x = up(h);
    void* q;
    ck(cudaMalloc(&q, size_t(rows) * k::native_q8_1_bytes(n_in)), "malloc");
    for (int r = 0; r < rows; ++r)   // a column at a time: one launch takes at most MAX_NCOLS columns
        k::native_quantize_q8_1(x + size_t(r) * n_in, (uint8_t*) q + r * k::native_q8_1_bytes(n_in), n_in, 1, st);
    ck(cudaStreamSynchronize(st), "quantize");
    cudaFree(x);
    return q;
}

size_t count_diff(const float* a, const float* b, size_t n) {
    std::vector<float> x(n), y(n);
    ck(cudaDeviceSynchronize(), "sync");
    ck(cudaMemcpy(x.data(), a, n * 4, cudaMemcpyDeviceToHost), "download");
    ck(cudaMemcpy(y.data(), b, n * 4, cudaMemcpyDeviceToHost), "download");
    size_t diff = 0;
    for (size_t i = 0; i < n; ++i) diff += std::memcmp(&x[i], &y[i], 4) != 0;
    return diff;
}

void same_bits(const std::string& what, const float* a, const float* b, size_t n) {
    const size_t diff = count_diff(a, b, n);
    ++checks;
    std::printf("%-60s %zu of %zu values differ  %s\n", what.c_str(), diff, n, diff ? "FAIL" : "ok");
    if (diff) ++failures;
}

// a NaN that no kernel writes: every slot's y range is followed by kPad of them, and every buffer starts full of it
constexpr uint32_t kCanary = 0x7fbadbadu;
constexpr int kPad = 37;

// rows still holding the canary (not written) and pad values no longer holding it (written past a slot's range)
void canaries(const std::string& what, const float* d, int n_out, int count) {
    const size_t stride = size_t(n_out) + kPad;
    std::vector<uint32_t> h(stride * count);
    ck(cudaDeviceSynchronize(), "sync");
    ck(cudaMemcpy(h.data(), d, h.size() * 4, cudaMemcpyDeviceToHost), "download");
    size_t unwritten = 0, overrun = 0;
    for (int i = 0; i < count; ++i)
        for (size_t r = 0; r < stride; ++r) {
            const bool canary = h[size_t(i) * stride + r] == kCanary;
            if (r < size_t(n_out)) unwritten += canary;
            else overrun += !canary;
        }
    ++checks;
    const bool bad = unwritten || overrun;
    std::printf("%-60s %zu rows unwritten, %zu pad values written  %s\n", what.c_str(), unwritten, overrun,
                bad ? "FAIL" : "ok");
    if (bad) ++failures;
}

void test_slots_lanes(const Kind& kd, int n_in, int n_out, int count) {
    cudaStream_t st;
    ck(cudaStreamCreate(&st), "stream");
    std::vector<void*> w((size_t) count);
    for (void*& p : w) p = random_weights(kd, n_in, n_out);
    void* xq = random_q8_1(n_in, count, st);
    const size_t xr = k::native_q8_1_bytes(n_in), stride = size_t(n_out) + kPad, total = stride * count;
    const std::vector<uint32_t> fill(total, kCanary);
    float *ya, *yb, *yc, *yd;
    for (float** p : {&ya, &yb, &yc, &yd}) {
        ck(cudaMalloc(p, total * sizeof(float)), "malloc");
        ck(cudaMemcpy(*p, fill.data(), total * sizeof(float), cudaMemcpyHostToDevice), "canaries");
    }
    k::MmvqSlots slots{}, plain{};
    slots.lanes = true;
    for (int i = 0; i < count; ++i) {
        const void* x = (const uint8_t*) xq + i * xr;
        slots.w[i] = plain.w[i] = w[(size_t) i], slots.x[i] = plain.x[i] = x;
        slots.y[i] = yb + size_t(i) * stride, plain.y[i] = yd + size_t(i) * stride;
        k::native_mmvq(kd.type, w[(size_t) i], x, ya + size_t(i) * stride, n_in, n_out, 1, st);
    }
    // the same through the card's picks: expert i is table[ids[i]], the table backwards
    std::vector<const void*> table(w.rbegin(), w.rend());
    std::vector<int32_t> ids((size_t) count);
    for (int i = 0; i < count; ++i) ids[(size_t) i] = count - 1 - i;
    void* dtable;
    ck(cudaMalloc(&dtable, table.size() * sizeof(void*)), "malloc");
    ck(cudaMemcpy(dtable, table.data(), table.size() * sizeof(void*), cudaMemcpyHostToDevice), "table");
    int32_t* dids = up(ids);
    const k::MmvqIds picks{(const void* const*) dtable, dids, 0, xq, xr, 1, yc, stride, true};
    const std::string what = std::string("slots lanes ") + kd.name + " " + std::to_string(n_in) + " -> " +
                             std::to_string(n_out) + " x" + std::to_string(count);
    if (!k::native_mmvq_slots(kd.type, slots, count, n_in, n_out, st) ||
        !k::native_mmvq_ids(kd.type, picks, count, n_in, n_out, st) ||
        !k::native_mmvq_slots(kd.type, plain, count, n_in, n_out, st)) {
        std::printf("%-60s not taken  FAIL\n", what.c_str());
        ++failures;
    } else {
        same_bits(what + " vs without lanes", yb, yd, total);
        same_bits(what + ", picks vs without lanes", yc, yd, total);
        canaries(what + ", lanes rows and pads", yb, n_out, count);
        canaries(what + ", picks rows and pads", yc, n_out, count);
        canaries(what + ", without lanes rows and pads", yd, n_out, count);
        // for information: one native_mmvq() a matrix (its own kernels; not what this test guards)
        const size_t single = count_diff(ya, yd, total);
        std::printf("%-60s %zu of %zu values differ  (info: native_mmvq vs slots without lanes)\n",
                    (what + ", single").c_str(), single, total);
    }
    for (void* p : w) cudaFree(p);
    cudaFree(xq), cudaFree(dtable), cudaFree(dids);
    cudaFree(ya), cudaFree(yb), cudaFree(yc), cudaFree(yd);
    cudaStreamDestroy(st);
}

// the best of 5 replays of a graph of `reps` launches, a launch
template <class F> float time_us(cudaStream_t s, F&& f, int reps = 200) {
    f();
    ck(cudaStreamSynchronize(s), "warm-up");
    cudaGraph_t gr;
    cudaGraphExec_t ex;
    ck(cudaStreamBeginCapture(s, cudaStreamCaptureModeRelaxed), "capture");
    for (int i = 0; i < reps; ++i) f();
    ck(cudaStreamEndCapture(s, &gr), "end capture");
    ck(cudaGraphInstantiate(&ex, gr, 0), "instantiate");
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0), cudaEventCreate(&e1);
    float best = 1e30f;
    for (int i = 0; i < 6; ++i) {
        cudaEventRecord(e0, s);
        cudaGraphLaunch(ex, s);
        cudaEventRecord(e1, s);
        cudaEventSynchronize(e1);
        float ms = 0;
        cudaEventElapsedTime(&ms, e0, e1);
        if (i) best = std::min(best, ms);   // the first replay uploads the graph
    }
    cudaEventDestroy(e0), cudaEventDestroy(e1);
    cudaGraphExecDestroy(ex), cudaGraphDestroy(gr);
    return best * 1e3f / reps;
}

// Inkling 2's cached experts' gate|up (2304 -> 2304) and down (1152 -> 2304; 1280 for the 256-value formats), 8
// experts a launch, every format, with and without lanes.  Four sets of 8 experts' weights take turns, so a launch's weights are not still in L2.
void bench() {
    cudaStream_t st;
    ck(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking), "stream");
    for (const Kind& kd : kinds()) {
        for (int n_in : {2304, kd.qk == 256 ? 1280 : 1152}) {
            const int n_out = 2304, sets = 4;
            std::vector<void*> w(8 * sets);
            for (void*& p : w) p = random_weights(kd, n_in, n_out);
            void* xq = random_q8_1(n_in, 8, st);
            float* y;
            ck(cudaMalloc(&y, size_t(8) * n_out * 4), "malloc");
            for (bool lanes : {false, true}) {
                std::vector<k::MmvqSlots> sl((size_t) sets);
                for (int j = 0; j < sets; ++j) {
                    sl[(size_t) j].lanes = lanes;
                    for (int i = 0; i < 8; ++i)
                        sl[(size_t) j].w[i] = w[size_t(8 * j + i)],
                        sl[(size_t) j].x[i] = (const uint8_t*) xq + i * k::native_q8_1_bytes(n_in),
                        sl[(size_t) j].y[i] = y + size_t(i) * n_out;
                }
                int r = 0;
                const float us = time_us(st, [&] {
                    k::native_mmvq_slots(kd.type, sl[(size_t) (r++ % sets)], 8, n_in, n_out, st);
                });
                std::printf("bench slots %-7s %4d -> %d x8%-7s %8.1f us %7.1f GB/s\n", kd.name, n_in, n_out,
                            lanes ? ", lanes" : "", us, 8.0 * k::native_mmvq_weight_bytes(kd.type, n_in, n_out) / (us * 1e3));
            }
            for (void* p : w) cudaFree(p);
            cudaFree(xq), cudaFree(y);
        }
    }
    cudaStreamDestroy(st);
}

}  // namespace

int main(int argc, char** argv) {
    int dev = 0;
    if (cudaGetDeviceCount(&dev) != cudaSuccess || dev == 0) {
        std::printf("no CUDA device: skipped\n");
        return 77;
    }
    int major = 0, minor = 0;
    cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, 0);
    cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, 0);
    const char* wide = std::getenv("STRATA_SLOTS_WARP");
    std::printf("sm_%d%d, STRATA_SLOTS_WARP=%s\n", major, minor, wide ? wide : "(unset)");
    // --quick (for compute-sanitizer): Inkling 2's two shapes with short tails, a launch of 3, each format
    const bool quick = argc > 1 && std::strcmp(argv[1], "--quick") == 0;
    for (const Kind& kd : kinds()) {
        const int down = kd.qk == 256 ? 1280 : 1152;   // the down projection's K; 1152 is not whole 256-value blocks
        // Inkling 2's shapes (gate|up 2304 -> 2304, down 1152 -> 2304) with 1 and 8 experts; a launch of 32; row
        // counts that leave a block's last warps without a row; and K from one block up to 4096 values, on both sides
        // of the block kernel's switch to two rows a block (fewer blocks a row than the block's thread groups)
        if (quick) {
            test_slots_lanes(kd, 2304, 2301, 3);
            test_slots_lanes(kd, down, 37, 3);
            continue;
        }
        for (int count : {1, 8}) {
            test_slots_lanes(kd, 2304, 2304, count);
            test_slots_lanes(kd, down, 2304, count);
        }
        test_slots_lanes(kd, 2304, 2301, 32);
        test_slots_lanes(kd, down, 37, 3);
        test_slots_lanes(kd, 4096, 1023, 2);
        test_slots_lanes(kd, kd.qk, 7, 5);
        test_slots_lanes(kd, 3 * kd.qk, 1, 1);
        test_slots_lanes(kd, 512, 130, 4);
        test_slots_lanes(kd, 768, 66, 6);
    }
    if (argc > 1 && std::strcmp(argv[1], "--bench") == 0) bench();
    std::printf("%s: %d failures of %d checks\n", failures ? "FAIL" : "ok", failures, checks);
    return failures ? 1 : 0;
}
