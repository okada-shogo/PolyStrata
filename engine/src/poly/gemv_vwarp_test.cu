// src/poly/gemv_vwarp_test.cu - the HIP gfx11 / gfx12 virtual-warp decode GEMVs (kernels::native_gemv_vwarp_force) against the
// block kernels they replace, byte for byte: Inkling 2 Mini's expert products through native_mmvq_slots() and
// native_mmvq_ids() (IQ4_NL and Q8_0, n_in 2304 and 1152, 1 to 32 experts), the grouped dense Q8_0 launch
// native_mmvq_many(), and the loop controllers' down product ink_ctrl_down() (F32 and BF16).  Every output of the new
// kernels must be the same bytes as the old kernels' (outputs start as 0xff bytes, so an unwritten value differs).
//
// usage: poly_gemv_vwarp_test   (exit 0: every byte the same; 77: not a gfx11 / gfx12 card, nothing to compare)
#include "inkling2_kernels.hpp"
#include "strata/kernels/native_mmvq.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace {

using namespace strata;
std::mt19937 rng(20261124);
cudaStream_t st;
int g_fail = 0, g_cases = 0;

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::printf("GPU error at %s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}
template <typename T> T* upload(const std::vector<T>& v) {
    T* d = nullptr;
    ck(cudaMalloc(&d, std::max<size_t>(1, v.size() * sizeof(T))), "malloc");
    ck(cudaMemcpy(d, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice), "upload");
    ck(cudaDeviceSynchronize(), "sync");   // the null stream's copies are done before a launch on the non-blocking stream
    return d;
}
float* fill_ff(size_t n) {
    float* d = nullptr;
    ck(cudaMalloc(&d, n * 4), "malloc");
    ck(cudaMemset(d, 0xff, n * 4), "memset");
    ck(cudaDeviceSynchronize(), "sync");   // the memset runs on the null stream: done before the launch on st
    return d;
}
std::vector<unsigned char> grab(const void* p, size_t n) {
    std::vector<unsigned char> h(n);
    ck(cudaStreamSynchronize(st), "sync");
    ck(cudaMemcpy(h.data(), p, n, cudaMemcpyDeviceToHost), "download");
    return h;
}
uint16_t half_bits(float f) {
    const __half h = __float2half(f);
    uint16_t b;
    std::memcpy(&b, &h, 2);
    return b;
}
// random weight blocks: a half scale in bytes 0..1, random code bytes after it (IQ4_NL 18 bytes, Q8_0 34)
std::vector<uint8_t> rand_blocks(size_t nblocks, int bsize) {
    std::vector<uint8_t> v(nblocks * bsize);
    std::uniform_int_distribution<int> byte(0, 255);
    std::uniform_real_distribution<float> sc(0.002f, 0.02f);
    for (size_t b = 0; b < nblocks; ++b) {
        uint8_t* p = v.data() + b * bsize;
        const uint16_t d = half_bits(sc(rng) * (byte(rng) & 1 ? 1.f : -1.f));
        std::memcpy(p, &d, 2);
        for (int i = 2; i < bsize; ++i) p[i] = uint8_t(byte(rng));
    }
    return v;
}
std::vector<uint8_t> rand_q81(int n) {   // n / 32 Q8_1 blocks: ds = (d, d * sum)
    const int nb = n / 32;
    std::vector<uint8_t> v(size_t(nb) * 36);
    std::uniform_int_distribution<int> q(-127, 127);
    std::uniform_real_distribution<float> sc(0.005f, 0.05f);
    for (int b = 0; b < nb; ++b) {
        uint8_t* p = v.data() + size_t(b) * 36;
        int8_t qs[32];
        int sum = 0;
        for (int i = 0; i < 32; ++i) qs[i] = int8_t(q(rng)), sum += qs[i];
        const float d = sc(rng);
        const uint16_t h0 = half_bits(d), h1 = half_bits(d * float(sum));
        std::memcpy(p, &h0, 2);
        std::memcpy(p + 2, &h1, 2);
        std::memcpy(p + 4, qs, 32);
    }
    return v;
}
std::vector<float> randf(size_t n, float s) {
    std::normal_distribution<float> g(0.0f, s);
    std::vector<float> v(n);
    for (auto& f : v) f = g(rng);
    return v;
}
void verdict(const char* what, bool same) {
    ++g_cases;
    if (!same) ++g_fail;
    std::printf("%s %s\n", same ? "ok  " : "FAIL", what);
}

int bsize_of(int type) { return type == 8 ? 34 : 18; }

// ---- expert slots and ids ----
void test_slots(int type, int n_in, int n_out, int count) {
    char what[160];
    std::snprintf(what, sizeof what, "slots type %2d n_in %4d n_out %4d experts %2d", type, n_in, n_out, count);
    std::vector<uint8_t*> w;
    std::vector<void*> x;
    kernels::MmvqSlots s0{}, s1{};
    std::vector<float*> y0, y1;
    for (int j = 0; j < count; ++j) {
        w.push_back(upload(rand_blocks(size_t(n_out) * (n_in / 32), bsize_of(type))));
        x.push_back(upload(rand_q81(n_in)));
        y0.push_back(fill_ff(n_out)), y1.push_back(fill_ff(n_out));
        s0.w[j] = s1.w[j] = w[j], s0.x[j] = s1.x[j] = x[j], s0.y[j] = y0[j], s1.y[j] = y1[j];
    }
    kernels::native_gemv_vwarp_force(0);
    const bool ok0 = kernels::native_mmvq_slots(type, s0, count, n_in, n_out, st);
    kernels::native_gemv_vwarp_force(1);
    const bool ok1 = kernels::native_mmvq_slots(type, s1, count, n_in, n_out, st);
    bool same = ok0 && ok1;
    for (int j = 0; j < count; ++j) same = same && grab(y0[j], size_t(n_out) * 4) == grab(y1[j], size_t(n_out) * 4);
    verdict(what, same);
    for (int j = 0; j < count; ++j) cudaFree(w[j]), cudaFree(x[j]), cudaFree(y0[j]), cudaFree(y1[j]);
}

void test_ids(int type, int n_in, int n_out, int count) {
    char what[160];
    std::snprintf(what, sizeof what, "ids   type %2d n_in %4d n_out %4d experts %2d", type, n_in, n_out, count);
    const int nexp = 12;   // the table's experts; the ids pick count of them (repeats allowed)
    const size_t off = 64;   // a byte offset into every expert's allocation
    const size_t wbytes = size_t(n_out) * (n_in / 32) * bsize_of(type);
    std::vector<uint8_t*> w;
    std::vector<const void*> table;
    for (int e = 0; e < nexp; ++e) {
        auto b = rand_blocks((wbytes + off) / bsize_of(type) + 1, bsize_of(type));
        w.push_back(upload(b));
        table.push_back(w.back());
    }
    std::vector<int32_t> ids(count);
    for (int j = 0; j < count; ++j) ids[j] = int32_t((j * 5 + 3) % nexp);
    const size_t xs = size_t(n_in / 32) * 36;
    std::vector<uint8_t> xh;
    for (int j = 0; j < count; ++j) {
        auto q = rand_q81(n_in);
        xh.insert(xh.end(), q.begin(), q.end());
    }
    void** dtable = nullptr;
    ck(cudaMalloc((void**) &dtable, nexp * sizeof(void*)), "malloc");
    ck(cudaMemcpy(dtable, table.data(), nexp * sizeof(void*), cudaMemcpyHostToDevice), "up");
    int32_t* dids = upload(ids);
    uint8_t* dx = upload(xh);
    float* y0 = fill_ff(size_t(count) * n_out);
    float* y1 = fill_ff(size_t(count) * n_out);
    kernels::MmvqIds a{dtable, dids, off, dx, xs, 1, y0, size_t(n_out)};
    kernels::MmvqIds b = a;
    b.y = y1;
    kernels::native_gemv_vwarp_force(0);
    const bool ok0 = kernels::native_mmvq_ids(type, a, count, n_in, n_out, st);
    kernels::native_gemv_vwarp_force(1);
    const bool ok1 = kernels::native_mmvq_ids(type, b, count, n_in, n_out, st);
    const size_t n = size_t(count) * n_out * 4;
    verdict(what, ok0 && ok1 && grab(y0, n) == grab(y1, n));
    for (auto* p : w) cudaFree(p);
    cudaFree(dtable), cudaFree(dids), cudaFree(dx), cudaFree(y0), cudaFree(y1);
}

// ---- grouped dense ----
void test_many(int type, int n_in, const std::vector<int>& n_outs) {
    char what[200];
    int off = std::snprintf(what, sizeof what, "many  type %2d n_in %4d rows", type, n_in);
    for (int o : n_outs) off += std::snprintf(what + off, sizeof what - off, " %d", o);
    kernels::NativeMmvqMany m{};
    m.count = int(n_outs.size());
    std::vector<uint8_t*> w;
    std::vector<void*> x;
    std::vector<float*> y0, y1;
    kernels::NativeMmvqMany m1{};
    m1.count = m.count;
    for (int j = 0; j < m.count; ++j) {
        w.push_back(upload(rand_blocks(size_t(n_outs[j]) * (n_in / 32), bsize_of(type))));
        x.push_back(upload(rand_q81(n_in)));
        y0.push_back(fill_ff(n_outs[j])), y1.push_back(fill_ff(n_outs[j]));
        m.w[j] = m1.w[j] = w[j], m.x_q8_1[j] = m1.x_q8_1[j] = x[j], m.y[j] = y0[j], m1.y[j] = y1[j];
        m.n_out[j] = m1.n_out[j] = n_outs[j];
    }
    if (!kernels::native_mmvq_many_fits(type, m, n_in)) {
        std::printf("FAIL %s: native_mmvq_many does not take it\n", what);
        ++g_fail, ++g_cases;
        return;
    }
    kernels::native_gemv_vwarp_force(0);
    kernels::native_mmvq_many(type, m, n_in, st);
    kernels::native_gemv_vwarp_force(1);
    kernels::native_mmvq_many(type, m1, n_in, st);
    bool same = true;
    for (int j = 0; j < m.count; ++j) same = same && grab(y0[j], size_t(n_outs[j]) * 4) == grab(y1[j], size_t(n_outs[j]) * 4);
    verdict(what, same);
    for (int j = 0; j < m.count; ++j) cudaFree(w[j]), cudaFree(x[j]), cudaFree(y0[j]), cudaFree(y1[j]);
}

// ---- loop controllers' down product ----
void test_ctrl_down(bool bf16, int count, int n_embd, int rank, int rows) {
    char what[160];
    std::snprintf(what, sizeof what, "ctrl_down %s controllers %d n_embd %d rank %d rows %d", bf16 ? "BF16" : "F32 ", count,
                  n_embd, rank, rows);
    std::vector<const float*> cur(count), msg(count);
    std::vector<const void*> down(count);
    std::vector<float*> y0(count), y1(count);
    std::vector<void*> frees;
    for (int j = 0; j < count; ++j) {
        float* c = upload(randf(size_t(n_embd) * rows, 1.0f));
        float* mg = upload(randf(size_t(n_embd) * rows, 0.7f));
        cur[j] = c, msg[j] = mg;
        const auto wf = randf(size_t(rank) * 2 * n_embd, 0.02f);
        if (bf16) {
            std::vector<uint16_t> wb(wf.size());
            for (size_t i = 0; i < wf.size(); ++i) {
                uint32_t u;
                std::memcpy(&u, &wf[i], 4);
                wb[i] = uint16_t(u >> 16);
            }
            down[j] = upload(wb);
        } else {
            down[j] = upload(wf);
        }
        y0[j] = fill_ff(size_t(rank) * rows), y1[j] = fill_ff(size_t(rank) * rows);
        frees.push_back(c), frees.push_back(mg), frees.push_back(const_cast<void*>(down[j]));
    }
    kernels::native_gemv_vwarp_force(0);
    const bool ok0 = poly::ink_ctrl_down(cur.data(), msg.data(), down.data(), y0.data(), count, n_embd, rank, rows, st, bf16);
    kernels::native_gemv_vwarp_force(1);
    const bool ok1 = poly::ink_ctrl_down(cur.data(), msg.data(), down.data(), y1.data(), count, n_embd, rank, rows, st, bf16);
    bool same = ok0 && ok1;
    for (int j = 0; j < count; ++j) same = same && grab(y0[j], size_t(rank) * rows * 4) == grab(y1[j], size_t(rank) * rows * 4);
    verdict(what, same);
    for (void* p : frees) cudaFree(p);
    for (int j = 0; j < count; ++j) cudaFree(y0[j]), cudaFree(y1[j]);
}

// ---- --bench [reps]: the old and the new kernels' time a launch, weights rotating through a pool far larger than the
// Infinity Cache ("cold": what a decode token sees), the two taken in turn ----
template <typename L> double time_it(L&& launch, int nsets, int reps) {
    cudaEvent_t a, b;
    ck(cudaEventCreate(&a), "ev");
    ck(cudaEventCreate(&b), "ev");
    for (int i = 0; i < 20; ++i) launch(i % nsets);
    std::vector<double> r;
    for (int k = 0; k < 5; ++k) {
        ck(cudaEventRecord(a, st), "rec");
        for (int i = 0; i < reps; ++i) launch((i * 7 + k) % nsets);
        ck(cudaEventRecord(b, st), "rec");
        ck(cudaEventSynchronize(b), "sync");
        float ms = 0;
        ck(cudaEventElapsedTime(&ms, a, b), "elapsed");
        r.push_back(1000.0 * ms / reps);
    }
    std::sort(r.begin(), r.end());
    return r[2];
}
void bench_slots(int type, int n_in, int n_out, int reps) {
    const int count = 8, pool = type == 8 ? 64 : 128;
    std::vector<uint8_t*> w;
    for (int i = 0; i < pool; ++i) w.push_back(upload(rand_blocks(size_t(n_out) * (n_in / 32), bsize_of(type))));
    std::vector<void*> x;
    std::vector<float*> y;
    for (int j = 0; j < count; ++j) x.push_back(upload(rand_q81(n_in))), y.push_back(fill_ff(n_out));
    const double bytes = double(count) * n_out * (n_in / 32) * bsize_of(type);
    auto run = [&](int mode) {
        kernels::native_gemv_vwarp_force(mode);
        return time_it([&](int set) {
            kernels::MmvqSlots sl{};
            for (int j = 0; j < count; ++j) sl.w[j] = w[(size_t(set) * count + j) % w.size()], sl.x[j] = x[j], sl.y[j] = y[j];
            kernels::native_mmvq_slots(type, sl, count, n_in, n_out, st);
        }, pool / count, reps);
    };
    const double o1 = run(0), n1 = run(1), o2 = run(0), n2 = run(1);
    const double o = std::min(o1, o2), nn = std::min(n1, n2);
    std::printf("BENCH slots type %2d %4d -> %4d x%d: old %6.1f us %6.1f GB/s | new %6.1f us %6.1f GB/s | %+.1f%%\n", type, n_in,
                n_out, count, o, bytes / o / 1e3, nn, bytes / nn / 1e3, 100.0 * (o / nn - 1));
    for (auto* p : w) cudaFree(p);
}
void bench_many(int reps) {
    const std::vector<int> outs = {3072, 1024, 1024, 3072, 384, 24};
    const int n_in = 2304, copies = 16;
    std::vector<std::vector<uint8_t*>> w(copies);
    double bytes = 0;
    for (int c = 0; c < copies; ++c)
        for (int o : outs) w[c].push_back(upload(rand_blocks(size_t(o) * (n_in / 32), 34)));
    std::vector<void*> x;
    std::vector<float*> y;
    for (int o : outs) x.push_back(upload(rand_q81(n_in))), y.push_back(fill_ff(o)), bytes += double(o) * (n_in / 32) * 34;
    auto run = [&](int mode) {
        kernels::native_gemv_vwarp_force(mode);
        return time_it([&](int set) {
            kernels::NativeMmvqMany m{};
            m.count = int(outs.size());
            for (int j = 0; j < m.count; ++j) m.w[j] = w[set][j], m.x_q8_1[j] = x[j], m.y[j] = y[j], m.n_out[j] = outs[j];
            kernels::native_mmvq_many(8, m, n_in, st);
        }, copies, reps);
    };
    const double o1 = run(0), n1 = run(1), o2 = run(0), n2 = run(1);
    const double o = std::min(o1, o2), nn = std::min(n1, n2);
    std::printf("BENCH many Q8_0 q|k|v|gate|rel|sink: old %6.1f us %6.1f GB/s | new %6.1f us %6.1f GB/s | %+.1f%%\n", o,
                bytes / o / 1e3, nn, bytes / nn / 1e3, 100.0 * (o / nn - 1));
}
void bench_ctrl(int reps) {
    const int n_embd = 2304, rank = 64, copies = 32;
    std::vector<std::vector<void*>> dw(copies);
    for (int c = 0; c < copies; ++c)
        for (int j = 0; j < 3; ++j) dw[c].push_back(upload(randf(size_t(rank) * 2 * n_embd, 0.02f)));
    float *cur[3], *msg[3], *ch[3];
    for (int j = 0; j < 3; ++j) cur[j] = upload(randf(n_embd, 1.0f)), msg[j] = upload(randf(n_embd, 0.7f)), ch[j] = fill_ff(rank);
    const double bytes = 3.0 * rank * 2 * n_embd * 4;
    auto run = [&](int mode) {
        kernels::native_gemv_vwarp_force(mode);
        return time_it([&](int set) {
            const float* c[3] = {cur[0], cur[1], cur[2]};
            const float* m[3] = {msg[0], msg[1], msg[2]};
            const void* d[3] = {dw[set][0], dw[set][1], dw[set][2]};
            poly::ink_ctrl_down(c, m, d, ch, 3, n_embd, rank, 1, st, false);
        }, copies, reps);
    };
    const double o1 = run(0), n1 = run(1), o2 = run(0), n2 = run(1);
    const double o = std::min(o1, o2), nn = std::min(n1, n2);
    std::printf("BENCH ctrl_down F32 3 x 64 x 4608: old %6.1f us %6.1f GB/s | new %6.1f us %6.1f GB/s | %+.1f%%\n", o,
                bytes / o / 1e3, nn, bytes / nn / 1e3, 100.0 * (o / nn - 1));
}

}  // namespace

int main(int argc, char** argv) {
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || n == 0) {
        std::printf("no card\n");
        return 77;
    }
    ck(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking), "stream");
    kernels::native_gemv_vwarp_force(1);
    if (!kernels::native_gemv_vwarp_active()) {
        std::printf("not a gfx11 / gfx12 HIP device: the virtual-warp kernels are not built for it, nothing to compare\n");
        return 77;
    }
    if (argc > 1 && std::strcmp(argv[1], "--bench") == 0) {
        const int reps = argc > 2 ? std::max(10, std::atoi(argv[2])) : 200;
        for (int type : {20, 8}) bench_slots(type, 2304, 2304, reps), bench_slots(type, 1152, 2304, reps);
        bench_many(reps);
        bench_ctrl(reps);
        return 0;
    }
    for (int type : {20, 8}) {
        for (int count : {1, 5, 8, 32}) {
            test_slots(type, 2304, 2304, count);   // gate|up
            test_slots(type, 1152, 2304, count);   // down
        }
        test_slots(type, 2304, 1152, 8);   // the up of a gate / up pair (UD-Q2's mixed layers)
        test_slots(type, 2304, 2303, 3);   // an odd row count: the tails of the row groups
        test_slots(type, 1152, 2301, 3);
        test_slots(type, 2304, 7, 2);
        test_slots(type, 1152, 1, 2);
        test_ids(type, 2304, 2304, 8);
        test_ids(type, 1152, 2304, 8);
        test_ids(type, 1152, 2303, 5);
    }
    test_many(8, 2304, {3072, 1024, 1024, 3072, 384, 24});   // q|k|v|gate|rel|sink
    test_many(8, 2304, {3072, 1024, 1024});
    test_many(8, 2304, {2304});
    test_many(8, 2304, {7, 1001, 3, 2});   // odd totals and row counts
    test_many(8, 2304, {1});
    test_many(8, 3072, {2304});   // attn_output's n_in: the block kernel
    test_many(20, 2304, {1024, 384});   // other formats are untouched
    test_many(2, 2304, {1024, 384});
    for (bool bf16 : {false, true})
        for (int count : {1, 2, 3}) {
            test_ctrl_down(bf16, count, 2304, 64, 1);
            test_ctrl_down(bf16, count, 2304, 64, 3);
        }
    test_ctrl_down(false, 3, 2304, 62, 2);   // an even rank that is not 64
    test_ctrl_down(false, 3, 2304, 63, 2);   // an odd rank: the block kernel
    std::printf("gemv_vwarp_test: %d cases, %d FAILED\n", g_cases, g_fail);
    return g_fail == 0 ? 0 : 1;
}
