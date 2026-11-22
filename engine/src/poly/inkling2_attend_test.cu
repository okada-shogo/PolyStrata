// poly_inkling2_attend_test: Inkling 2's token attention (ink_attend(), the score/softmax/value/join kernels) on
// made-up caches of Inkling 2 Mini's shape (24 query heads on 8 key heads of 128 values, 16 relative values).
//   hash <file>   a battery of calls (fp16, q8_0, and FP8 where the card takes it, full and window layers, 1 to 7
//                 rows, positions on the host and on the card, gate and q8 output on and off): one FNV-1a hash of each
//                 call's output bytes a line.  Two builds' files differ where their outputs differ; equal files mean the same bits
//                 (but for a hash collision).  Exit 1 if a call's output is not finite.
//   bench         one token of a full layer, rooms 32K to 1M: microseconds a call and the caches' bytes a second.
// Needs a card; run by hand.
#include "inkling2_kernels.hpp"

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

using namespace strata::poly;

namespace {

void ck(cudaError_t e, const char* w) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "CUDA error %s: %s\n", w, cudaGetErrorString(e));
        std::exit(1);
    }
}

constexpr int kHeads = 24, kKv = 8, kDim = 128, kRel = 16;
std::mt19937 rng(77);

std::vector<float> randf(size_t n, float sd) {
    std::normal_distribution<float> d(0.0f, sd);
    std::vector<float> v(n);
    for (auto& x : v) x = d(rng);
    return v;
}
template <class T> T* up(const std::vector<T>& v) {
    T* d = nullptr;
    ck(cudaMalloc(&d, v.size() * sizeof(T) + 16), "malloc");
    ck(cudaMemcpy(d, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice), "upload");
    return d;
}

enum Fmt { kF16, kF8, kQ8 };
const char* fmt_name(Fmt f) { return f == kF16 ? "fp16" : f == kF8 ? "fp8" : "q8_0"; }
size_t row_bytes(Fmt f) {
    return f == kF16 ? (size_t) kKv * kDim * 2 : f == kF8 ? ink_kv8_row(kKv, kDim) : ink_kvq8_row(kKv, kDim);
}

// A cache of `rows` rows in format f, random values of about unit size.
uint8_t* make_cache(Fmt f, size_t rows) {
    const size_t rb = row_bytes(f);
    std::vector<uint8_t> h(rows * rb);
    if (f == kF16) {
        const auto v = randf(rows * kKv * kDim, 1.0f);
        for (size_t i = 0; i < v.size(); ++i) {
            const uint16_t b = __half_as_ushort(__float2half(v[i]));
            std::memcpy(&h[2 * i], &b, 2);
        }
    } else if (f == kQ8) {   // Q8_0 codes, then a half scale a block of 32
        std::uniform_int_distribution<int> code(-127, 127);
        std::uniform_real_distribution<float> sc(0.004f, 0.012f);
        for (size_t r = 0; r < rows; ++r) {
            uint8_t* p = &h[r * rb];
            for (int i = 0; i < kKv * kDim; ++i) p[i] = (uint8_t) (int8_t) code(rng);
            for (int b = 0; b < kKv * kDim / 32; ++b) {
                const uint16_t s = __half_as_ushort(__float2half(sc(rng)));
                std::memcpy(p + kKv * kDim + 2 * b, &s, 2);
            }
        }
    } else {   // E4M3 codes without NaN (0x7f, 0xff), then a float scale a head
        std::uniform_int_distribution<int> code(0, 253);
        std::uniform_real_distribution<float> sc(0.002f, 0.01f);
        for (size_t r = 0; r < rows; ++r) {
            uint8_t* p = &h[r * rb];
            for (int i = 0; i < kKv * kDim; ++i) {
                int c = code(rng);
                if (c >= 0x7f) ++c;
                p[i] = (uint8_t) c;
            }
            for (int g = 0; g < kKv; ++g) {
                const float s = sc(rng);
                std::memcpy(p + kKv * kDim + 4 * g, &s, 4);
            }
        }
    }
    return up(h);
}

struct Bufs {
    float *q, *r, *sink, *bias_full, *bias_win, *gate, *out, *score;
    void* q8;
    long long* at;
    size_t score_n;
};

constexpr int kMaxRows = 8;

// scores for `rows` rows of `keys` keys (the bench takes one row)
Bufs make_bufs(int64_t keys, int rows = kMaxRows) {
    Bufs b{};
    b.q = up(randf((size_t) kMaxRows * kHeads * kDim, 1.5f));
    b.r = up(randf((size_t) kMaxRows * kHeads * kRel, 1.0f));
    b.sink = up(randf((size_t) kMaxRows * kHeads, 1.0f));
    b.bias_full = up(randf((size_t) 1024 * kRel, 0.5f));
    b.bias_win = up(randf((size_t) 1024 * kRel, 0.5f));
    b.gate = up(randf((size_t) kMaxRows * kHeads * kDim, 1.0f));
    ck(cudaMalloc(&b.out, (size_t) kMaxRows * kHeads * kDim * 4), "malloc");
    ck(cudaMalloc(&b.q8, (size_t) kHeads * kDim / 32 * 36 + 64), "malloc");
    ck(cudaMalloc(&b.at, (size_t) 2 * kMaxRows * sizeof(long long)), "malloc");
    b.score_n = ink_attend_floats(kHeads, kKv, kDim, keys) * rows;
    ck(cudaMalloc(&b.score, b.score_n * 4), "malloc");
    return b;
}

InkAttn base(const Bufs& b, const uint8_t* k, const uint8_t* v, Fmt f) {
    InkAttn a;
    a.q = b.q;
    a.r = b.r;
    a.sink = b.sink;
    a.keys = (const uint16_t*) k;
    a.vals = (const uint16_t*) v;
    a.out = b.out;
    a.score = b.score;
    a.score_n = b.score_n;
    a.n_head = kHeads;
    a.n_kv = kKv;
    a.dim = kDim;
    a.n_rel = kRel;
    a.scale = 1.0f / std::sqrt((float) kDim);
    a.kv8 = f == kF8;
    a.kvq8 = f == kQ8;
    return a;
}

uint64_t fnv(const void* p, size_t n, uint64_t x = 1469598103934665603ull) {
    const uint8_t* c = (const uint8_t*) p;
    for (size_t i = 0; i < n; ++i) x = (x ^ c[i]) * 1099511628211ull;
    return x;
}

int run_hash(const char* path) {
    FILE* f = std::fopen(path, "w");
    if (!f) return 2;
    const int64_t ctx = 40000, ring = 1024 + kMaxRows;
    Bufs b = make_bufs(ctx + kMaxRows);
    std::vector<Fmt> fmts = {kF16, kQ8};
    if (ink_kv8_ok()) fmts.push_back(kF8);
    const int64_t poss[] = {0, 1, 5, 30, 31, 32, 33, 100, 1000, 1022, 1023, 1024, 1025, 2047, 4095, 4100, 32766,
                            32767, 32768, 32769, 39990};
    const int rowss[] = {1, 2, 3, 7};
    std::vector<float> h((size_t) kMaxRows * kHeads * kDim);
    std::vector<uint8_t> hq((size_t) kHeads * kDim / 32 * 36);
    int cases = 0, nonfinite = 0;
    for (Fmt fm : fmts) {
        uint8_t *kf = make_cache(fm, ctx + kMaxRows), *vf = make_cache(fm, ctx + kMaxRows);
        uint8_t *kw = make_cache(fm, ring), *vw = make_cache(fm, ring);
        for (int win = 0; win < 2; ++win)
            for (int64_t pos : poss)
                for (int rows : rowss)
                    for (int mode = 0; mode < 4; ++mode) {   // 0 plain, 1 gate, 2 positions on the card, 3 q8 out
                        if (mode == 3 && rows != 1) continue;
                        InkAttn a = win ? base(b, kw, vw, fm) : base(b, kf, vf, fm);
                        a.rel_bias = win ? b.bias_win : b.bias_full;
                        a.extent = 1024;
                        a.window = win ? 1024 : 0;
                        a.ring = win ? (int) ring : 0;
                        a.pos = pos;
                        a.rows = rows;
                        if (mode == 1 || mode == 3) a.gate = b.gate;
                        if (mode == 3) a.q8 = b.q8;
                        std::vector<long long> at(2 * kMaxRows, 0);
                        if (mode == 2) {
                            for (int t = 0; t < rows; ++t) at[2 * t] = pos + t;
                            ck(cudaMemcpy(b.at, at.data(), at.size() * sizeof(long long), cudaMemcpyHostToDevice),
                               "at");
                            a.at = b.at;
                            a.room = (int) std::min<int64_t>(ctx, pos + rows + 3);
                        }
                        ck(cudaMemset(b.out, 0, h.size() * 4), "memset");
                        ink_attend(a, nullptr);
                        ck(cudaDeviceSynchronize(), "run");
                        ck(cudaMemcpy(h.data(), b.out, (size_t) rows * kHeads * kDim * 4, cudaMemcpyDeviceToHost),
                           "download");
                        uint64_t x = fnv(h.data(), (size_t) rows * kHeads * kDim * 4);
                        if (mode == 3) {
                            ck(cudaMemcpy(hq.data(), b.q8, hq.size(), cudaMemcpyDeviceToHost), "download");
                            x = fnv(hq.data(), hq.size(), x);
                        }
                        bool finite = true;
                        for (size_t i = 0; i < (size_t) rows * kHeads * kDim; ++i) finite = finite && std::isfinite(h[i]);
                        std::fprintf(f, "%s %s pos %lld rows %d mode %d %016llx%s\n", fmt_name(fm),
                                     win ? "window" : "full", (long long) pos, rows, mode, (unsigned long long) x,
                                     finite ? "" : " NONFINITE");
                        nonfinite += !finite;
                        ++cases;
                    }
        (void) cudaFree(kf), (void) cudaFree(vf), (void) cudaFree(kw), (void) cudaFree(vw);
    }
    std::fclose(f);
    std::printf("hash: %d calls written to %s\n", cases, path);
    if (nonfinite) {
        std::fprintf(stderr, "hash: %d calls gave a value that is not finite (NONFINITE in the file)\n", nonfinite);
        return 1;
    }
    return 0;
}

int run_bench() {
    const int64_t rooms[] = {32768, 131072, 262144, 524288, 1048576};
    size_t free_b = 0, total_b = 0;
    ck(cudaMemGetInfo(&free_b, &total_b), "meminfo");
    std::vector<Fmt> fmts = {kF16, kQ8};
    if (ink_kv8_ok()) fmts.push_back(kF8);
    for (Fmt fm : fmts) {
        int64_t most = 0;
        for (int64_t r : rooms)
            if ((double) r * row_bytes(fm) * 2 < (double) free_b * 0.8) most = r;
        if (!most) continue;
        Bufs b = make_bufs(most, 1);
        uint8_t *k = make_cache(fm, most), *v = make_cache(fm, most);
        cudaEvent_t e0, e1;
        ck(cudaEventCreate(&e0), "event"), ck(cudaEventCreate(&e1), "event");
        for (int64_t room : rooms) {
            if (room > most) break;
            InkAttn a = base(b, k, v, fm);
            a.rel_bias = b.bias_full;
            a.extent = 1024;
            a.pos = room - 1;
            a.rows = 1;
            a.gate = b.gate;
            a.q8 = b.q8;
            for (int i = 0; i < 3; ++i) ink_attend(a, nullptr);
            const int reps = room >= 524288 ? 20 : 50;
            ck(cudaEventRecord(e0), "event");
            for (int i = 0; i < reps; ++i) ink_attend(a, nullptr);
            ck(cudaEventRecord(e1), "event");
            ck(cudaEventSynchronize(e1), "bench");
            float ms = 0.0f;
            ck(cudaEventElapsedTime(&ms, e0, e1), "event");
            const double us = 1000.0 * ms / reps, bytes = (double) room * row_bytes(fm) * 2;
            std::printf("bench %s full room %lld: %.1f us a call, %.0f GB/s of the caches\n", fmt_name(fm),
                        (long long) room, us, bytes / us / 1000.0);
        }
        (void) cudaFree(k), (void) cudaFree(v);
    }
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || n == 0) {
        std::printf("no card: skipped\n");
        return 77;
    }
    if (argc >= 3 && std::string(argv[1]) == "hash") return run_hash(argv[2]);
    if (argc >= 2 && std::string(argv[1]) == "bench") return run_bench();
    std::fprintf(stderr, "usage: poly_inkling2_attend_test hash <file> | bench\n");
    return 2;
}
