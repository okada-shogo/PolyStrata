// src/poly/i8m_shapes_test.cu - the int8 prompt kernel (kernels/cuda/i8_mma.cuh) in each tile shape a card may be given
// (the CF bits of i8m_kernel: threads, tiles a warp, the sums' conversion; and the tile's width in activation rows)
// against the shape it had before, bit for bit, for Inkling 2 Mini's products: its experts' gate|up (2,304 values a
// row) and downs (1,152) of 128 experts with the rows of 1,024 tokens' top-8 picks, launched by class of rows as the
// engine does (up to 32 rows, up to 128, more), and its dense attention products (Q8_0, 1,024 rows).  The weights are
// random blocks with sane scales, the activations quantized by the kernel's own quantizer.
//
// "the launcher's pick" is i8m_launch_type() itself (on Ampere and newer the raw stage copy where it takes it; on
// Turing i8m_cf75()), so every other shape is also checked against what a card runs by default.
// usage: poly_i8m_shapes_test [--time]   (--time: also each shape's time on this card, the best of 3 runs of 10;
// exit 0: every shape gives the same bits; 77: no card or no int8 MMA)
#if !defined(__HIPCC__)
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/dp4a.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#define GGML_COMMON_DECL_CUDA
#define GGML_COMMON_IMPL_CUDA
#include "ggml-common.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <numeric>
#include <random>
#include <string>
#include <type_traits>
#include <vector>

#define ggml_cuda_dp4a(a, b, c) STRATA_DP4A((a), (b), (c))

namespace strata::kernels {
namespace {
#include "i8_expert.cuh"
#include "i8_mma.cuh"
}  // namespace
}  // namespace strata::kernels

using namespace strata::kernels;

namespace {

bool ok(cudaError_t e, const char* what) {
    if (e != cudaSuccess) std::printf("%s: %s\n", what, cudaGetErrorString(e));
    return e == cudaSuccess;
}

// bytes and values of a type's block, and where its FP16 scale is (-1: none)
struct Blk { int bytes, vals, d_at; };
Blk blk(int t) {
    switch (t) {
        case 20: return {18, 32, 0};    // IQ4_NL
        case 8: return {34, 32, 0};     // Q8_0
        case 16: return {66, 256, 0};   // IQ2_XXS
        case 17: return {74, 256, 0};   // IQ2_XS
        case 22: return {82, 256, 0};   // IQ2_S
        case 19: return {50, 256, 0};   // IQ1_S
        case 40: return {36, 64, -1};   // NVFP4 (UE4M3 scales: any byte is a number)
        default: return {0, 0, 0};
    }
}

struct Launch { int nj, n_mats, first, most; };   // a class's launch: its tile width, experts, first expert, most rows

// a problem: weights, the rows' quantized activations, the classes' launches
struct Problem {
    int T, n_out, K;
    size_t row_bytes, mat_bytes;
    int n_distinct;                  // matrices stored; expert e uses matrix e % n_distinct (by offset, as mul_at())
    uint8_t* W = nullptr;
    uint8_t* xq = nullptr;
    long long* at = nullptr;         // per expert, in launch order
    int32_t* bounds = nullptr;       // per expert, the first row (and the end)
    int32_t* ids = nullptr;          // scattered: the row of Y of each launch row, and the row of X it reads (from)
    int32_t* from = nullptr;
    std::vector<Launch> launches;
    int64_t rows = 0;
    float* Y = nullptr;
};

// the tile width the engine's launcher gives a launch's most rows
int nj_of(int most) { return most <= 16 ? 1 : most <= 32 ? 2 : most <= 64 ? 4 : 8; }

// scatter: the rows read by index and written by index (ids, from: two shuffles, as a call's rows of several tokens),
// and the experts without rows inside the launches (bounds that are equal), not left out
bool make(Problem& p, int T, int n_out, int K, const std::vector<int>& rows_of, int n_distinct, uint32_t seed, bool scatter = false) {
    p.T = T; p.n_out = n_out; p.K = K;
    const Blk b = blk(T);
    p.row_bytes = (size_t) K / b.vals * b.bytes;
    p.mat_bytes = p.row_bytes * n_out;
    p.n_distinct = n_distinct;
    std::mt19937 rng(seed);
    std::vector<uint8_t> w(p.mat_bytes * n_distinct);
    for (auto& v : w) v = (uint8_t) rng();
    if (b.d_at >= 0) {
        std::uniform_real_distribution<float> d(0.002f, 0.02f);
        for (size_t o = 0; o + b.bytes <= w.size(); o += b.bytes) {
            const __half h = __float2half(d(rng));
            std::memcpy(&w[o + b.d_at], &h, 2);
        }
    }
    // experts by class (up to 32 rows, up to 128, more), as the engine lays them out; the empty ones in no launch
    const int n_e = (int) rows_of.size();
    const auto cls = [&](int r) { return scatter && r == 0 ? 1 : r <= 32 ? 0 : r <= 128 ? 1 : 2; };
    std::vector<int> order;
    for (int c = 0; c < 3; ++c)
        for (int e = 0; e < n_e; ++e) {
            const int r = rows_of[(size_t) e];
            if ((r > 0 || scatter) && cls(r) == c) order.push_back(e);
        }
    std::vector<int32_t> bounds(order.size() + 1, 0);
    std::vector<long long> at(order.size());
    for (size_t i = 0; i < order.size(); ++i) {
        bounds[i + 1] = bounds[i] + rows_of[(size_t) order[i]];
        at[i] = (long long) ((size_t) (order[i] % n_distinct) * p.mat_bytes);
    }
    p.rows = bounds.back();
    for (size_t i = 0; i < order.size();) {
        const int r = rows_of[(size_t) order[i]], c = cls(r);
        size_t j = i;
        int most = 0;
        for (; j < order.size(); ++j) {
            const int rj = rows_of[(size_t) order[j]];
            if (cls(rj) != c) break;
            most = std::max(most, rj);
        }
        p.launches.push_back({nj_of(most), (int) (j - i), (int) i, most});
        i = j;
    }
    // activations: normal values with a few channels 30 times larger
    std::vector<float> x((size_t) p.rows * K);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    for (int64_t r = 0; r < p.rows; ++r)
        for (int k = 0; k < K; ++k) x[(size_t) r * K + k] = nd(rng) * (k % 97 == 5 ? 30.0f : 1.0f);
    float* dx = nullptr;
    const size_t xq_bytes = (size_t) p.rows * K + 2 * (size_t) p.rows * (K / 32) * sizeof(float);
    if (!ok(cudaMalloc(&p.W, w.size()), "cudaMalloc") || !ok(cudaMalloc(&p.xq, xq_bytes), "cudaMalloc") ||
        !ok(cudaMalloc(&dx, x.size() * sizeof(float)), "cudaMalloc") || !ok(cudaMalloc(&p.at, at.size() * sizeof(long long) + 8), "cudaMalloc") ||
        !ok(cudaMalloc(&p.bounds, bounds.size() * sizeof(int32_t)), "cudaMalloc") ||
        !ok(cudaMalloc(&p.Y, (size_t) p.rows * n_out * sizeof(float)), "cudaMalloc"))
        return false;
    cudaMemcpy(p.W, w.data(), w.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(dx, x.data(), x.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(p.at, at.data(), at.size() * sizeof(long long), cudaMemcpyHostToDevice);
    cudaMemcpy(p.bounds, bounds.data(), bounds.size() * sizeof(int32_t), cudaMemcpyHostToDevice);
    if (scatter) {
        std::vector<int32_t> ids((size_t) p.rows), from((size_t) p.rows);
        std::iota(ids.begin(), ids.end(), 0);
        std::iota(from.begin(), from.end(), 0);
        std::shuffle(ids.begin(), ids.end(), rng);
        std::shuffle(from.begin(), from.end(), rng);
        if (!ok(cudaMalloc(&p.ids, ids.size() * sizeof(int32_t)), "cudaMalloc") || !ok(cudaMalloc(&p.from, from.size() * sizeof(int32_t)), "cudaMalloc"))
            return false;
        cudaMemcpy(p.ids, ids.data(), ids.size() * sizeof(int32_t), cudaMemcpyHostToDevice);
        cudaMemcpy(p.from, from.data(), from.size() * sizeof(int32_t), cudaMemcpyHostToDevice);
    }
    gemm_iq_i8_quantize(dx, nullptr, K, p.rows, K, p.xq, nullptr);
    cudaFree(dx);
    return ok(cudaDeviceSynchronize(), "quantize");
}

void release(Problem& p) {
    for (void* v : {(void*) p.W, (void*) p.xq, (void*) p.at, (void*) p.bounds, (void*) p.Y, (void*) p.ids, (void*) p.from}) cudaFree(v);
    p = Problem{};
}

bool g_occupancy_ok = true;

// one launch of the kernel in shape (NJ, CF) over a class's experts
template <int T, int NJ, int CF>
bool launch(const Problem& p, const Launch& l) {
    constexpr size_t bytes = I8mSmem<T, NJ, 64, false, i8m_ks(CF), (CF & I8M_CF_SWZ) != 0>::bytes;
    static bool attr = false;
    if (!attr) {
        if (!ok(cudaFuncSetAttribute(i8m_kernel<T, NJ, 64, false, CF>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) bytes), "attribute"))
            return false;
        if (CF & I8M_CF_K128) cudaFuncSetAttribute(i8m_kernel<T, NJ, 64, false, CF>, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
        int per_sm = 0;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, i8m_kernel<T, NJ, 64, false, CF>, i8m_nt(T, NJ, 64, CF), bytes);
        std::printf("    [type %d, tile %d rows, CF %d: %zu bytes of shared memory, %d threads, %d blocks a multiprocessor]\n", T,
                    16 * NJ, CF, bytes, i8m_nt(T, NJ, 64, CF), per_sm);
        // Turing's shapes are there for two blocks a multiprocessor: fewer is a failure (the same bits, the speed lost)
        if (CF != 0 && CF == i8m_cf75(T, NJ, 64) && i8m_card().arch < 800 && per_sm < 2) {
            std::printf("    FAILED: type %d, tile %d rows, CF %d fits %d block a multiprocessor, not 2\n", T, 16 * NJ, CF, per_sm);
            g_occupancy_ok = false;
        }
        attr = true;
    }
    const int ch = (l.most + 16 * NJ - 1) / (16 * NJ);
    const dim3 grid((unsigned) ((p.n_out + 63) / 64), (unsigned) (l.n_mats * ch));
    const float* XD = (const float*) (p.xq + (size_t) p.rows * p.K);
    const float* XS = XD + (size_t) p.rows * (p.K / 32);
    i8m_kernel<T, NJ, 64, false, CF><<<grid, i8m_nt(T, NJ, 64, CF), bytes>>>(
        p.W, p.mat_bytes, p.row_bytes, p.n_out, p.K, p.xq, XD, XS, p.bounds + l.first, ch, p.ids, p.from, p.at + l.first, p.Y,
        p.n_out);
    return ok(cudaGetLastError(), "i8m_kernel");
}

using Fn = bool (*)(const Problem&, const Launch&);
template <int T, int NJ, int CF>
Fn one() {
    if constexpr (i8m_shape_ok(T, NJ, 64, CF)) return launch<T, NJ, CF>;
    else return nullptr;
}
// (tiles of 16 and 64 rows only in the shapes that use them: fewer kernels to compile)
template <int T, int CF>
Fn by_nj(int nj) {
    constexpr bool narrow = CF == 0 || CF == I8M_CF_CVT2 || CF == (I8M_CF_K128 | I8M_CF_T256 | I8M_CF_CVT2);
    if constexpr (narrow) {
        if (nj == 1) return one<T, 1, CF>();
        if (nj == 4) return one<T, 4, CF>();
    }
    return nj == 2 ? one<T, 2, CF>() : nj == 8 ? one<T, 8, CF>() : nullptr;
}

// A shape: CF, and the most activation rows a tile has (8: as the launcher picks, 4 / 2: no wider than 64 / 32)
struct Shape { const char* name; int cf, cap; };
constexpr int K128 = I8M_CF_K128 | I8M_CF_T256;
constexpr Shape kShapes[] = {
    {"as before", 0, 8},
    {"Turing's (i8m_cf75)", -1, 8},
    {"the launcher's pick", -2, 8},
    {"cvt", I8M_CF_CVT, 8},
    {"cvt2", I8M_CF_CVT2, 8},
    {"256 threads cvt2", I8M_CF_T256 | I8M_CF_CVT2, 8},
    {"na1", I8M_CF_NA1, 8},
    {"tile 64 rows cvt2", I8M_CF_CVT2, 4},
    {"k128", K128, 8},
    {"k128 cvt", K128 | I8M_CF_CVT, 8},
    {"k128 cvt2", K128 | I8M_CF_CVT2, 8},
    {"k128 tile 64 cvt2", K128 | I8M_CF_CVT2, 4},
    {"swz cvt2", I8M_CF_SWZ | I8M_CF_CVT2, 8},
    {"k128 swz", K128 | I8M_CF_SWZ, 8},
    {"k128 swz cvt2", K128 | I8M_CF_SWZ | I8M_CF_CVT2, 8},
};

template <int T>
Fn pick(int cf, int nj) {
    switch (cf) {
#define S8_CASE(C) case C: return by_nj<T, C>(nj);
        S8_CASE(0) S8_CASE(I8M_CF_CVT) S8_CASE(I8M_CF_CVT2) S8_CASE(I8M_CF_T256 | I8M_CF_CVT2) S8_CASE(I8M_CF_NA1)
        S8_CASE(K128) S8_CASE(K128 | I8M_CF_CVT) S8_CASE(K128 | I8M_CF_CVT2) S8_CASE(I8M_CF_SWZ | I8M_CF_CVT2)
        S8_CASE(K128 | I8M_CF_SWZ) S8_CASE(K128 | I8M_CF_SWZ | I8M_CF_CVT2)
#undef S8_CASE
        default: return nullptr;
    }
}
Fn pick_type(int T, int cf, int nj) {
    switch (T) {
        case 20: return pick<20>(cf, nj);
        case 8: return pick<8>(cf, nj);
        case 16: return pick<16>(cf, nj);
        case 17: return pick<17>(cf, nj);
        case 22: return pick<22>(cf, nj);
        case 40: return pick<40>(cf, nj);
        default: return nullptr;
    }
}

// all of a problem's launches in a shape (only = i: launch i alone); false: a launch failed
bool run(const Problem& p, const Shape& s, int only = -1) {
    for (size_t i = 0; i < p.launches.size(); ++i) {
        if (only >= 0 && (size_t) only != i) continue;
        const Launch& l = p.launches[i];
        if (s.cf == -2) {   // the launcher's own pick for this card (on Ampere and newer: the raw stage copy where it takes it)
            const float* XD = (const float*) (p.xq + (size_t) p.rows * p.K);
            int dev = 0;
            cudaGetDevice(&dev);
            bool done = false;
            switch (p.T) {
#define S8_LAUNCHER(T) \
                case T: done = i8m_launch_type<T>(i8m_card(), dev, nullptr, p.W, p.mat_bytes, p.row_bytes, p.n_out, p.K, p.xq, XD, \
                                                  XD + (size_t) p.rows * (p.K / 32), p.bounds + l.first, l.n_mats, l.most, p.ids, \
                                                  p.from, p.at + l.first, p.Y, p.n_out); break;
                S8_LAUNCHER(20) S8_LAUNCHER(8) S8_LAUNCHER(16) S8_LAUNCHER(17) S8_LAUNCHER(22) S8_LAUNCHER(40)
#undef S8_LAUNCHER
                default: break;
            }
            if (!done) return false;
            continue;
        }
        const int nj = std::min(l.nj, s.cap), cf = s.cf >= 0 ? s.cf : i8m_cf75(p.T, nj, 64);
        const Fn f = pick_type(p.T, cf, nj);
        if (!f) {   // (a shape not compiled for this tile: a failure, not a launch of another shape)
            std::printf("  no kernel for type %d, tile %d rows, CF %d\n", p.T, 16 * nj, cf);
            return false;
        }
        if (!f(p, l)) return false;
    }
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    const bool timing = argc > 1 && std::strcmp(argv[1], "--time") == 0;
    int cards = 0;
    if (cudaGetDeviceCount(&cards) != cudaSuccess || cards == 0) {
        std::puts("poly_i8m_shapes_test: no card, skipped");
        return 77;
    }
    gemm_iq_i8_turing(true);                                       // the library's quantizer
    strata::kernels::i8m_turing.store(true, std::memory_order_relaxed);   // and this file's copy of the kernel
    if (!i8m_card().on) {
        std::puts("poly_i8m_shapes_test: no int8 MMA on this card's code, skipped");
        return 77;
    }
    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, 0);
    std::printf("card: %s, sm_%d%d, %d multiprocessors\n", prop.name, prop.major, prop.minor, prop.multiProcessorCount);

    // 1,024 tokens' top-8 of 128 experts, the experts' popularity skewed as a layer's is (median ~50 rows, a few over 300)
    std::mt19937 rng(7);
    std::vector<double> pop(128);
    for (int e = 0; e < 128; ++e) pop[(size_t) e] = 1.0 / std::pow((double) (e + 4), 0.9);
    std::shuffle(pop.begin(), pop.end(), rng);
    std::vector<int> rows_of(128, 0);
    for (int t = 0; t < 1024; ++t) {
        std::vector<double> w = pop;
        for (int k = 0; k < 8; ++k) {
            std::discrete_distribution<int> pick_e(w.begin(), w.end());
            const int e = pick_e(rng);
            ++rows_of[(size_t) e];
            w[(size_t) e] = 0.0;
        }
    }
    {
        std::vector<int> s = rows_of;
        std::sort(s.begin(), s.end());
        std::printf("rows a expert: min %d, median %d, max %d\n", s.front(), s[64], s.back());
    }
    struct Case { const char* name; int T, n_out, K; bool dense; bool scatter = false; };
    const Case cases[] = {
        {"IQ4_NL experts gate|up", 20, 2304, 2304, false}, {"IQ4_NL experts down", 20, 2304, 1152, false},
        {"Q8_0 experts gate|up", 8, 2304, 2304, false},    {"Q8_0 dense [2304 -> 3072]", 8, 3072, 2304, true},
        {"Q8_0 dense [2304 -> 1024]", 8, 1024, 2304, true}, {"IQ2_S experts gate|up", 22, 2304, 2304, false},
        {"IQ2_XXS experts gate|up", 16, 2304, 2304, false}, {"IQ2_XS experts gate|up", 17, 2304, 2304, false},
        {"NVFP4 experts down", 40, 2304, 1152, false},       {"NVFP4 experts gate|up", 40, 2304, 2304, false},
        // rows that end inside a stage of 128 (Gemma 4 26B's downs: 704 values), outputs not a whole tile of 64, and
        // the rows read and written by index with experts without rows among them
        {"IQ4_NL K = 704", 20, 2304, 704, false},             {"Q8_0 K = 704", 8, 2304, 704, false},
        {"NVFP4 K = 1088", 40, 2304, 1088, false},             {"IQ4_NL 2280 outputs", 20, 2280, 2304, false},
        {"IQ2_XXS 2280 outputs", 16, 2280, 2304, false},
        {"IQ4_NL scattered", 20, 2304, 2304, false, true},      {"IQ2_S scattered", 22, 2304, 1152 * 2, false, true},
        {"NVFP4 down scattered", 40, 2304, 1152, false, true},
    };
    bool all = true;
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    for (const Case& c : cases) {
        Problem p;
        const std::vector<int> dense_rows = {1024};
        std::vector<int> rows_s = rows_of;   // scattered: two experts without rows
        rows_s[5] = rows_s[77] = 0;
        if (!make(p, c.T, c.n_out, c.K, c.dense ? dense_rows : c.scatter ? rows_s : rows_of, c.dense ? 1 : 16, 11u + (uint32_t) c.T, c.scatter))
            return 1;
        const double macs = (double) p.rows * c.n_out * c.K;
        std::vector<uint32_t> ref((size_t) p.rows * c.n_out), got(ref.size());
        std::printf("%s (%lld rows, %zu launches):\n", c.name, (long long) p.rows, p.launches.size());
        {   // the launcher's first tests for the raw stage copy (it also wants the tile to fit as many blocks a multiprocessor)
            const bool raw = i8m_card().arch >= 800 && i8m_raw_type(c.T) && c.K % 256 == 0 &&
                             (i8m_raw_bytes(c.T) % 16 != 0 || (p.row_bytes % 16 == 0 && p.mat_bytes % 16 == 0));
            std::printf("  (the launcher's pick: %s)\n", i8m_card().arch < 800 ? "i8m_cf75() on Turing" : raw ? "the raw stage copy, if the tile fits" : "no raw stage copy");
        }
        float base_ms = 0;
        for (const Shape& s : kShapes) {
            if ((s.cf > 0 || s.cf == -1) && i8m_card().arch >= 800) {   // (compiled empty there: Turing's only)
                if (!timing) std::printf("  %-20s not compiled for this card\n", s.name);
                continue;
            }
            cudaMemset(p.Y, 0xFF, got.size() * sizeof(float));
            if (!run(p, s) || !ok(cudaDeviceSynchronize(), s.name)) { all = false; continue; }
            cudaMemcpy(got.data(), p.Y, got.size() * sizeof(float), cudaMemcpyDeviceToHost);
            bool same = true;
            if (s.cf == 0 && s.cap == 8) ref = got;
            else same = std::memcmp(ref.data(), got.data(), ref.size() * sizeof(uint32_t)) == 0;
            size_t diff = 0;
            if (!same)
                for (size_t i = 0; i < ref.size(); ++i) diff += ref[i] != got[i];
            all = all && same;
            if (!timing) {
                std::printf("  %-20s %s", s.name, same ? "same bits\n" : "DIFFERENT");
                if (!same) std::printf(" (%zu of %zu values)\n", diff, ref.size());
                continue;
            }
            float best = 1e30f;
            for (int rep = 0; rep < 3; ++rep) {
                cudaEventRecord(e0);
                for (int i = 0; i < 10; ++i) run(p, s);
                cudaEventRecord(e1);
                cudaEventSynchronize(e1);
                float ms = 0;
                cudaEventElapsedTime(&ms, e0, e1);
                best = std::min(best, ms / 10);
            }
            if (s.cf == 0 && s.cap == 8) base_ms = best;
            std::printf("  %-22s %s  %8.1f us  %5.1f T MAC/s  %+6.1f%%  by launch:", s.name, same ? "same bits" : "DIFFERENT", best * 1e3,
                        macs / (best * 1e-3) / 1e12, base_ms > 0 ? (base_ms / best - 1) * 100 : 0.0);
            for (size_t li = 0; li < p.launches.size(); ++li) {   // each class's launch alone: its most rows, its time
                float one = 1e30f;
                for (int rep = 0; rep < 3; ++rep) {
                    cudaEventRecord(e0);
                    for (int i = 0; i < 10; ++i) run(p, s, (int) li);
                    cudaEventRecord(e1);
                    cudaEventSynchronize(e1);
                    float ms = 0;
                    cudaEventElapsedTime(&ms, e0, e1);
                    one = std::min(one, ms / 10);
                }
                std::printf("  [%d experts, up to %d rows: %.1f us]", p.launches[li].n_mats, p.launches[li].most, one * 1e3);
            }
            std::printf("\n");
        }
        release(p);
    }
    all = all && g_occupancy_ok;
    std::printf("poly_i8m_shapes_test: %s\n", all ? "every shape gives the same bits" : "FAILED");
    return all ? 0 : 1;
}
#else
int main() { return 77; }
#endif
