// src/poly/i8m_dp4a_test.cu - the int8 prompt kernel's products by dp4a (gemm_iq_i8_dp4a(1), i8_mma.cuh's multiply_dp:
// Turing's code on a GeForce GTX 16, which has no tensor cores) against the same kernel's MMAs, bit for bit, for every
// type the kernel takes, with rows of 2,048 values and of 1,152 (Inkling 2 Mini's experts' downs), each tile width
// (16 to 128 activation rows a block: launches of up to 10, 30, 60 and 300 rows a matrix), matrices of 200 outputs (a
// last tile of 8 rows) and 512 (IQ3_XXS's 128-row tile), experts with uneven and empty row ranges, products scattered
// to other rows.  Each way is also timed (microseconds a launch, the card's warm best of 10).  The weights are ggml's
// quantization of random values; the activations normal values with a few channels 30 times larger.
//
// usage: poly_i8m_dp4a_test   (exit 0: every product the same bits both ways; 1: one differs; 77: no card, or no
// Turing code on it: dp4a is taken only there.)  STRATA_I8M_DP4A_LOG=1 prints the card's own pick's probe.
#include "ggml.h"
#include "strata/kernels/iq_kernels.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace kn = strata::kernels;

namespace {

bool ok(cudaError_t e, const char* what) {
    if (e != cudaSuccess) std::printf("%s: %s\n", what, cudaGetErrorString(e));
    return e == cudaSuccess;
}

template <typename T> T* upload(const std::vector<T>& v) {
    T* d = nullptr;
    if (cudaMalloc(&d, v.size() * sizeof(T)) != cudaSuccess) return nullptr;
    cudaMemcpy(d, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice);
    return d;
}

}  // namespace

int main() {
    int cards = 0;
    if (cudaGetDeviceCount(&cards) != cudaSuccess || cards == 0) {
        std::puts("poly_i8m_dp4a_test: no card, skipped");
        return 77;
    }
    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, 0);
    kn::gemm_iq_i8_turing(true);
    kn::gemm_iq_i8_allow(19, true);
    kn::gemm_iq_i8_allow(39, true);
    kn::gemm_iq_i8_dp4a(-1);
    const bool pick = kn::gemm_iq_i8_dp4a_on();
    kn::gemm_iq_i8_dp4a(1);
    if (!kn::gemm_iq_i8_dp4a_on()) {
        std::printf("poly_i8m_dp4a_test: %s runs no Turing code (sm_75), skipped\n", prop.name);
        return 77;
    }
    std::printf("%s (sm_%d%d, %d SMs): the card's own pick is %s\n", prop.name, prop.major, prop.minor, prop.multiProcessorCount,
                pick ? "dp4a" : "the MMAs");
    const std::vector<int> types = {2, 6, 7, 8, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20, 21, 22, 23, 29, 39, 40, 42};
    // launches: each matrix's rows (the bounds' steps); the largest says the tile width
    const std::vector<std::vector<int>> launches = {{10, 0, 3, 9}, {30, 17, 0, 25}, {60, 33, 1, 47}, {300, 0, 7, 23, 129}};
    constexpr int kTok = 512;
    std::mt19937 rng(7);
    std::normal_distribution<float> normal(0.f, 1.f);
    int bad = 0, tried = 0;
    for (const int K : {2048, 1152}) {
        std::vector<float> x((size_t) kTok * K), ch((size_t) K, 1.f);
        for (int c = 0; c < K; c += 97) ch[(size_t) c] = 30.f;
        for (int r = 0; r < kTok; ++r) {
            const float s = std::exp(normal(rng));
            for (int c = 0; c < K; ++c) x[(size_t) r * K + c] = normal(rng) * s * ch[(size_t) c];
        }
        float* dx = upload(x);
        void* xq = nullptr;
        if (!dx || !ok(cudaMalloc(&xq, (size_t) kTok * K * 5 / 4 + 64), "cudaMalloc")) return 1;
        kn::gemm_iq_i8_quantize(dx, nullptr, K, kTok, K, xq, nullptr);
        for (const int n_out : {200, 512}) {
            for (int t : types) {
                if (!kn::gemm_iq_i8_takes(t, K)) continue;
                const auto ty = (ggml_type) t;
                if (K % ggml_blck_size(ty) != 0) continue;
                const int n_mats = 5;
                ggml_quantize_init(ty);
                const size_t row = ggml_row_size(ty, K), mat = row * (size_t) n_out;
                std::vector<float> wf((size_t) n_mats * n_out * K), imp((size_t) K);
                for (auto& v : wf) v = normal(rng) * 0.02f;
                for (auto& v : imp) v = 0.5f + std::fabs(normal(rng));
                std::vector<uint8_t> wq(mat * n_mats + 64);
                ggml_quantize_chunk(ty, wf.data(), wq.data(), 0, (int64_t) n_mats * n_out, K, imp.data());
                uint8_t* dw = upload(wq);
                if (!dw) return 1;
                for (const auto& l : launches) {
                    std::vector<int32_t> bounds(1, 0);
                    int max_rows = 0;
                    for (int r : l) {
                        bounds.push_back(bounds.back() + r);
                        max_rows = std::max(max_rows, r);
                    }
                    const int rows = bounds.back(), nm = (int) l.size();
                    std::vector<int32_t> ids((size_t) rows);
                    for (int i = 0; i < rows; ++i) ids[(size_t) i] = (int32_t) ((i * 37 + 11) % rows);
                    int32_t* db = upload(bounds);
                    int32_t* di = upload(ids);
                    float* dy = nullptr;
                    if (!db || !di || !ok(cudaMalloc(&dy, (size_t) rows * n_out * sizeof(float)), "cudaMalloc")) return 1;
                    std::vector<float> y[2];
                    float us[2] = {0, 0};
                    bool ran = true;
                    for (int way = 0; way < 2 && ran; ++way) {
                        kn::gemm_iq_i8_dp4a(way);
                        cudaMemset(dy, 0xFF, (size_t) rows * n_out * sizeof(float));
                        ran = kn::gemm_iq_i8_grouped(t, dw, mat, n_out, K, xq, kTok, db, nm, max_rows, di, dy, n_out, nullptr,
                                                     nullptr, nullptr);
                        if (!ran || !ok(cudaDeviceSynchronize(), "product")) { ran = false; break; }
                        y[way].resize((size_t) rows * n_out);
                        cudaMemcpy(y[way].data(), dy, y[way].size() * sizeof(float), cudaMemcpyDeviceToHost);
                        cudaEvent_t e0, e1;
                        cudaEventCreate(&e0);
                        cudaEventCreate(&e1);
                        float best = 1e30f;
                        for (int rep = 0; rep < 10; ++rep) {
                            cudaEventRecord(e0);
                            kn::gemm_iq_i8_grouped(t, dw, mat, n_out, K, xq, kTok, db, nm, max_rows, di, dy, n_out, nullptr, nullptr,
                                                   nullptr);
                            cudaEventRecord(e1);
                            cudaEventSynchronize(e1);
                            float ms = 0;
                            cudaEventElapsedTime(&ms, e0, e1);
                            best = std::min(best, ms);
                        }
                        cudaEventDestroy(e0);
                        cudaEventDestroy(e1);
                        us[way] = best * 1000.f;
                    }
                    if (ran) {
                        ++tried;
                        size_t diff = 0, nonfinite = 0;
                        for (size_t i = 0; i < y[0].size(); ++i) {
                            if (std::memcmp(&y[0][i], &y[1][i], 4) != 0) ++diff;
                            if (!std::isfinite(y[0][i])) ++nonfinite;
                        }
                        if (diff) ++bad;
                        std::printf("%-8s K %4d out %3d rows %3d (max %3d): %s  MMAs %8.1f us  dp4a %8.1f us  (x%.2f)%s\n",
                                    ggml_type_name(ty), K, n_out, rows, max_rows, diff ? "DIFFERENT" : "same bits", us[0], us[1],
                                    us[1] > 0 ? us[0] / us[1] : 0.0, nonfinite ? "  (non-finite values!)" : "");
                        if (diff) std::printf("    %zu of %zu values differ\n", diff, y[0].size());
                        if (nonfinite) ++bad;
                    } else {
                        std::printf("%-8s K %4d out %3d rows %3d: not launched\n", ggml_type_name(ty), K, n_out, rows);
                    }
                    cudaFree(db);
                    cudaFree(di);
                    cudaFree(dy);
                }
                cudaFree(dw);
            }
        }
        cudaFree(dx);
        cudaFree(xq);
    }
    kn::gemm_iq_i8_dp4a(-1);
    std::printf("%d of %d products differ\n", bad, tried);
    return bad || tried == 0 ? 1 : 0;
}
