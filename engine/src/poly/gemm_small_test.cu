// src/poly/gemm_small_test.cu - gemm_f32() for one group of up to 256 outputs (gemm_small_kernel, gemm_tile_kernel past
// 64 of them) and gemm_f32_tiles() (gemm_ahead_kernel for 33 to 256) against the same products on the host: each
// output's sum of fmaf in the inputs' order, so the bits must be the same.  --bench also times both on the calls of
// Inkling 2 Mini's prompt batches (its router, 2,304 inputs to 128 outputs, and its controllers' down products, 4,608
// to 64) and prints a hash of each one's outputs, which must be the same for both.
//
// usage: poly_gemm_small_test [--bench]   (exit 0: every output the same)
#include "strata/poly/kernels.hpp"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace {

using namespace strata::poly;

struct Product {
    std::vector<float> w, x, y;
    float *dw = nullptr, *dx = nullptr, *dy = nullptr;
    Product(int n_in, int n_out, int rows, std::mt19937& rng)
        : w(size_t(n_in) * n_out), x(size_t(n_in) * rows), y(size_t(n_out) * rows) {
        std::normal_distribution<float> nd(0.f, 1.f);
        for (float& v : w) v = 0.05f * nd(rng);
        for (float& v : x) v = nd(rng);
        cudaMalloc(&dw, w.size() * sizeof(float));
        cudaMalloc(&dx, x.size() * sizeof(float));
        cudaMalloc(&dy, y.size() * sizeof(float));
        cudaMemcpy(dw, w.data(), w.size() * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(dx, x.data(), x.size() * sizeof(float), cudaMemcpyHostToDevice);
    }
    ~Product() { cudaFree(dw), cudaFree(dx), cudaFree(dy); }
    void get() { cudaMemcpy(y.data(), dy, y.size() * sizeof(float), cudaMemcpyDeviceToHost); }
};

void run(cublasHandle_t h, bool tiles, const Product& p, int n_in, int n_out, int rows) {
    if (tiles) gemm_f32_tiles(h, p.dw, p.dx, p.dy, n_in, n_out, rows);
    else gemm_f32(h, p.dw, p.dx, p.dy, n_in, n_out, n_out, rows);
}

int check(cublasHandle_t h, int n_in, int n_out, int rows, std::mt19937& rng) {
    Product p(n_in, n_out, rows, rng);
    std::vector<float> want(p.y.size());
    for (int t = 0; t < rows; ++t)
        for (int o = 0; o < n_out; ++o) {
            float acc = 0.f;
            for (int k = 0; k < n_in; ++k)
                acc = std::fma(p.x[size_t(t) * n_in + k], p.w[size_t(o) * n_in + k], acc);
            want[size_t(t) * n_out + o] = acc;
        }
    int bad = 0;
    for (bool tiles : {false, true}) {
        cudaMemset(p.dy, 0xff, p.y.size() * sizeof(float));
        run(h, tiles, p, n_in, n_out, rows);
        p.get();
        const bool ok = !std::memcmp(want.data(), p.y.data(), want.size() * sizeof(float));
        std::printf("%s %-14s n_in %4d n_out %3d rows %4d\n", ok ? "ok  " : "FAIL", tiles ? "gemm_f32_tiles" : "gemm_f32",
                    n_in, n_out, rows);
        bad += !ok;
    }
    return bad;
}

void bench(cublasHandle_t h, int n_in, int n_out, int rows, std::mt19937& rng) {
    Product p(n_in, n_out, rows, rng);
    cudaStream_t stream = nullptr;
    cublasGetStream(h, &stream);
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0), cudaEventCreate(&e1);
    for (bool tiles : {false, true}) {
        for (int i = 0; i < 5; ++i) run(h, tiles, p, n_in, n_out, rows);
        constexpr int reps = 50;
        cudaEventRecord(e0, stream);
        for (int i = 0; i < reps; ++i) run(h, tiles, p, n_in, n_out, rows);
        cudaEventRecord(e1, stream);
        cudaEventSynchronize(e1);
        float ms = 0.f;
        cudaEventElapsedTime(&ms, e0, e1);
        p.get();
        uint64_t hash = 14695981039346656037ull;   // FNV-1a of the outputs' bytes
        const unsigned char* const bytes = reinterpret_cast<const unsigned char*>(p.y.data());
        for (size_t i = 0; i < p.y.size() * sizeof(float); ++i) hash = (hash ^ bytes[i]) * 1099511628211ull;
        const double us = 1000.0 * ms / reps;
        std::printf("bench %-14s n_in %4d n_out %3d rows %4d  %8.1f us  %5.2f TFLOPS  fnv %016llx\n",
                    tiles ? "gemm_f32_tiles" : "gemm_f32", n_in, n_out, rows, us,
                    2.0 * n_in * n_out * rows / us * 1e-6, (unsigned long long) hash);
    }
    cudaEventDestroy(e0), cudaEventDestroy(e1);
}

}  // namespace

int main(int argc, char** argv) {
    cublasHandle_t h = nullptr;
    if (cublasCreate(&h) != CUBLAS_STATUS_SUCCESS) {
        std::printf("no cuBLAS\n");
        return 1;
    }
    std::mt19937 rng(3);
    int bad = 0;
    for (int n_out : {32, 64, 100, 128, 256})
        for (int rows : {32, 37, 1024, 2048}) bad += check(h, 2048, n_out, rows, rng);
    bad += check(h, 2000, 64, 45, rng);    // inputs that end inside a step
    bad += check(h, 2000, 256, 45, rng);   // and of gemm_tile_kernel's
    bad += check(h, 2000, 33, 45, rng);    // the fewest outputs that take gemm_ahead_kernel
    for (int rows : {16, 333, 1024}) {     // Inkling 2 Mini's router and controllers
        bad += check(h, 2304, 128, rows, rng);
        bad += check(h, 4608, 64, rows, rng);
    }
    if (argc > 1 && !std::strcmp(argv[1], "--bench")) {
        std::mt19937 brng(11);
        for (int rows : {333, 1024, 4096}) {
            bench(h, 2304, 128, rows, brng);
            bench(h, 4608, 64, rows, brng);
        }
    }
    cublasDestroy(h);
    if (const cudaError_t e = cudaGetLastError(); e != cudaSuccess) {
        std::printf("CUDA: %s\n", cudaGetErrorString(e));
        return 1;
    }
    std::printf(bad ? "%d failed\n" : "all passed\n", bad);
    return bad ? 1 : 0;
}
