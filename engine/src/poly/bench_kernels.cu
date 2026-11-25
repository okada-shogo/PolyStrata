// src/poly/bench_kernels.cu - --bench-report's kernel probes (bench_report.hpp probe_kernels()): what a launch costs
// (back to back, and as a node of a replayed recording), how fast a kernel can read the card's memory (a read-only
// kernel, not a copy), and an fp32 FMA rate.  Run before the model loads; ~1 s.
#include "bench_report.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <vector>

namespace strata::poly::bench {
namespace {

__global__ void empty_kernel() {}

// every thread reads 16-byte words with a grid stride; one word a block written so the reads are not dropped
__global__ void read_kernel(const uint4* __restrict__ p, size_t n, uint4* __restrict__ out) {
    uint32_t a = 0, b = 0, c = 0, d = 0;
    const size_t step = (size_t) gridDim.x * blockDim.x;
    size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    for (; i + 3 * step < n; i += 4 * step) {   // four loads in flight a thread
        const uint4 v0 = p[i], v1 = p[i + step], v2 = p[i + 2 * step], v3 = p[i + 3 * step];
        a ^= v0.x ^ v1.x ^ v2.x ^ v3.x, b ^= v0.y ^ v1.y ^ v2.y ^ v3.y;
        c ^= v0.z ^ v1.z ^ v2.z ^ v3.z, d ^= v0.w ^ v1.w ^ v2.w ^ v3.w;
    }
    for (; i < n; i += step) {
        const uint4 v = p[i];
        a ^= v.x, b ^= v.y, c ^= v.z, d ^= v.w;
    }
    if ((a ^ b ^ c ^ d) == 0x9e3779b9u) out[blockIdx.x] = make_uint4(a, b, c, d);   // practically never
}

// a small dependent step, as a norm or a mix of one token is: one block reads 4 KiB the last kernel wrote, writes 4 KiB
__global__ void small_kernel(const float* __restrict__ x, float* __restrict__ y) {
    for (int i = threadIdx.x; i < 1024; i += blockDim.x) y[i] = fmaf(x[i], 0.999f, 0.001f);
}

// the same across the card: a thread a float, every block of a grid as wide as a token's products are
__global__ void wide_kernel(const float* __restrict__ x, float* __restrict__ y) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    y[i] = fmaf(x[i], 0.999f, 0.001f);
}

// 2 FLOPs a multiply-add, 8 independent chains a thread
__global__ void fma_kernel(float* out, int iters, float s) {
    float x[8];
    for (int k = 0; k < 8; ++k) x[k] = (float) (threadIdx.x + k) * 1e-3f;
    for (int i = 0; i < iters; ++i)
        for (int k = 0; k < 8; ++k) x[k] = fmaf(x[k], s, 0.5f);
    float t = 0;
    for (int k = 0; k < 8; ++k) t += x[k];
    if (t == 1234.5f) out[blockIdx.x] = t;
}

using Clock = std::chrono::steady_clock;
double secs(Clock::time_point t) { return std::chrono::duration<double>(Clock::now() - t).count(); }

}  // namespace

void probe_kernels(Probes& p, int device) {
    const Clock::time_point t0 = Clock::now();
    (void) cudaSetDevice(device);
    cudaStream_t s = nullptr;
    if (cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking) != cudaSuccess) {
        (void) cudaGetLastError();
        return;
    }
    // a launch back to back: the host's cost to hand the card an empty kernel, and the card's to run it
    const int n = 2000;
    for (int i = 0; i < 100; ++i) empty_kernel<<<1, 32, 0, s>>>();
    (void) cudaStreamSynchronize(s);
    Clock::time_point t = Clock::now();
    for (int i = 0; i < n; ++i) empty_kernel<<<1, 32, 0, s>>>();
    if (cudaStreamSynchronize(s) == cudaSuccess) p.launch_us = secs(t) * 1e6 / n;
    // a host round trip: an empty kernel, then the host waits for its event (as the host waits for a router's scores);
    // and a small copy to the host waited for on the stream
    {
        cudaEvent_t ev = nullptr;
        if (cudaEventCreateWithFlags(&ev, cudaEventDisableTiming) == cudaSuccess) {
            const int trips = 300;
            for (int i = 0; i < 20; ++i) {
                empty_kernel<<<1, 32, 0, s>>>();
                (void) cudaEventRecord(ev, s);
                (void) cudaEventSynchronize(ev);
            }
            t = Clock::now();
            bool ok = true;
            for (int i = 0; i < trips && ok; ++i) {
                empty_kernel<<<1, 32, 0, s>>>();
                ok = cudaEventRecord(ev, s) == cudaSuccess && cudaEventSynchronize(ev) == cudaSuccess;
            }
            if (ok) p.sync_us = secs(t) * 1e6 / trips;
            (void) cudaEventDestroy(ev);
        }
        void* d = nullptr;
        void* h = nullptr;
        if (cudaMalloc(&d, 4096) == cudaSuccess && cudaMallocHost(&h, 4096) == cudaSuccess) {
            const int trips = 300;
            for (int i = 0; i < 20; ++i) {
                (void) cudaMemcpyAsync(h, d, 1024, cudaMemcpyDeviceToHost, s);
                (void) cudaStreamSynchronize(s);
            }
            t = Clock::now();
            bool ok = true;
            for (int i = 0; i < trips && ok; ++i)
                ok = cudaMemcpyAsync(h, d, 1024, cudaMemcpyDeviceToHost, s) == cudaSuccess &&
                     cudaStreamSynchronize(s) == cudaSuccess;
            if (ok) p.copy_sync_us = secs(t) * 1e6 / trips;
        }
        if (d) (void) cudaFree(d);
        if (h) (void) cudaFreeHost(h);
        (void) cudaGetLastError();
    }
    // a node of a replayed recording: n empty kernels recorded once, replayed
    cudaGraph_t g = nullptr;
    cudaGraphExec_t exec = nullptr;
    const int nodes = 500;
    if (cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal) == cudaSuccess) {
        for (int i = 0; i < nodes; ++i) empty_kernel<<<1, 32, 0, s>>>();
        if (cudaStreamEndCapture(s, &g) == cudaSuccess && g &&
            cudaGraphInstantiateWithFlags(&exec, g, 0) == cudaSuccess) {
            (void) cudaGraphLaunch(exec, s);
            (void) cudaStreamSynchronize(s);
            const int reps = 8;
            t = Clock::now();
            for (int r = 0; r < reps; ++r) (void) cudaGraphLaunch(exec, s);
            if (cudaStreamSynchronize(s) == cudaSuccess) p.graph_node_us = secs(t) * 1e6 / (reps * nodes);
        }
    }
    if (exec) (void) cudaGraphExecDestroy(exec);
    if (g) (void) cudaGraphDestroy(g);
    (void) cudaGetLastError();
    // the floor of a real kernel: small dependent kernels back to back, launched and as a replayed recording.  An
    // empty kernel only shows the queue's rate; a kernel that reads what the last one wrote also pays the boundary
    // between them (the wait for the last one, its writes made visible), which is what a token's ~1,500 small
    // kernels pay
    {
        float* xy = nullptr;
        if (cudaMalloc(&xy, 2 * 1024 * sizeof(float)) == cudaSuccess) {
            (void) cudaMemsetAsync(xy, 0, 2 * 1024 * sizeof(float), s);
            float* a = xy;
            float* b = xy + 1024;
            for (int i = 0; i < 100; ++i) small_kernel<<<1, 256, 0, s>>>(i % 2 ? b : a, i % 2 ? a : b);
            (void) cudaStreamSynchronize(s);
            const int k = 2000;
            t = Clock::now();
            for (int i = 0; i < k; ++i) small_kernel<<<1, 256, 0, s>>>(i % 2 ? b : a, i % 2 ? a : b);
            if (cudaStreamSynchronize(s) == cudaSuccess) p.small_us = secs(t) * 1e6 / k;
            cudaGraph_t sg = nullptr;
            cudaGraphExec_t se = nullptr;
            const int kn = 500;
            if (cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal) == cudaSuccess) {
                for (int i = 0; i < kn; ++i) small_kernel<<<1, 256, 0, s>>>(i % 2 ? b : a, i % 2 ? a : b);
                if (cudaStreamEndCapture(s, &sg) == cudaSuccess && sg &&
                    cudaGraphInstantiateWithFlags(&se, sg, 0) == cudaSuccess) {
                    (void) cudaGraphLaunch(se, s);
                    (void) cudaStreamSynchronize(s);
                    const int reps = 8;
                    t = Clock::now();
                    for (int r = 0; r < reps; ++r) (void) cudaGraphLaunch(se, s);
                    if (cudaStreamSynchronize(s) == cudaSuccess) p.small_node_us = secs(t) * 1e6 / (reps * kn);
                }
            }
            if (se) (void) cudaGraphExecDestroy(se);
            if (sg) (void) cudaGraphDestroy(sg);
            (void) cudaFree(xy);
        }
        int cus = 0;
        (void) cudaDeviceGetAttribute(&cus, cudaDevAttrMultiProcessorCount, device);
        const int wb = std::max(1, cus) * 2, wn = wb * 256;
        float* w = nullptr;
        if (cudaMalloc(&w, 2 * (size_t) wn * sizeof(float)) == cudaSuccess) {
            (void) cudaMemsetAsync(w, 0, 2 * (size_t) wn * sizeof(float), s);
            float* a = w;
            float* b = w + wn;
            cudaGraph_t wg = nullptr;
            cudaGraphExec_t we = nullptr;
            const int kn = 500;
            if (cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal) == cudaSuccess) {
                for (int i = 0; i < kn; ++i) wide_kernel<<<wb, 256, 0, s>>>(i % 2 ? b : a, i % 2 ? a : b);
                if (cudaStreamEndCapture(s, &wg) == cudaSuccess && wg &&
                    cudaGraphInstantiateWithFlags(&we, wg, 0) == cudaSuccess) {
                    (void) cudaGraphLaunch(we, s);
                    (void) cudaStreamSynchronize(s);
                    const int reps = 8;
                    t = Clock::now();
                    for (int r = 0; r < reps; ++r) (void) cudaGraphLaunch(we, s);
                    if (cudaStreamSynchronize(s) == cudaSuccess) p.wide_node_us = secs(t) * 1e6 / (reps * kn);
                }
            }
            if (we) (void) cudaGraphExecDestroy(we);
            if (wg) (void) cudaGraphDestroy(wg);
            (void) cudaFree(w);
        }
        (void) cudaGetLastError();
    }
    // a read-only kernel over 512 MiB: the rate a matrix-vector product can at best read its weights at
    const size_t bytes = 512ull << 20, words = bytes / sizeof(uint4);
    uint4* buf = nullptr;
    uint4* out = nullptr;
    int sms = 0;
    (void) cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device);
    const int blocks = std::max(1, sms) * 8;
    if (cudaMalloc(&buf, bytes) == cudaSuccess && cudaMalloc(&out, (size_t) blocks * sizeof(uint4)) == cudaSuccess) {
        (void) cudaMemsetAsync(buf, 1, bytes, s);
        read_kernel<<<blocks, 256, 0, s>>>(buf, words, out);   // warm
        (void) cudaStreamSynchronize(s);
        const int reps = 10;
        t = Clock::now();
        for (int r = 0; r < reps; ++r) read_kernel<<<blocks, 256, 0, s>>>(buf, words, out);
        if (cudaStreamSynchronize(s) == cudaSuccess) p.read_gbs = (double) bytes * reps / 1e9 / secs(t);
    }
    if (buf) (void) cudaFree(buf);
    if (out) (void) cudaFree(out);
    (void) cudaGetLastError();
    // fp32 multiply-adds: the card's plain arithmetic rate (not its int8 or tensor-core one)
    float* f = nullptr;
    if (cudaMalloc(&f, (size_t) blocks * 4 * sizeof(float)) == cudaSuccess) {
        const int iters = 4096, threads = 256, grid = std::max(1, sms) * 4 * 4;
        fma_kernel<<<grid, threads, 0, s>>>(f, 64, 1.0001f);
        (void) cudaStreamSynchronize(s);
        const int reps = 4;
        t = Clock::now();
        for (int r = 0; r < reps; ++r) fma_kernel<<<grid, threads, 0, s>>>(f, iters, 1.0001f);
        if (cudaStreamSynchronize(s) == cudaSuccess)
            p.fma_gflops = 2.0 * 8 * (double) iters * threads * grid * reps / 1e9 / secs(t);
        (void) cudaFree(f);
    }
    (void) cudaGetLastError();
    (void) cudaStreamDestroy(s);
    p.seconds += secs(t0);
}

}  // namespace strata::poly::bench
