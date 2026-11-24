// src/poly/gemm_small_test.cu - gemm_f32() for one group of up to 256 outputs (gemm_small_kernel, gemm_tile_kernel past
// 64 of them) and gemm_f32_tiles()' every way (gemm_ahead_kernel's four tiles for 33 to 256, its pick, #224's first
// rule, the cost model) against the same products on the host: each output's sum of fmaf in the inputs' order, so the
// bits must be the same.  Then gemm_f32_tiles_autotune() on Inkling 2 Mini's shapes (its router, 2,304 inputs to 128
// outputs, and its controllers' down products, 4,608 to 64): every pick a way, a cache round trip that gives the same
// picks, a second engine's write that keeps the first's lines and picks, no temporary file left, and the cost model
// where the cache cannot be written.  --bench times every way on those shapes at 333, 1,024
// and 4,096 rows with the timed picks, prints each shape's picks with their grids and the SM (CU) count, and a hash of
// each way's outputs, which must be the same for all.
//
// usage: poly_gemm_small_test [--bench [rounds]]   (exit 0: every output the same; rounds default 3)
#include "strata/poly/kernels.hpp"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <random>
#include <string>
#include <utility>
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

// how a product is done: gemm_f32_tiles() (its pick: tuned, else the cost model), #224's first rule (the largest tile
// that gave half the SMs a block), the cost model, or one way of gemm_f32_tiles_at() (0..3 the tiles, kF32Way gemm_f32)
enum Way { kTiles = -3, kPr224 = -2, kModel = -1 };

int sms_now() {
    int dev = 0, sms = 1;
    if (cudaGetDevice(&dev) != cudaSuccess || cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev) != cudaSuccess)
        sms = 1;
    return sms;
}

int pr224_pick(int n_out, int rows) {
    const int sms = sms_now();
    const auto half = [&](int tok, int out) { return 2LL * ((rows + tok - 1) / tok) * ((n_out + out - 1) / out) >= sms; };
    if (n_out > 64 && half(64, 128)) return 0;
    if (half(64, 64)) return 1;
    if (half(32, 64)) return 2;
    return 3;
}

void run(cublasHandle_t h, int way, const Product& p, int n_in, int n_out, int rows) {
    if (way == kTiles) return gemm_f32_tiles(h, p.dw, p.dx, p.dy, n_in, n_out, rows);
    if (way == kPr224) way = pr224_pick(n_out, rows);
    else if (way == kModel) way = gemm_f32_tiles_model(n_in, n_out, rows);
    gemm_f32_tiles_at(way, h, p.dw, p.dx, p.dy, n_in, n_out, rows);
}

const char* const kWayName[kF32Ways] = {"tile_64x128", "tile_64x64", "tile_32x64", "tile_32x32", "gemm_f32"};

const char* name_of(int way) {
    return way == kTiles ? "gemm_f32_tiles" : way == kPr224 ? "pr224_rule" : way == kModel ? "cost_model" : kWayName[way];
}

// the ways to check or time for n_out outputs: gemm_f32, then (33 to 256 outputs) the pick, #224's rule, the cost
// model and each tile
std::vector<int> ways_of(int n_out) {
    std::vector<int> ways{kF32Way, kTiles};
    if (gemm_f32_tiles_takes(n_out)) {
        ways.push_back(kPr224);
        ways.push_back(kModel);
        for (int t = 0; t < 4; ++t)
            if (gemm_f32_tiles_way_ok(t, n_out)) ways.push_back(t);
    }
    return ways;
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
    for (int way : ways_of(n_out)) {
        cudaMemset(p.dy, 0xff, p.y.size() * sizeof(float));
        run(h, way, p, n_in, n_out, rows);
        p.get();
        const bool ok = !std::memcmp(want.data(), p.y.data(), want.size() * sizeof(float));
        std::printf("%s %-14s n_in %4d n_out %3d rows %4d\n", ok ? "ok  " : "FAIL", name_of(way), n_in, n_out, rows);
        bad += !ok;
    }
    return bad;
}

// each way's time on the shape, `rounds` times over (each the mean of 50 calls), and a hash of its outputs
void bench(cublasHandle_t h, int n_in, int n_out, int rows, std::mt19937& rng, int rounds) {
    Product p(n_in, n_out, rows, rng);
    cudaStream_t stream = nullptr;
    cublasGetStream(h, &stream);
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0), cudaEventCreate(&e1);
    static const int tok[4] = {64, 64, 32, 32}, out[4] = {128, 64, 64, 32};
    const auto grid = [&](int way) {
        return way < 4 ? (long long) ((rows + tok[way] - 1) / tok[way]) * ((n_out + out[way] - 1) / out[way]) : 0LL;
    };
    const int tuned = gemm_f32_tiles_pick(n_in, n_out, rows), model = gemm_f32_tiles_model(n_in, n_out, rows),
              old = pr224_pick(n_out, rows);
    std::printf("pick n_in %4d n_out %3d rows %4d bucket %4d  sms %d  tuned %s grid %lld  model %s grid %lld  "
                "pr224 %s grid %lld\n",
                n_in, n_out, rows, gemm_f32_tiles_bucket(rows), sms_now(), kWayName[tuned], grid(tuned),
                kWayName[model], grid(model), kWayName[old], grid(old));
    const std::vector<int> ways = ways_of(n_out);
    for (int r = 0; r < rounds; ++r)
        for (int way : ways) {
            for (int i = 0; i < 5; ++i) run(h, way, p, n_in, n_out, rows);
            constexpr int reps = 50;
            cudaEventRecord(e0, stream);
            for (int i = 0; i < reps; ++i) run(h, way, p, n_in, n_out, rows);
            cudaEventRecord(e1, stream);
            cudaEventSynchronize(e1);
            float ms = 0.f;
            cudaEventElapsedTime(&ms, e0, e1);
            p.get();
            uint64_t hash = 14695981039346656037ull;   // FNV-1a of the outputs' bytes
            const unsigned char* const bytes = reinterpret_cast<const unsigned char*>(p.y.data());
            for (size_t i = 0; i < p.y.size() * sizeof(float); ++i) hash = (hash ^ bytes[i]) * 1099511628211ull;
            const double us = 1000.0 * ms / reps;
            std::printf("bench %-14s n_in %4d n_out %3d rows %4d round %d  %8.1f us  %5.2f TFLOPS  fnv %016llx\n", name_of(way),
                        n_in, n_out, rows, r + 1, us, 2.0 * n_in * n_out * rows / us * 1e-6, (unsigned long long) hash);
        }
    cudaEventDestroy(e0), cudaEventDestroy(e1);
}

// gemm_f32_tiles_autotune() on Inkling 2 Mini's shapes: every pick a way for its shape; a second call takes them
// all from the cache file and gives the same; a cache that cannot be written gives the cost model's
int check_autotune(cublasHandle_t h, const std::string& cache, int top) {
    const std::vector<std::pair<int, int>> shapes{{2304, 128}, {4608, 64}};
    std::remove(cache.c_str());
    std::string log;
    const std::vector<F32TilePick> a = gemm_f32_tiles_autotune(h, shapes, top, cache, &log);
    std::printf("autotune   %s\n", log.c_str());
    int bad = 0;
    for (const F32TilePick& p : a)
        if (!gemm_f32_tiles_way_ok(p.way, p.n_out) || gemm_f32_tiles_pick(p.n_in, p.n_out, p.rows) != p.way) ++bad;
    std::vector<F32TilePick> read;
    const bool got = gemm_f32_tiles_cache_read(cache, gemm_f32_tiles_device_key(), read);
    const std::vector<F32TilePick> b = gemm_f32_tiles_autotune(h, shapes, top, cache, &log);
    std::printf("autotune   %s\n", log.c_str());
    const auto same = [](const std::vector<F32TilePick>& x, const std::vector<F32TilePick>& y) {
        if (x.size() != y.size()) return false;
        for (size_t i = 0; i < x.size(); ++i)
            if (x[i].n_in != y[i].n_in || x[i].n_out != y[i].n_out || x[i].rows != y[i].rows || x[i].way != y[i].way)
                return false;
        return true;
    };
    const bool ok_valid = bad == 0 && !a.empty(), ok_trip = got && same(a, read) && same(a, b) &&
                                                          log.find(" 0 timed") != std::string::npos;
    std::printf("%s autotune picks valid (%zu)\n%s autotune cache round trip\n", ok_valid ? "ok  " : "FAIL", a.size(),
                ok_trip ? "ok  " : "FAIL");
    // a second engine with another shape and a smaller prompt batch: it times only its own, and its write keeps the
    // first's lines; the card's picks keep the first's too
    const std::vector<F32TilePick> m = gemm_f32_tiles_autotune(h, {{2000, 100}}, 32, cache, &log);
    std::printf("autotune   %s\n", log.c_str());
    std::vector<F32TilePick> merged;
    gemm_f32_tiles_cache_read(cache, gemm_f32_tiles_device_key(), merged);
    bool ok_merge = m.size() == 1 && log.find(" 1 timed") != std::string::npos && merged.size() == a.size() + 1;
    for (const F32TilePick& p : a) {
        bool in = false;
        for (const F32TilePick& q : merged) in = in || (q.n_in == p.n_in && q.n_out == p.n_out && q.rows == p.rows && q.way == p.way);
        ok_merge = ok_merge && in && gemm_f32_tiles_pick(p.n_in, p.n_out, p.rows) == p.way;
    }
    ok_merge = ok_merge && !m.empty() && gemm_f32_tiles_pick(2000, 100, 32) == m[0].way;
    // every write and probe went through a temporary file of its own and left none
    namespace fs = std::filesystem;
    const std::string leaf = fs::path(cache).filename().string();
    int left = 0;
    std::error_code ec;
    for (const fs::directory_entry& e : fs::directory_iterator(fs::path(cache).parent_path(), ec)) {
        const std::string n = e.path().filename().string();
        left += n.size() > leaf.size() + 1 && n.compare(0, leaf.size() + 2, "." + leaf + ".") == 0;
    }
    std::printf("%s autotune merged write (%zu lines)\n%s autotune no temporary file left (%d)\n", ok_merge ? "ok  " : "FAIL",
                merged.size(), left == 0 ? "ok  " : "FAIL", left);
    // a cache under a regular file (no folder can be made there, on any system): the cost model's picks
    const std::vector<F32TilePick> c = gemm_f32_tiles_autotune(h, shapes, top, cache + "/sub/f32-tiles.txt", &log);
    bool model = true;
    for (const F32TilePick& p : c) model = model && p.way == gemm_f32_tiles_model(p.n_in, p.n_out, p.rows);
    std::printf("autotune   %s\n%s autotune unwritable cache: cost model\n", log.c_str(), model ? "ok  " : "FAIL");
    // the fallback: a tile only on the families the model was checked on (CUDA sm_75, sm_100 to sm_121), else
    // gemm_f32() on every shape
    bool fam = true;
    {
        int dev = 0, major = 0, minor = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
        cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev);
#if defined(STRATA_USE_HIP)
        const bool want_known = false;
#else
        const bool want_known = (major == 7 && minor == 5) || major == 10 || major == 11 || major == 12;
#endif
        fam = gemm_f32_tiles_model_known() == want_known;
        for (const auto& [n_in, n_out] : shapes)
            for (int rows : {32, 333, 1024, 4096}) {
                const int m = gemm_f32_tiles_model(n_in, n_out, rows);
                fam = fam && (want_known ? m >= 0 && m < 4 && gemm_f32_tiles_way_ok(m, n_out) : m == kF32Way);
            }
        std::printf("%s cost model fallback: %s (sm %d.%d)\n", fam ? "ok  " : "FAIL",
                    want_known ? "tiles" : "gemm_f32", major, minor);
    }
    // the timed picks again, for --bench
    gemm_f32_tiles_use(a.data(), (int) a.size());
    return !ok_valid + !ok_trip + !ok_merge + (left != 0) + !model + !fam;
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
    bad += check(h, 2008, 100, 45, rng);   // inputs that end inside a 16-input step (the 64-token tiles')
    bad += check(h, 7, 64, 45, rng);       // fewer inputs than one step
    for (int rows : {16, 333, 1024}) {     // Inkling 2 Mini's router and controllers
        bad += check(h, 2304, 128, rows, rng);
        bad += check(h, 4608, 64, rows, rng);
    }
    {
        std::error_code ec;
        const std::filesystem::path dir = std::filesystem::temp_directory_path(ec);
        const std::string cache = ((ec ? std::filesystem::path(".") : dir) / "poly_gemm_small_test-f32-tiles.txt").string();
        bad += check_autotune(h, cache, 4096);
        std::remove(cache.c_str());
    }
    if (argc > 1 && !std::strcmp(argv[1], "--bench")) {
        const int rounds = argc > 2 ? std::atoi(argv[2]) : 3;
        std::mt19937 brng(11);
        for (int rows : {333, 1024, 4096}) {
            bench(h, 2304, 128, rows, brng, rounds > 0 ? rounds : 1);
            bench(h, 4608, 64, rows, brng, rounds > 0 ? rounds : 1);
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
