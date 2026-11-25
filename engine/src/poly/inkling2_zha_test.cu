// src/poly/inkling2_zha_test.cu - the ZHA index's prompt kernels (inkling2_zha.cu) against a CPU reference: the chunk
// scores (zha_score_kernel on NVIDIA, zha_score_scalar_kernel and zha_score_wmma_kernel on AMD), the exact keys
// (zha_exact_kernel / zha_exact_scalar_kernel, zha_exact_wmma_kernel) and the tail (zha_tail_mma_kernel /
// zha_tail_scalar_kernel), on one KV group of Inkling 2 Mini's shape with random inputs.  On AMD both paths run: the
// scalar kernels everywhere, the matrix-core ones too on a card that has them (gfx11 / gfx12, zha_wmma_card()).  The
// reference rounds where the kernels do (the queries, keys, values and weights in fp16) and sums in double, so what is
// left is the order of the kernels' fp32 sums.
// inkling2_zha_test [--dump FILE]: FILE gets every output (score, rstar, top, the exact and the tail parts) of the
// path the engine runs on this card raw, to compare two builds.  Exit 0 all within the tolerances, 1 not, 77 no GPU.
#include "inkling2_zha.cu"

#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

using namespace strata::poly::ink;

namespace {

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

template <typename T> T* up(const std::vector<T>& v) {
    T* p = nullptr;
    ck(cudaMalloc((void**) &p, std::max<size_t>(v.size(), 1) * sizeof(T)), "cudaMalloc");
    ck(cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice), "upload");
    return p;
}
template <typename T> std::vector<T> down(const T* p, size_t n) {
    std::vector<T> v(n);
    ck(cudaMemcpy(v.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost), "download");
    return v;
}

float h2f(uint16_t h) { return __half2float(__ushort_as_half(h)); }
uint16_t f2h(float f) { return __half_as_ushort(__float2half_rn(f)); }
float r16(double x) { return h2f(f2h((float) x)); }

struct Rng {   // the same numbers on every platform (no std:: distributions)
    std::mt19937 g{20261125u};
    float u() { return (float) (g() >> 8) * (1.f / 16777216.f); }   // [0, 1)
    float sym(float a) { return (2.f * u() - 1.f) * a; }
    int code(int n) { return (int) (g() % (unsigned) n); }
};

// Inkling 2 Mini's full layers: 24 heads in 8 KV groups of 3, chunks of 64, 4 references, the last 1,024 keys recent.
constexpr int kHeads = 24, kKv = 8, kPer = 3, kG = 2, kC = 64, kR = 4, kRecent = 1024;
constexpr int kRows = 150;                 // 3 blocks of 64 tokens, the last one partial
constexpr long long kPos = 40000;          // the first token's position
constexpr int kMax = 640;                  // max_chunks: past the last token's chunk, words of 32
constexpr int kBuilt = (int) (kPos / kC);  // summaries of the complete chunks
constexpr int kExtent = 200;
constexpr long long kRing = 51 * 64;
constexpr int kSplit = 3, kTsplit = 3, kSegs = kSplit + kTsplit;
constexpr float kLog2e = 1.44269504f;

int far_of(long long own, long long* fr) {
    *fr = std::max(1LL, own + 1 - kRecent) / kC * kC;
    return std::max(0, (int) (*fr / kC) - 1);
}

struct Check {
    const char* name;
    double worst = 0, tol;
    long long bad = 0, n = 0;
    Check(const char* nm, double t) : name(nm), tol(t) {}
    void see(double got, double want, double scale) {
        ++n;
        const bool both_inf = std::isinf(got) && std::isinf(want) && (got > 0) == (want > 0);
        const double e = both_inf ? 0 : std::fabs(got - want) / std::max(1.0, std::fabs(scale));
        if (!(e <= tol)) ++bad;
        if (e > worst || std::isnan(e)) worst = std::isnan(e) ? INFINITY : e;
    }
    bool report() const {
        std::printf("%-12s %9lld values, worst error %.3g (tolerance %.1g), %lld over\n", name, n, worst, tol, bad);
        return bad == 0;
    }
};

}  // namespace

int main(int argc, char** argv) {
    std::string dump;
    for (int i = 1; i < argc; ++i)
        if (!std::strcmp(argv[i], "--dump") && i + 1 < argc) dump = argv[++i];
    int dev_count = 0;
    if (cudaGetDeviceCount(&dev_count) != cudaSuccess || dev_count == 0) {
        std::printf("no GPU: skipped\n");
        return 77;
    }
    Rng rng;
    const size_t hs = (size_t) kRows * kMax;
    // ---- inputs ----
    std::vector<float> q((size_t) kRows * kHeads * 128);
    for (float& x : q) x = rng.sym(2.f);
    // 4-bit summaries of every KV head (khat, vhat): codes then fp16 scales a 32 values; bhat
    const size_t srows = (size_t) kKv * kMax * kR;
    const size_t q8n = zha_q8n_make(srows, kMax, true);
    std::vector<uint8_t> khat(srows * 64 + srows * 8), vhat(khat.size());
    for (auto* s : {&khat, &vhat}) {
        for (size_t i = 0; i < srows * 64; ++i) (*s)[i] = (uint8_t) rng.code(256);
        uint16_t* sc = reinterpret_cast<uint16_t*>(s->data() + srows * 64);
        for (size_t i = 0; i < srows * 4; ++i) sc[i] = f2h(0.02f + rng.u() * 0.06f);
    }
    std::vector<float> bhat(srows);
    for (float& x : bhat) x = rng.sym(3.f);
    const auto sdeq = [&](const std::vector<uint8_t>& s, size_t row, int d) {
        const uint8_t b = s[row * 64 + d / 2];
        const int code = d & 1 ? b >> 4 : b & 15;
        return r16((double) (code - 8) * h2f(reinterpret_cast<const uint16_t*>(s.data() + srows * 64)[row * 4 + d / 32]));
    };
    // the tail's inputs: each (head, token, chunk)'s score and best reference, the group's selection
    std::vector<float> tscore(kPer * hs);
    std::vector<uint8_t> trstar(kPer * hs), sel(hs, 0);
    for (float& x : tscore) x = rng.sym(4.f);
    for (uint8_t& x : trstar) x = (uint8_t) rng.code(kR);
    for (int t = 0; t < kRows; ++t) {
        long long fr;
        const int nf = far_of(kPos + t, &fr);
        for (int c = 1; c <= nf; ++c) sel[(size_t) t * kMax + c] = rng.u() < 0.16f;
    }
    // the staging buffer (4-bit, chunk c at slot cmap[c]) and the ring (Q8_0, every KV head's row, key j at j % ring)
    std::vector<int> cmap(kMax);
    for (int c = 0; c < kMax; ++c) cmap[c] = (int) ((7LL * c) % kMax);
    const size_t skeys = (size_t) kMax * kC;
    std::vector<uint8_t> skc(skeys * 64), svc(skeys * 64);
    std::vector<uint16_t> sks(skeys * 4), svs(skeys * 4);
    for (auto* v : {&skc, &svc}) for (uint8_t& x : *v) x = (uint8_t) rng.code(256);
    for (auto* v : {&sks, &svs}) for (uint16_t& x : *v) x = f2h(0.05f + rng.u() * 0.1f);
    const size_t rrow = (size_t) kKv * 128;
    std::vector<int8_t> rkc(kRing * rrow), rvc(kRing * rrow);
    std::vector<uint16_t> rks(kRing * rrow / 32), rvs(kRing * rrow / 32);
    for (auto* v : {&rkc, &rvc}) for (int8_t& x : *v) x = (int8_t) (rng.code(255) - 127);
    for (auto* v : {&rks, &rvs}) for (uint16_t& x : *v) x = f2h(0.003f + rng.u() * 0.006f);
    std::vector<float> bias((size_t) kPer * kRows * kExtent);
    for (float& x : bias) x = rng.sym(1.f);

    // ---- on the card ----
    float* d_q = up(q);
    __half* d_kh = (__half*) up(khat);
    __half* d_vh = (__half*) up(vhat);
    float* d_bh = up(bhat);
    float* d_score = up(std::vector<float>(kPer * hs, 0.f));
    uint8_t* d_rstar = up(std::vector<uint8_t>(kPer * hs, 0));
    float* d_top = up(std::vector<float>(kPer * kRows, 0.f));
    ck(cudaMemset(d_top, 0xFF, kPer * kRows * sizeof(float)), "memset");
    float* d_ts = up(tscore);
    uint8_t* d_tr = up(trstar);
    uint8_t* d_sel = up(sel);
    int* d_cmap = up(cmap);
    ZhaKv stage, ring;
    stage.kc = (const int8_t*) up(skc), stage.vc = (const int8_t*) up(svc), stage.ks = up(sks), stage.vs = up(svs);
    stage.row = 128, stage.headed = 0, stage.q4 = 1;
    ring.kc = up(rkc), ring.vc = up(rvc), ring.ks = up(rks), ring.vs = up(rvs);
    ring.row = (long long) rrow;
    float* d_bias = up(bias);
    const int blocks = (kRows + kZhaTokens - 1) / kZhaTokens, words = kMax / 32;
    uint32_t* d_bits = up(std::vector<uint32_t>((size_t) blocks * words, 0));
    const size_t n_part = (size_t) kRows * kPer * kSegs * 130;
    float* d_part = up(std::vector<float>(n_part, 0.f));
    zha_bits_kernel<<<dim3(blocks, grid(words, 128)), 128>>>(d_sel, 1, hs, kRows, kMax, d_bits, words, nullptr);
    // a path's outputs: 0 the tensor cores (NVIDIA) or the scalar kernels (AMD), 1 the matrix cores (AMD)
    struct Out {
        const char* name;
        std::vector<float> score;
        std::vector<uint8_t> rstar;
        std::vector<float> top, part;
    };
    const auto run = [&](int path) {
        ck(cudaMemset(d_score, 0, kPer * hs * sizeof(float)), "memset");
        ck(cudaMemset(d_rstar, 0, kPer * hs), "memset");
        ck(cudaMemset(d_top, 0xFF, kPer * kRows * sizeof(float)), "memset");
        ck(cudaMemset(d_part, 0, n_part * sizeof(float)), "memset");
#if defined(__HIPCC__)
        const char* name = path ? "wmma" : "scalar";
        if (path) {
            zha_score_wmma_kernel<4><<<dim3(blocks, 3), kZhaWarps * 32>>>(d_q, d_kh, d_bh, kBuilt, kMax, kRows, kHeads,
                                                                           kG, kPer, d_score, d_rstar, hs, q8n, kPos, kC,
                                                                           kRecent, d_top);
            zha_exact_wmma_kernel<<<dim3(blocks, kSplit), kZhaWarps * 32>>>(
                d_q, stage, d_cmap, ring, kRing, d_sel, 1, hs, kMax, kPos, kRows, kHeads, kG, kPer, kC, kRecent, d_bias,
                kExtent, d_part, kSegs, d_bits, words);
        } else {
            zha_score_scalar_kernel<4><<<dim3(blocks, 3), kZhaScalarThreads>>>(d_q, d_kh, d_bh, kBuilt, kMax, kRows,
                                                                             kHeads, kG, kPer, d_score, d_rstar, hs, q8n,
                                                                             kPos, kC, kRecent, d_top);
            zha_exact_scalar_kernel<<<dim3(blocks, kSplit), kZhaScalarThreads>>>(
                d_q, stage, d_cmap, ring, kRing, d_sel, 1, hs, kMax, kPos, kRows, kHeads, kG, kPer, kC, kRecent, d_bias,
                kExtent, d_part, kSegs, d_bits, words);
        }
        zha_tail_scalar_kernel<4><<<dim3(blocks, kTsplit), kZhaScalarThreads>>>(
            d_ts, d_sel, d_tr, d_vh, kMax, kPos, kRows, kG, kPer, kC, kRecent, hs, 1, d_part, kSegs, kSplit, q8n);
#else
        const char* name = "mma";
        (void) path;
        zha_score_kernel<4><<<dim3(blocks, 3), kZhaWarps * 32>>>(d_q, d_kh, d_bh, kBuilt, kMax, kRows, kHeads, kG, kPer,
                                                                  d_score, d_rstar, hs, q8n, kPos, kC, kRecent, d_top);
        ck(cudaFuncSetAttribute(zha_exact_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 2 * kZhaRaw), "attribute");
        zha_exact_kernel<<<dim3(blocks, kSplit), kZhaWarps * 32, 2 * kZhaRaw>>>(
            d_q, stage, d_cmap, ring, kRing, d_sel, 1, hs, kMax, kPos, kRows, kHeads, kG, kPer, kC, kRecent, d_bias,
            kExtent, d_part, kSegs, d_bits, words);
        zha_tail_mma_kernel<4><<<dim3(blocks, kTsplit), kZhaWarps * 32>>>(d_ts, d_sel, d_tr, d_vh, kMax, kPos, kRows, kG,
                                                                         kPer, kC, kRecent, hs, 1, d_part, kSegs, kSplit,
                                                                         q8n);
#endif
        ck(cudaGetLastError(), "launch");
        ck(cudaDeviceSynchronize(), "the kernels");
        return Out{name, down(d_score, kPer * hs), down(d_rstar, kPer * hs), down(d_top, kPer * kRows),
                   down(d_part, n_part)};
    };
    std::vector<Out> outs;
    outs.push_back(run(0));
#if defined(__HIPCC__)
    if (zha_wmma_card()) outs.push_back(run(1));
#endif
    const int nv = (int) outs.size();
    if (!dump.empty()) {   // the engine's path on this card: the last
        const Out& d = outs.back();
        FILE* f = std::fopen(dump.c_str(), "wb");
        if (!f) return 1;
        std::fwrite(d.score.data(), 4, d.score.size(), f);
        std::fwrite(d.rstar.data(), 1, d.rstar.size(), f);
        std::fwrite(d.top.data(), 4, d.top.size(), f);
        std::fwrite(d.part.data(), 4, d.part.size(), f);
        std::fclose(f);
    }

    // ---- the reference ----
    const auto qr = [&](int t, int p, int d, float scale) {
        return r16((double) (q[((size_t) t * kHeads + kG * kPer + p) * 128 + d] * scale));
    };
    struct Checks {
        Check score{"score", 1e-4}, rstar{"rstar", 0}, top{"top", 1e-4};
        Check em{"exact top", 1e-4}, el{"exact sum", 1e-4}, eo{"exact out", 1e-3};
        Check tm{"tail top", 1e-5}, tl{"tail sum", 1e-4}, to{"tail out", 1e-3};
    };
    std::vector<Checks> cks(nv);
    const float sc = 1.f / std::sqrt(128.f);
    for (int p = 0; p < kPer; ++p)
        for (int t = 0; t < kRows; ++t) {
            double qv[128];
            for (int d = 0; d < 128; ++d) qv[d] = qr(t, p, d, sc);
            long long fr;
            const int nf = far_of(kPos + t, &fr);
            double tmax = -INFINITY;
            for (int c = 0; c < kBuilt; ++c) {
                double v[kR];
                int arg = 0;
                for (int r = 0; r < kR; ++r) {
                    const size_t row = ((size_t) kG * kMax + c) * kR + r;
                    double s = 0;
                    for (int d = 0; d < 128; ++d) s += qv[d] * sdeq(khat, row, d);
                    v[r] = s + bhat[row];
                    if (v[r] > v[arg]) arg = r;
                }
                double gap = INFINITY;   // to the next best: a near tie may go either way in fp32
                for (int r = 0; r < kR; ++r)
                    if (r != arg) gap = std::min(gap, v[arg] - v[r]);
                const size_t at = (size_t) p * hs + (size_t) t * kMax + c;
                for (int k = 0; k < nv; ++k) {
                    cks[k].score.see(outs[k].score[at], v[arg], v[arg]);
                    cks[k].rstar.see(gap < 1e-4 ? arg : outs[k].rstar[at], arg, 1);
                }
                if (c >= 1 && c <= nf) tmax = std::max(tmax, v[arg]);
            }
            for (int k = 0; k < nv; ++k) cks[k].top.see(outs[k].top[(size_t) p * kRows + t], tmax, tmax);
        }

    // the parts: (top, sum, values) of a segment; values compared as values / sum
    const auto see_part = [&](int t, int p, int seg, double m, double l, const double* o, bool exact) {
        for (int k = 0; k < nv; ++k) {
            Check& cm = exact ? cks[k].em : cks[k].tm;
            Check& cl = exact ? cks[k].el : cks[k].tl;
            Check& co = exact ? cks[k].eo : cks[k].to;
            const float* pp = &outs[k].part[(((size_t) t * kPer + p) * kSegs + seg) * 130];
            cm.see(pp[128], m, m);
            cl.see(pp[129], l, l);   // the sum: relative
            if (l > 0 && pp[129] > 0) {
                double big = 0;
                for (int d = 0; d < 128; ++d) big = std::max(big, std::fabs(o[d] / l));
                for (int d = 0; d < 128; ++d) co.see(pp[d] / pp[129], o[d] / l, big);
            }
        }
    };
    // exact: the block's window list (zha_exact_kernel), every kSplit-th chunk a segment
    for (int b = 0; b < blocks; ++b) {
        const int t0 = b * kZhaTokens, tl = std::min(t0 + kZhaTokens, kRows) - 1;
        long long ffirst, flast;
        far_of(kPos + t0, &ffirst);
        const int nf_last = far_of(kPos + tl, &flast);
        const int c_ring = (int) (ffirst / kC), c_end = (int) ((kPos + tl) / kC) + 1;
        std::vector<int> list;
        for (int c = 0; c < c_end; ++c) {
            bool any = false;
            for (int t = t0; t <= tl; ++t) any = any || sel[(size_t) t * kMax + c];
            if (c == 0 || c >= c_ring || (c <= nf_last && any)) list.push_back(c);
        }
        for (int seg = 0; seg < kSplit; ++seg)
            for (int p = 0; p < kPer; ++p)
                for (int t = t0; t <= tl; ++t) {
                    double qv[128];
                    for (int d = 0; d < 128; ++d) qv[d] = qr(t, p, d, sc * kLog2e);
                    const long long own = kPos + t;
                    long long fr;
                    far_of(own, &fr);
                    double m = -INFINITY, l = 0, o[128] = {};
                    for (size_t k = seg; k < list.size(); k += kSplit) {
                        const int c = list[k];
                        if (!((long long) c * kC >= fr || c == 0 || sel[(size_t) t * kMax + c])) continue;
                        double s[64], kv[64][128], vv[64][128], mx = -INFINITY;
                        for (int j = 0; j < 64; ++j) {
                            for (int d = 0; d < 128; ++d) {
                                if (c >= c_ring) {
                                    const size_t at = (size_t) (((long long) c * kC) % kRing + j) * rrow + kG * 128;
                                    kv[j][d] = r16((double) rkc[at + d] * h2f(rks[at / 32 + d / 32]));
                                    vv[j][d] = r16((double) rvc[at + d] * h2f(rvs[at / 32 + d / 32]));
                                } else {
                                    const size_t key = (size_t) cmap[c] * kC + j;
                                    const int ck_ = d & 1 ? skc[key * 64 + d / 2] >> 4 : skc[key * 64 + d / 2] & 15;
                                    const int cv_ = d & 1 ? svc[key * 64 + d / 2] >> 4 : svc[key * 64 + d / 2] & 15;
                                    kv[j][d] = r16((double) (ck_ - 8) * h2f(sks[key * 4 + d / 32]));
                                    vv[j][d] = r16((double) (cv_ - 8) * h2f(svs[key * 4 + d / 32]));
                                }
                            }
                            const long long key = (long long) c * kC + j;
                            double v = 0;
                            for (int d = 0; d < 128; ++d) v += qv[d] * kv[j][d];
                            if (key > own) v = -INFINITY;
                            else if (own - key < kExtent) v += bias[((size_t) p * kRows + t) * kExtent + (own - key)] * kLog2e;
                            s[j] = v;
                            mx = std::max(mx, v);
                        }
                        const double now = std::max(m, mx), base = now == -INFINITY ? 0 : now, f = std::exp2(m - base);
                        m = now;
                        l *= f;
                        for (double& x : o) x *= f;
                        for (int j = 0; j < 64; ++j) {
                            const double e = std::exp2(s[j] - base);
                            l += e;
                            const double pe = r16(e);
                            for (int d = 0; d < 128; ++d) o[d] += pe * vv[j][d];
                        }
                    }
                    see_part(t, p, seg, m == -INFINITY ? -INFINITY : m * 0.6931471805599453, l, o, true);
                }
    }
    // tail: segment kSplit + y of chunks [cb, ce), tiles of 16 chunks; a chunk's weight on its best reference's row
    constexpr int CPT = 64 / kR;
    for (int b = 0; b < blocks; ++b) {
        const int t0 = b * kZhaTokens, tl = std::min(t0 + kZhaTokens, kRows) - 1;
        long long frl;
        const int nf_last = far_of(kPos + tl, &frl);
        const int span = ((nf_last + 1 + kTsplit - 1) / kTsplit + CPT - 1) / CPT * CPT;
        for (int y = 0; y < kTsplit; ++y) {
            const int cb = y * span, ce = std::min(nf_last + 1, cb + span);
            for (int p = 0; p < kPer; ++p)
                for (int t = t0; t <= tl; ++t) {
                    long long fr;
                    const int nf = far_of(kPos + t, &fr);
                    double m = -INFINITY, l = 0, o[128] = {};
                    for (int c0 = cb; c0 < ce; c0 += CPT) {
                        double mx = -INFINITY;
                        for (int c = c0; c < c0 + CPT; ++c)
                            if (c >= 1 && c <= nf && !sel[(size_t) t * kMax + c]) mx = std::max(mx, (double) tscore[p * hs + (size_t) t * kMax + c]);
                        if (mx == -INFINITY) continue;
                        const double now = std::max(m, mx), base = now == -INFINITY ? 0 : now, f = std::exp(m - base);
                        m = now;
                        l *= f;
                        for (double& x : o) x *= f;
                        for (int c = c0; c < c0 + CPT; ++c) {
                            if (!(c >= 1 && c <= nf && !sel[(size_t) t * kMax + c])) continue;
                            const size_t at = p * hs + (size_t) t * kMax + c;
                            const double e = std::exp(tscore[at] - base);
                            l += e;
                            const double pe = r16(e);
                            const size_t row = ((size_t) kG * kMax + c) * kR + trstar[at];
                            for (int d = 0; d < 128; ++d) o[d] += pe * (c < ce ? sdeq(vhat, row, d) : 0.0);
                        }
                    }
                    see_part(t, p, kSplit + y, m, l, o, false);
                }
        }
    }
    bool ok = true;
    for (int k = 0; k < nv; ++k) {
        std::printf("-- %s\n", outs[k].name);
        const Checks& c = cks[k];
        for (const Check* x : {&c.score, &c.rstar, &c.top, &c.em, &c.el, &c.eo, &c.tm, &c.tl, &c.to}) ok = x->report() && ok;
    }
    std::printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
