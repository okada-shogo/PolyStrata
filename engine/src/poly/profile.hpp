// src/poly/profile.hpp - the event profiler's figures (STRATA_PROFILE=1, --bench-report): what the card's stream did
// during a request's prompt and its answer, family by family, and what the host did meanwhile.
//
// The engine records an event on its stream each time the stream's work changes family (Engine::pmark()), and the
// time from one event to the next is that family's.  So a family's time is the stream's time from its first launch to
// the next family's first launch: the card's work on it, plus any wait for the host before the next launch reaches
// the card.  The waits have families of their own where the engine knows why it waits (wait_upload: the experts'
// copies into VRAM; wait_cpu: the CPU's experts; host_gap: the host's own work between two calls).  The events of a
// call follow one another on one stream, so the families add up to the stream's whole time, and with the gaps between
// calls to the phase's wall time.
//
// A prompt's every call is split.  An answer's tokens are split one in `every` (STRATA_PROFILE_EVERY, default 16):
// the others record two events, the call's start and end ("unsplit").  A split token is launched kernel by kernel,
// layer by layer, as the first tokens are before anything is recorded (a recording has no events in it): the same
// answers, and that one token a little slower.  No event goes into a recording, so the replays are as they were.
//
// With --layer-split the profile is the first card's: its stream and its calls (the later cards' engines record
// nothing).  The answer's tokens count as the window keeps them (Engine::keep()), not as fed with their guesses; a
// batch's step() counts its rows.
//
// This header has no GPU code: the aggregation and the JSON are tested on their own (bench_report_test.cpp).
#pragma once

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace strata::poly::prof {

enum Fam : int {
    kAttention = 0,   ///< attention scores, softmax and values (and a family's attention block it does not split)
    kKvStore,         ///< the keys' and values' convolutions, clips and stores into the caches
    kDense,           ///< dense products: attention projections, shared experts, controllers, the output head
    kNorms,           ///< norms, mixes, convolutions and residual sums
    kRouting,         ///< the router's product, its copy to the host and the host's routing
    kExpertsGpu,      ///< routed experts on the card, and their sum
    kCopyH2D,         ///< copies to the card in the stream: embeddings, tables, the CPU experts' sums
    kCopyD2H,         ///< copies to the host in the stream: the logits
    kWaitUpload,      ///< the stream waits for experts on their way into VRAM (and the host's staging of them)
    kWaitCpu,         ///< the stream waits for the CPU's experts
    kHostGap,         ///< the stream has nothing to do: the host's work between two calls
    kGraph,           ///< a recording replayed inside a split call (a batch step's, a block of guesses'): not split
    kUnsplit,         ///< an answer's token between its start and end events, not split (see above)
    kFams
};

enum Phase : int { kPrompt = 0, kDecode = 1, kPhases };

inline const char* fam_key(int f) {
    static const char* const k[kFams] = {"attention", "kv_store",    "dense",    "norms",       "routing",
                                         "experts_gpu", "copy_h2d",  "copy_d2h", "wait_upload", "wait_cpu",
                                         "host_gap",  "graph",       "unsplit"};
    return f >= 0 && f < kFams ? k[f] : "?";
}

inline const char* fam_title(int f) {
    static const char* const t[kFams] = {"attention",
                                         "KV store and clips",
                                         "dense products",
                                         "norms, mixes, convolutions",
                                         "routing",
                                         "experts on the GPU",
                                         "copies to the GPU",
                                         "copies to the host",
                                         "GPU waits for expert uploads",
                                         "GPU waits for CPU experts",
                                         "GPU idle, host between calls",
                                         "recorded launches, not split",
                                         "not split (sampled out)"};
    return f >= 0 && f < kFams ? t[f] : "?";
}

/// The families that are the card waiting, not working.
inline bool fam_idle(int f) { return f == kWaitUpload || f == kWaitCpu || f == kHostGap; }

inline const char* phase_key(int p) { return p == kPrompt ? "prompt" : "decode"; }

/// A set of calls' figures.
struct Acc {
    double fam_ms[kFams] = {};
    double fam_launches[kFams] = {};   ///< kernel launches in each family (see launch_counter())
    double wall_ms = 0;         ///< host clock from each call's start to its end, and the gaps between calls
    int64_t tokens = 0, calls = 0;
    double cpu_ms = 0;          ///< CPU experts (Stats::cpu_ms)
    double route_wait_ms = 0;   ///< the host waiting for the router's copy (Stats::route_wait_ms)
    int64_t cpu_experts = 0, gpu_experts = 0;
    double upload_bytes = 0;    ///< experts copied into VRAM in the stream (wait_upload's bytes)
    double side_ms = 0, side_bytes = 0;   ///< copies of experts on the upload stream, beside the work

    double timeline_ms() const {
        double s = 0;
        for (double v : fam_ms) s += v;
        return s;
    }
    double launches() const {
        double s = 0;
        for (double v : fam_launches) s += v;
        return s;
    }
    void add(const Acc& o) {
        for (int f = 0; f < kFams; ++f) fam_ms[f] += o.fam_ms[f], fam_launches[f] += o.fam_launches[f];
        wall_ms += o.wall_ms;
        tokens += o.tokens;
        calls += o.calls;
        cpu_ms += o.cpu_ms;
        route_wait_ms += o.route_wait_ms;
        cpu_experts += o.cpu_experts;
        gpu_experts += o.gpu_experts;
        upload_bytes += o.upload_bytes;
        side_ms += o.side_ms;
        side_bytes += o.side_bytes;
    }
};

/// Launch counting: every kernel launch the engine's own code makes, and every kernel node of a recording it replays.
/// The direct launches are counted where the build wraps the runtime's launch call (Linux: launch_count.cpp, linked
/// with --wrap); elsewhere only the recordings' nodes are.  A BLAS library's own launches inside one call are not
/// counted.  launch_counter() is 0 when counting is off.
inline std::atomic<uint64_t>& launch_count() {
    static std::atomic<uint64_t> n{0};
    return n;
}
/// The profiler is on (Engine::enable_profile()): only then does the wrapper count.
inline std::atomic<bool>& launch_counting() {
    static std::atomic<bool> on{false};
    return on;
}
/// launch_count.cpp's wrapper is linked into this program (it says so at start-up).
inline std::atomic<bool>& launch_wrapper_linked() {
    static std::atomic<bool> on{false};
    return on;
}
inline uint64_t launch_counter() { return launch_count().load(std::memory_order_relaxed); }
/// True where direct launches are counted (the wrapper is linked), false where only recordings' nodes are.
inline bool launches_wrapped() { return launch_wrapper_linked().load(std::memory_order_relaxed); }

/// The context lengths the figures are kept apart by: a call counts at the context it starts from.
inline const std::vector<int64_t>& bucket_edges() {
    static const std::vector<int64_t> e{0, 2048, 8192, 32768, 65536, 131072, 262144};
    return e;
}

inline int bucket_of(int64_t pos) {
    const auto& e = bucket_edges();
    int b = 0;
    while (b + 1 < (int) e.size() && pos >= e[(size_t) b + 1]) ++b;
    return b;
}

inline std::string bucket_name(int b) {
    const auto& e = bucket_edges();
    const auto k = [](int64_t v) { return std::to_string(v / 1024) + "K"; };
    if (b + 1 >= (int) e.size()) return k(e[(size_t) b]) + "+";
    return k(e[(size_t) b]) + "-" + k(e[(size_t) b + 1]);
}

/// A run's figures: every call (all), and the calls that were split (split), by phase and by context.
struct Profile {
    Acc all[kPhases], split[kPhases];
    std::vector<Acc> by_ctx_all[kPhases], by_ctx_split[kPhases];
    int every = 16;
    int64_t requests = 0;

    void reset() {
        const int e = every;
        *this = Profile{};
        every = e;
    }
    void add_call(int phase, int64_t pos, bool is_split, const Acc& a) {
        const size_t b = (size_t) bucket_of(pos);
        auto grow = [&](std::vector<Acc>& v) {
            if (v.size() <= b) v.resize(b + 1);
        };
        all[phase].add(a);
        grow(by_ctx_all[phase]);
        by_ctx_all[phase][b].add(a);
        if (is_split) {
            split[phase].add(a);
            grow(by_ctx_split[phase]);
            by_ctx_split[phase][b].add(a);
        }
    }
};

/// A small JSON writer: values in the order they are given, numbers rounded to `digits` decimals.
class Json {
public:
    std::string s;
    /// Applied to every string value (not to the keys) before it is escaped: the bench report's privacy scrub.
    std::string (*filter)(const std::string&) = nullptr;
    void open(const char* key = nullptr) { sep(key), s += '{', first_ = true; }
    void close() { s += '}', first_ = false; }
    void open_list(const char* key = nullptr) { sep(key), s += '[', first_ = true; }
    void close_list() { s += ']', first_ = false; }
    void str(const char* key, const std::string& v) { sep(key), quote(filter ? filter(v) : v); }
    void num(const char* key, double v, int digits = 3) {
        sep(key);
        if (!std::isfinite(v)) {
            s += "null";
            return;
        }
        char b[64];
        std::snprintf(b, sizeof b, "%.*f", digits, v);
        // drop the zeros a whole number does not need: 12.500 -> 12.5, 3.000 -> 3
        std::string t = b;
        if (t.find('.') != std::string::npos) {
            while (t.back() == '0') t.pop_back();
            if (t.back() == '.') t.pop_back();
        }
        if (t == "-0") t = "0";
        s += t;
    }
    void integer(const char* key, int64_t v) { sep(key), s += std::to_string(v); }
    void boolean(const char* key, bool v) { sep(key), s += v ? "true" : "false"; }
    void null(const char* key) { sep(key), s += "null"; }

private:
    bool first_ = true;
    void sep(const char* key) {
        if (!first_) s += ',';
        first_ = false;
        if (key) quote(key), s += ':';
    }
    void quote(const std::string& v) {
        s += '"';
        for (const unsigned char c : v) {
            if (c == '"' || c == '\\') s += '\\', s += (char) c;
            else if (c == '\n') s += "\\n";
            else if (c < 0x20) {
                char b[8];
                std::snprintf(b, sizeof b, "\\u%04x", c);
                s += b;
            } else s += (char) c;
        }
        s += '"';
    }
};

/// One Acc as JSON: per token where it has tokens, and the families' shares of the stream's time.
inline void acc_json(Json& j, const char* key, const Acc& a) {
    j.open(key);
    const double tok = (double) std::max<int64_t>(a.tokens, 1);
    const double tl = a.timeline_ms();
    j.integer("calls", a.calls);
    j.integer("tokens", a.tokens);
    j.num("wall_ms", a.wall_ms, 2);
    j.num("timeline_ms", tl, 2);
    j.num("tok_s", a.wall_ms > 0 ? 1000.0 * (double) a.tokens / a.wall_ms : 0.0, 2);
    j.num("cpu_experts_ms", a.cpu_ms, 2);
    j.num("router_wait_ms", a.route_wait_ms, 2);
    j.integer("cpu_experts", a.cpu_experts);
    j.integer("gpu_experts", a.gpu_experts);
    j.num("upload_gb", a.upload_bytes / 1e9, 4);
    // the upload stream's copies beside the work: their span, from the first copy to the last (with the host's
    // staging between them where there is no pinned mirror), not the link's time alone
    j.num("side_upload_span_ms", a.side_ms, 2);
    j.num("side_upload_gb", a.side_bytes / 1e9, 4);
    j.num("launches", a.launches(), 0);
    j.num("launches_per_token", a.launches() / tok, 1);
    j.open("ms");
    for (int f = 0; f < kFams; ++f)
        if (a.fam_ms[f] > 0) j.num(fam_key(f), a.fam_ms[f], 3);
    j.close();
    j.open("launches_by_family");
    for (int f = 0; f < kFams; ++f)
        if (a.fam_launches[f] > 0) j.num(fam_key(f), a.fam_launches[f], 0);
    j.close();
    j.open("ms_per_token");
    for (int f = 0; f < kFams; ++f)
        if (a.fam_ms[f] > 0) j.num(fam_key(f), a.fam_ms[f] / tok, 4);
    j.close();
    j.open("share");
    for (int f = 0; f < kFams; ++f)
        if (a.fam_ms[f] > 0 && tl > 0) j.num(fam_key(f), a.fam_ms[f] / tl, 4);
    j.close();
    j.close();
}

/// The whole profile as one JSON object (under `key`).
inline void profile_json(Json& j, const char* key, const Profile& p) {
    j.open(key);
    j.integer("split_every", p.every);
    j.str("launch_counting", launches_wrapped() ? "direct launches and recordings' nodes" : "recordings' nodes only");
    j.integer("requests", p.requests);
    for (int ph = 0; ph < kPhases; ++ph) {
        j.open(phase_key(ph));
        acc_json(j, "all", p.all[ph]);
        acc_json(j, "split", p.split[ph]);
        j.open_list("by_context");
        for (size_t b = 0; b < p.by_ctx_all[ph].size(); ++b) {
            if (p.by_ctx_all[ph][b].calls == 0) continue;
            j.open();
            j.str("context", bucket_name((int) b));
            acc_json(j, "all", p.by_ctx_all[ph][b]);
            if (b < p.by_ctx_split[ph].size() && p.by_ctx_split[ph][b].calls > 0)
                acc_json(j, "split", p.by_ctx_split[ph][b]);
            j.close();
        }
        j.close_list();
        j.close();
    }
    j.close();
}

}  // namespace strata::poly::prof
