// src/poly/bench_report.cpp - --bench-report's JSON and Markdown, the explain section and the privacy scrub
// (bench_report.hpp).  No GPU code: bench_report_test.cpp builds a report from figures of its own.
#include "bench_report.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <map>
#include <set>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <unistd.h>
#endif

#ifndef STRATA_GIT_COMMIT
#define STRATA_GIT_COMMIT "unknown"
#endif

namespace strata::poly::bench {

using prof::Acc;

const char* build_commit() { return STRATA_GIT_COMMIT; }
using prof::Json;

bool looks_like_path(const std::string& v) {
    if (v.empty()) return false;
    if (v.find('/') != std::string::npos || v.find('\\') != std::string::npos) return true;
    if (v[0] == '~' || v[0] == '.') return true;
    static const char* const ext[] = {".gguf", ".json", ".txt", ".bin", ".md", ".log", ".csv", ".ids"};
    for (const char* e : ext) {
        const size_t n = std::strlen(e);
        if (v.size() > n && v.compare(v.size() - n, n, e) == 0) return true;
    }
    return false;
}

std::vector<std::pair<std::string, std::string>> clean_args(const std::vector<std::string>& argv) {
    // options whose value is a file or a prompt, whatever it looks like
    static const std::set<std::string> files{"--model",       "--expert-profile", "--draft-model", "--pinned-session",
                                             "--images",      "--bench-report",   "--follow"};
    static const std::set<std::string> ids{"--tokens", "--eos-ids", "--follow"};
    std::vector<std::pair<std::string, std::string>> out;
    for (size_t i = 0; i < argv.size(); ++i) {
        const std::string& a = argv[i];
        if (a.rfind("--", 0) != 0) continue;   // a stray value: left out
        const bool has_value = i + 1 < argv.size() && argv[i + 1].rfind("--", 0) != 0;
        std::string v = has_value ? argv[i + 1] : "";
        if (has_value) ++i;
        if (a == "--model" || a == "--bench-report") continue;   // the model is described apart; the report is this file
        if (ids.count(a)) v = "<ids>";
        else if (files.count(a) || looks_like_path(v)) v = "<path>";
        out.emplace_back(a, v);
    }
    return out;
}

std::vector<std::pair<std::string, std::string>> clean_env(char** envp) {
    std::vector<std::pair<std::string, std::string>> out;
    for (char** e = envp; e && *e; ++e) {
        const std::string kv = *e;
        if (kv.rfind("STRATA_", 0) != 0) continue;
        const size_t eq = kv.find('=');
        if (eq == std::string::npos) continue;
        std::string k = kv.substr(0, eq), v = kv.substr(eq + 1);
        for (const char* hide : {"KEY", "TOKEN", "SECRET", "PASS", "AUTH"})
            if (k.find(hide) != std::string::npos) v = "<hidden>";
        if (looks_like_path(v)) v = "<path>";
        out.emplace_back(k, v);
    }
    std::sort(out.begin(), out.end());
    return out;
}

std::vector<std::string> private_strings() {
    std::vector<std::string> s;
    const auto add = [&](const std::string& v) {
        if (v.size() >= 4 && std::find(s.begin(), s.end(), v) == s.end()) s.push_back(v);
    };
    for (const char* k : {"USER", "USERNAME", "LOGNAME", "HOME", "USERPROFILE", "COMPUTERNAME", "HOSTNAME"})
        if (const char* v = std::getenv(k)) add(v);
#ifdef _WIN32
    char host[256] = {};
    DWORD n = sizeof host;
    if (GetComputerNameA(host, &n)) add(host);
#else
    char host[256] = {};
    if (gethostname(host, sizeof host - 1) == 0) add(host);
#endif
    std::error_code ec;
    const std::filesystem::path cwd = std::filesystem::current_path(ec);
    if (!ec) add(cwd.string());
    // longest first: a home folder goes before the user name inside it
    std::sort(s.begin(), s.end(), [](const std::string& a, const std::string& b) { return a.size() > b.size(); });
    return s;
}

std::string scrub(std::string text, const std::vector<std::string>& secrets) {
    const auto word = [](char c) { return std::isalnum((unsigned char) c) || c == '_'; };
    for (const std::string& x : secrets) {
        if (x.size() < 4) continue;
        // a folder anywhere; a name only as a whole word, so a user called "cuda" leaves the backend's name alone
        const bool folder = x.find('/') != std::string::npos || x.find('\\') != std::string::npos;
        for (size_t at = text.find(x); at != std::string::npos;) {
            const bool whole = folder || ((at == 0 || !word(text[at - 1])) &&
                                          (at + x.size() >= text.size() || !word(text[at + x.size()])));
            if (whole) {
                text.replace(at, x.size(), "<redacted>");
                at = text.find(x, at + 10);
            } else {
                at = text.find(x, at + 1);
            }
        }
    }
    return text;
}

namespace {

std::string fmt(const char* f, double v) {
    char b[64];
    std::snprintf(b, sizeof b, f, v);
    return b;
}

std::string pct(double share) { return fmt("%.0f%%", 100.0 * share); }

// one run's profiles of a phase, added
Acc sum_of(const std::vector<Run>& runs, int phase, bool split) {
    Acc a;
    for (const Run& r : runs) a.add(split ? r.profile.split[phase] : r.profile.all[phase]);
    return a;
}

double share(const Acc& a, int f) {
    const double tl = a.timeline_ms();
    return tl > 0 ? a.fam_ms[f] / tl : 0.0;
}

double busy_share(const Acc& a) {
    double s = 0;
    for (int f = 0; f < prof::kFams; ++f)
        if (!prof::fam_idle(f)) s += share(a, f);
    return s;
}

}  // namespace

namespace {

// `a` less `b`: the unsplit calls of a phase (every call less the split ones)
Acc minus(const Acc& a, const Acc& b) {
    Acc c = a;
    for (int f = 0; f < prof::kFams; ++f) {
        c.fam_ms[f] -= b.fam_ms[f], c.fam_launches[f] -= b.fam_launches[f];
        c.fam_graph_nodes[f] -= b.fam_graph_nodes[f], c.fam_bytes[f] -= b.fam_bytes[f];
    }
    c.wall_ms -= b.wall_ms, c.tokens -= b.tokens, c.calls -= b.calls, c.cpu_ms -= b.cpu_ms;
    c.route_wait_ms -= b.route_wait_ms, c.route_waits -= b.route_waits;
    c.cpu_experts -= b.cpu_experts, c.gpu_experts -= b.gpu_experts, c.staged_experts -= b.staged_experts;
    c.upload_bytes -= b.upload_bytes, c.side_ms -= b.side_ms, c.side_bytes -= b.side_bytes;
    c.cpu_expert_bytes -= b.cpu_expert_bytes;
    return c;
}

// One phase's figures into a diagnosis.  The whole token's (or call's) figures come from the calls the profiler left
// alone (`all` less `split`: replays as in normal use), where there are any; the families' from the split calls,
// launched kernel by kernel with their families timed.  `per` divides into the unit (tokens, or 1 for a call).
//
// Launch cost: a replayed recording's nodes run one after the other, so their probed cost adds to the token's time;
// a direct launch overlaps the kernels before it, so it only bounds the time where the kernels are no longer than a
// launch (the family's time a launch near the probed latency).  Launch figures are upper bounds.
Diagnosis diagnose_one(const Report& r, const std::string& phase, const std::string& unit, const Acc& all_in,
                       const Acc& split, double split_per, bool prompt) {
    Diagnosis g;
    g.phase = phase;
    g.unit = unit;
    const Probes& p = r.probes;
    if (all_in.calls == 0 || split.calls == 0 || split_per <= 0) return g;
    const Acc rest = minus(all_in, split);
    const bool use_rest = !prompt && rest.tokens > 0 && rest.calls > 0;
    const Acc& all = use_rest ? rest : all_in;
    const double all_per = prompt ? 1.0 : (double) all.tokens;
    if (all_per <= 0) return g;
    g.ok = true;
    g.total_ms = all.wall_ms / all_per;
    double nodes = 0;
    for (double v : all.fam_graph_nodes) nodes += v;
    g.launches = all.launches() / all_per;
    g.nodes = nodes / all_per;
    g.direct = g.launches - g.nodes;
    // a kernel's floor: the probed small dependent kernel (launched, or a replayed node), else the empty one's.  The
    // kernels of a stream run one after another, so each costs at least its floor: launches x floor is what the
    // kernel count alone costs, whatever the kernels do
    const double k_us = p.small_us > 0 ? p.small_us : p.launch_us;
    // (a token's kernels span the card: the wide kernel's floor where it is the higher)
    const double n_us = std::max(p.wide_node_us, p.small_node_us > 0 ? p.small_node_us : p.graph_node_us);
    g.launch_ms = (g.nodes * n_us + g.direct * k_us) / 1000.0;
    const double gpu_b = (all.fam_bytes[prof::kDense] + all.fam_bytes[prof::kRouting] + all.fam_bytes[prof::kAttention] +
                          all.fam_bytes[prof::kExpertsGpu]) / all_per;
    g.gpu_gb = gpu_b / 1e9;
    g.ideal_ms = p.read_gbs > 0 ? gpu_b / (p.read_gbs * 1e9) * 1000.0 : 0.0;
    g.wait_cpu_ms = split.fam_ms[prof::kWaitCpu] / split_per;
    g.wait_upload_ms = split.fam_ms[prof::kWaitUpload] / split_per;
    g.host_gap_ms = split.fam_ms[prof::kHostGap] / split_per;
    const double setup_ms = split.fam_ms[prof::kHostSetup] / split_per;
    const double looked = (double) (all.gpu_experts + all.cpu_experts) + all.staged_experts;
    if (looked > 0) g.hit_rate = (double) all.gpu_experts / looked;
    g.cpu_gb = all.cpu_expert_bytes / all_per / 1e9;
    g.warm_cache = !prompt && g.hit_rate > 0.97;
    g.pcie_gb = (all.upload_bytes + all.side_bytes) / all_per / 1e9;
    // host round trips (no doorbell: the host routes, from the scores it waits for): the routing family's time beyond
    // its kernels is the card's idle while the host waits, routes and launches the experts
    g.round_trips = all.route_waits / all_per;
    const double split_calls = (double) split.calls;
    {
        const double rms = split.fam_ms[prof::kRouting] / split_per;
        const double rl = split.fam_launches[prof::kRouting] / split_per;
        const double rkernels = std::min(rms, rl * k_us / 1000.0);
        g.round_trip_ms = split.route_waits > 0 ? std::max(0.0, rms - rkernels) : 0.0;
        if (split.route_waits <= 0) g.round_trips = 0;
    }
    (void) split_calls;
    // a split call that replays a recording books its time to "recorded launches": its bytes are then not the
    // families', and no GB/s is given a family
    const bool bytes_by_family = split.fam_ms[prof::kGraph] < 0.05 * split.timeline_ms();
    const auto fam_name = [](int f) { return std::string(prof::fam_title(f)); };
    std::vector<std::pair<double, std::string>> rec;
    for (int f = 0; f < prof::kFams; ++f) {
        const double ms = split.fam_ms[f] / split_per;
        if (ms <= 0.005 && split.fam_launches[f] <= 0) continue;
        FamilyVerdict v;
        v.fam = f;
        v.ms = ms;
        v.launches = split.fam_launches[f] / split_per;
        const double fnodes = split.fam_graph_nodes[f] / split_per;
        v.avg_kernel_us = v.launches > 0 ? 1000.0 * ms / v.launches : 0.0;
        v.launch_ms = std::min(ms, (fnodes * n_us + (v.launches - fnodes) * k_us) / 1000.0);
        const bool fam_direct_bound = k_us > 0 && v.launches > 0 && v.launch_ms >= 0.5 * ms;
        v.gb = bytes_by_family ? split.fam_bytes[f] / split_per / 1e9 : 0.0;
        v.gbs = ms > 0 ? v.gb / (ms / 1000.0) : 0.0;
        v.read_share = p.read_gbs > 0 ? v.gbs / p.read_gbs : 0.0;
        v.tiny = v.launches > 0 && k_us > 0 && v.avg_kernel_us < 2.0 * k_us;
        const double kernel_ms = std::max(0.0, ms - v.launch_ms);
        if (f == prof::kWaitCpu) v.verdict = "waiting on the CPU's experts";
        else if (f == prof::kRouting && split.route_waits > 0 && g.round_trip_ms > 0.5 * ms)
            v.verdict = "host round trips (" + fmt("%.0f", g.round_trips) + " " + unit +
                        "): the card idles while the host routes";
        else if (f == prof::kWaitUpload) v.verdict = "waiting on expert uploads";
        else if (f == prof::kHostGap) v.verdict = "idle: the host between calls";
        else if (f == prof::kHostSetup) v.verdict = "idle: the host setting the read up";
        else if (v.launches > 0 && fam_direct_bound && r.launches_wrapped)
            v.verdict = "launch-bound (" + fmt("%.1f", v.avg_kernel_us) + " us a kernel, the probed floor " +
                        fmt("%.1f", k_us) + " us)";
        else if (v.gb > 0 && v.read_share >= 0.6)
            v.verdict = "bandwidth-bound (" + pct(v.read_share) + " of the probed read bandwidth)";
        else if (v.gb > 0)
            v.verdict = "under bandwidth (" + pct(v.read_share) + " of the probed read): compute or latency in the kernels";
        else if (f == prof::kAttention && split.fam_bytes[prof::kAttention] <= 0 && ms > 0)
            v.verdict = "KV rows not modelled for this architecture: compute or latency";
        else if (ms > 0)
            v.verdict = "no byte model: compute or latency";
        if (!r.launches_wrapped && v.launches <= 0 && ms > 0) v.verdict += " (direct launches not counted in this build)";
        if (v.tiny) v.verdict += "; small kernels (" + fmt("%.1f", v.avg_kernel_us) + " us on average)";
        const bool routed_on_host = f == prof::kRouting && split.route_waits > 0;   // (round trips, below)
        if (v.gb > 0 && p.read_gbs > 0 && !routed_on_host) {
            const double gap = kernel_ms - v.gb * 1e9 / (p.read_gbs * 1e9) * 1000.0;
            if (gap > 0) rec.push_back({gap, "a faster " + fam_name(f) + " kernel (now " + pct(v.read_share) + " of the probed read bandwidth)"});
        }
        g.families.push_back(v);
    }
    if (g.launch_ms > 0)
        rec.push_back({g.launch_ms, "fewer launches: fused kernels or fewer recorded nodes (" + fmt("%.0f", g.launches) +
                                        " " + unit + ")"});
    if (g.wait_cpu_ms > 0.05)
        rec.push_back({g.wait_cpu_ms, "more experts in VRAM" + (g.hit_rate >= 0 ? " (hit rate " + pct(g.hit_rate) + ")" : std::string())});
    if (g.wait_upload_ms > 0.05) rec.push_back({g.wait_upload_ms, "faster or fewer expert uploads"});
    if (g.round_trip_ms > 0.05)
        rec.push_back({g.round_trip_ms, "fewer host round trips (" + fmt("%.0f", g.round_trips) + " " + unit +
                                            "): routing on the card, or the host's experts behind a doorbell"});
    if (g.host_gap_ms > 0.05) rec.push_back({g.host_gap_ms, "less host work between calls"});
    if (setup_ms > 0.05)
        rec.push_back({setup_ms, "less host setup before a read (the caches' growth, staging taken back from the expert "
                                 "cache)"});
    std::sort(rec.begin(), rec.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
    if (rec.size() > 3) rec.resize(3);
    g.recover = rec;
    const double T = g.total_ms, waits = g.wait_cpu_ms + g.wait_upload_ms;
    if (k_us <= 0 || p.read_gbs <= 0)
        g.verdict = "not judged: the kernel probes did not run or failed";
    else if (g.launch_ms >= 0.4 * T && g.launch_ms >= g.ideal_ms && g.launch_ms >= waits)
        g.verdict = "launch-bound: " + fmt("%.0f", g.launches) + " kernels " + unit + " at the probed floor (" +
                    fmt("%.1f", n_us) + " us a replayed node, " + fmt("%.1f", k_us) + " us launched) cost ~" +
                    fmt("%.1f", g.launch_ms) + " of " + fmt("%.1f", T) + " ms";
    else if (g.wait_cpu_ms >= 0.3 * T && g.wait_cpu_ms >= g.ideal_ms)
        g.verdict = "waiting on CPU experts: " + fmt("%.1f", g.wait_cpu_ms) + " of " + fmt("%.1f", T) + " ms";
    else if (g.round_trip_ms >= 0.25 * T && g.round_trip_ms >= g.ideal_ms)
        g.verdict = "host round trips: the host waits for router scores " + fmt("%.0f", g.round_trips) + " times " + unit +
                    " and the card idles ~" + fmt("%.1f", g.round_trip_ms) + " of " + fmt("%.1f", T) + " ms meanwhile";
    else if (g.wait_upload_ms >= 0.3 * T)
        g.verdict = "waiting on expert uploads: " + fmt("%.1f", g.wait_upload_ms) + " of " + fmt("%.1f", T) + " ms";
    else if (g.ideal_ms >= 0.6 * T)
        g.verdict = "bandwidth-bound: " + fmt("%.2f", g.gpu_gb) + " GB " + unit + " at the probed read bandwidth take " +
                    fmt("%.1f", g.ideal_ms) + " of " + fmt("%.1f", T) + " ms";
    else
        g.verdict = "compute- or latency-bound: launches up to ~" + fmt("%.1f", g.launch_ms) + " ms, bytes at the probed "
                    "bandwidth " + fmt("%.1f", g.ideal_ms) + " ms, waits " + fmt("%.1f", waits) + " ms, of " +
                    fmt("%.1f", T) + " ms";
    if (!r.launches_wrapped) g.verdict += " (direct launches are not counted in this build: replays only)";
    return g;
}

}  // namespace

std::vector<Diagnosis> diagnose(const Report& r) {
    std::vector<Diagnosis> out;
    std::vector<Run> singles;
    for (const Run& run : r.runs)
        if (run.single) singles.push_back(run);
    const std::vector<Run>& dec = singles.empty() ? r.runs : singles;
    const Acc d = sum_of(dec, prof::kDecode, false), ds = sum_of(dec, prof::kDecode, true);
    Diagnosis dg = diagnose_one(r, singles.empty() ? "decode" : "decode, a token a call", "a token", d, ds,
                                (double) ds.tokens, false);
    if (dg.ok) out.push_back(dg);
    const Run* big = nullptr;
    for (const Run& run : r.runs)
        if (!big || run.read > big->read) big = &run;
    if (big) {
        const Acc& a = big->profile.all[prof::kPrompt];
        const Acc& sp = big->profile.split[prof::kPrompt];
        Diagnosis pg = diagnose_one(r, "prompt " + std::to_string(big->read), "a prompt", a, sp, 1.0, true);
        if (pg.ok) out.push_back(pg);
    }
    return out;
}

std::vector<Finding> explain(const Report& r) {
    std::vector<Finding> out;
    const Probes& p = r.probes;
    const double expert_b = r.model.expert_mib * 1048576.0;
    // The prompt: the quick tier's longest read
    const Run* big = nullptr;
    for (const Run& run : r.runs)
        if (!big || run.read > big->read) big = &run;
    if (big && big->profile.split[prof::kPrompt].calls > 0) {
        const Acc& a = big->profile.split[prof::kPrompt];
        const std::string ph = "prompt " + std::to_string(big->read);
        const double up = share(a, prof::kWaitUpload), wait_ms = a.fam_ms[prof::kWaitUpload];
        if (wait_ms > 0 || a.side_bytes > 0) {
            // copies in the stream: their rate with the host's staging; beside it: their own rate while copying
            const double in_gbs = a.upload_bytes > 0 && wait_ms > 0 ? a.upload_bytes / 1e9 / (wait_ms / 1000.0) : 0.0;
            const double side_gbs = a.side_ms > 0 ? a.side_bytes / 1e9 / (a.side_ms / 1000.0) : 0.0;
            std::string t = pct(up) + " of the GPU's time (" + fmt("%.0f", wait_ms) + " ms) waits for expert uploads";
            if (a.upload_bytes > 0)
                t += "; " + fmt("%.1f", a.upload_bytes / 1e9) + " GB copied in the stream at " + fmt("%.1f", in_gbs) +
                     " GB/s with the host's staging";
            if (a.side_bytes > 0)
                t += "; " + fmt("%.1f", a.side_bytes / 1e9) + " GB copied beside the work, " + fmt("%.1f", side_gbs) +
                     " GB/s over the copies' span";
            t += "; probed H2D " + fmt("%.1f", p.h2d_pinned_gbs) + " GB/s pinned, " + fmt("%.1f", p.h2d_pageable_gbs) +
                 " GB/s pageable";
            Finding f{ph, t};
            // only the copies in the stream are a rate the stream waited on; the side copies' span holds their own
            // waits and staging, so it says nothing against the link
            f.flag = up > 0.2 && p.h2d_pinned_gbs > 0 && a.upload_bytes > 0 && in_gbs > 0 && in_gbs < 0.6 * p.h2d_pinned_gbs;
            if (f.flag) f.text += " (the uploads run well under the link's speed)";
            out.push_back(f);
        }
        out.push_back({ph, "GPU work " + pct(busy_share(a)) + " of the time: experts " +
                               pct(share(a, prof::kExpertsGpu)) + ", dense products " + pct(share(a, prof::kDense)) +
                               ", attention " + pct(share(a, prof::kAttention)) + ", norms/mixes " +
                               pct(share(a, prof::kNorms)) + ", KV store " + pct(share(a, prof::kKvStore)) +
                               ", routing " + pct(share(a, prof::kRouting))});
        const double h2d = share(a, prof::kCopyH2D);
        if (h2d > 0.1) {   // tables, embeddings and the CPU experts' sums: a few MB, so a share like this is a slow path
            Finding f{ph, pct(h2d) + " of the GPU's time (" + fmt("%.0f", a.fam_ms[prof::kCopyH2D]) +
                              " ms) is copies to the GPU in the stream (tables, embeddings, the CPU experts' sums), "
                              "a few MB that should take a few ms"};
            f.flag = h2d > 0.2;
            out.push_back(f);
        }
        const double gap = share(a, prof::kHostGap), cpu = share(a, prof::kWaitCpu);
        if (gap > 0.05) {
            Finding f{ph, pct(gap) + " of the GPU's time it is idle between calls (the host's own work)"};
            f.flag = gap > 0.15;
            out.push_back(f);
        }
        const double setup = share(a, prof::kHostSetup);
        if (setup > 0.05) {
            Finding f{ph, pct(setup) + " of the GPU's time (" + fmt("%.0f", a.fam_ms[prof::kHostSetup]) +
                              " ms) it is idle while the host sets a read up (the caches' growth, staging taken back "
                              "from the expert cache)"};
            f.flag = setup > 0.15;
            out.push_back(f);
        }
        if (cpu > 0.05) {
            Finding f{ph, pct(cpu) + " of the GPU's time waits for the CPU's experts (" +
                              fmt("%.0f", a.cpu_ms) + " ms of CPU experts)"};
            f.flag = cpu > 0.25;
            out.push_back(f);
        }
    }
    // The answers: the quick runs' decode tokens together, the ones written a token a call where there are any
    std::vector<Run> singles;
    for (const Run& run : r.runs)
        if (run.single) singles.push_back(run);
    const std::vector<Run>& dec = singles.empty() ? r.runs : singles;
    const Acc d = sum_of(dec, prof::kDecode, false), ds = sum_of(dec, prof::kDecode, true);
    if (d.tokens > 0) {
        const double tok = (double) d.tokens, ms_tok = d.wall_ms / tok;
        const std::string ph = singles.empty() ? "decode" : "decode, a token a call";
        // A split token is launched kernel by kernel: where the others replay recordings, its launch gaps are in its
        // families, and on a fast card they can be most of them
        if (ds.tokens > 0 && d.tokens > ds.tokens) {
            const double split_ms = ds.wall_ms / (double) ds.tokens;
            const double rest_ms = (d.wall_ms - ds.wall_ms) / (double) (d.tokens - ds.tokens);
            if (rest_ms > 0 && split_ms > 1.3 * rest_ms)
                out.push_back({ph, "a split token took " + fmt("%.1f", split_ms) + " ms against " + fmt("%.1f", rest_ms) +
                                       " ms for the others (" + fmt("%.1f", split_ms / rest_ms) +
                                       "x): launched kernel by kernel, its families hold the launch gaps too, so read "
                                       "their shares as kernels plus launches"});
        }
        out.push_back({ph, fmt("%.1f", ms_tok) + " ms a token (" + fmt("%.1f", 1000.0 / std::max(ms_tok, 1e-9)) +
                               " tok/s over the quick runs); split tokens: GPU work " + pct(busy_share(ds)) +
                               ", waits for CPU experts " + pct(share(ds, prof::kWaitCpu)) + ", for uploads " +
                               pct(share(ds, prof::kWaitUpload)) + ", host gaps " + pct(share(ds, prof::kHostGap)) +
                               (ds.fam_ms[prof::kGraph] > 0 ? ", not split " + pct(share(ds, prof::kGraph)) : "")});
        if (d.cpu_experts > 0 && d.cpu_ms > 0) {
            const double per_tok = (double) d.cpu_experts / tok;
            const double gbs = expert_b > 0 ? (double) d.cpu_experts * expert_b / 1e9 / (d.cpu_ms / 1000.0) : 0.0;
            const double dot = p.dots.empty() ? 0.0 : p.dots.front().gbs_all;
            const double limit = dot > 0 && p.ram_read_gbs > 0 ? std::min(dot, p.ram_read_gbs) : std::max(dot, p.ram_read_gbs);
            Finding f{ph, "CPU experts " + fmt("%.1f", d.cpu_ms / tok) + " ms a token for " + fmt("%.0f", per_tok) +
                              " experts (" + fmt("%.0f", per_tok * expert_b / 1048576.0) + " MiB): " + fmt("%.1f", gbs) +
                              " GB/s of weights; probed dot kernel " + fmt("%.1f", dot) + " GB/s on " +
                              std::to_string(p.dots.empty() ? 0 : p.dots.front().threads) + " threads, RAM read " +
                              fmt("%.1f", p.ram_read_gbs) + " GB/s"};
            f.flag = limit > 0 && gbs > 0 && gbs < 0.5 * limit && share(ds, prof::kWaitCpu) > 0.2;
            if (f.flag) f.text += " (the CPU experts run well under what the probes reach)";
            out.push_back(f);
        }
        if (d.launches() > 0)
            out.push_back({ph, fmt("%.0f", d.launches() / tok) + " kernel launches a token (" +
                                   fmt("%.1f", d.launches() > 0 ? 1000.0 * ms_tok / (d.launches() / tok) : 0.0) +
                                   " us of the token's time each)"});
        if (d.route_wait_ms > 0)
            out.push_back({ph, "router wait " + fmt("%.2f", d.route_wait_ms / tok) + " ms a token: the host waiting for "
                                   "each layer's router scores, which is the GPU's work up to the router (GPU time, "
                                   "not idle) where the host routes"});
        const double gap = share(ds, prof::kHostGap);
        if (gap > 0.1) {
            Finding f{ph, pct(gap) + " of a split token's time the GPU is idle between calls (sampling, guesses, "
                                     "the host's bookkeeping)"};
            f.flag = gap > 0.2;
            out.push_back(f);
        }
        const double upw = share(ds, prof::kWaitUpload);
        if (upw > 0.1) {
            Finding f{ph, pct(upw) + " of a split token's time the GPU waits for expert uploads"};
            f.flag = upw > 0.25;
            out.push_back(f);
        }
    }
    // --long: decode against the context
    if (r.long_points.size() >= 2) {
        const Run& a = r.long_points.front();
        const Run& b = r.long_points.back();
        const Acc& sa = a.profile.split[prof::kDecode];
        const Acc& sb = b.profile.split[prof::kDecode];
        Finding f{"long", "decode " + fmt("%.1f", a.decode_tok_s()) + " tok/s at " + std::to_string(a.context_end / 1024) +
                              "K, " + fmt("%.1f", b.decode_tok_s()) + " tok/s at " + std::to_string(b.context_end / 1024) +
                              "K; attention " + pct(share(sa, prof::kAttention)) + " -> " +
                              pct(share(sb, prof::kAttention)) + " of a split token"};
        f.flag = a.decode_tok_s() > 0 && b.decode_tok_s() < 0.5 * a.decode_tok_s();
        out.push_back(f);
    }
    return out;
}

namespace {

void gpu_json(Json& j, const Gpu& g) {
    j.open("gpu");
    j.str("name", g.name);
    j.str("backend", g.backend);
    j.str("arch", g.arch);
    j.integer("sms_or_cus", g.sms);
    j.num("vram_gib", g.vram_gib, 2);
    j.num("mem_peak_gbs", g.mem_peak_gbs, 1);
    j.str("driver", g.driver);
    j.str("runtime", g.runtime);
    if (g.pcie_gen > 0) j.integer("pcie_gen", g.pcie_gen); else j.null("pcie_gen");
    if (g.pcie_width > 0) j.integer("pcie_width", g.pcie_width); else j.null("pcie_width");
    if (g.pcie_gen_max > 0) j.integer("pcie_gen_max", g.pcie_gen_max); else j.null("pcie_gen_max");
    if (g.pcie_width_max > 0) j.integer("pcie_width_max", g.pcie_width_max); else j.null("pcie_width_max");
    j.str("pcie_source", g.pcie_source.empty() ? "none" : g.pcie_source);
    j.close();
}

void host_json(Json& j, const Host& h) {
    j.open("cpu");
    j.str("model", h.cpu);
    j.str("os", h.os);
    j.str("isa", h.isa);
    j.open_list("flags");
    for (const std::string& f : h.flags) j.str(nullptr, f);
    j.close_list();
    j.integer("threads", h.threads);
    j.integer("engine_threads", h.engine_threads);
    j.num("ram_gib", h.ram_gib, 1);
    j.close();
}

void probes_json(Json& j, const Probes& p) {
    j.open("probes");
    j.num("h2d_pinned_gbs", p.h2d_pinned_gbs, 2);
    j.num("h2d_pageable_gbs", p.h2d_pageable_gbs, 2);
    j.num("d2h_pinned_gbs", p.d2h_pinned_gbs, 2);
    j.num("gpu_mem_gbs", p.gpu_mem_gbs, 1);
    j.num("ram_read_gbs", p.ram_read_gbs, 1);
    j.num("launch_us", p.launch_us, 2);
    j.num("graph_node_us", p.graph_node_us, 2);
    j.num("small_kernel_us", p.small_us, 2);
    j.num("small_kernel_node_us", p.small_node_us, 2);
    j.num("wide_kernel_node_us", p.wide_node_us, 2);
    j.num("sync_round_trip_us", p.sync_us, 2);
    j.num("copy_to_host_sync_us", p.copy_sync_us, 2);
    j.num("gpu_read_gbs", p.read_gbs, 1);
    j.num("fp32_fma_gflops", p.fma_gflops, 0);
    j.integer("ram_threads", p.ram_threads);
    j.open_list("cpu_dot");
    for (const DotRate& d : p.dots) {
        j.open();
        j.str("type", d.type);
        j.str("kernel", d.kernel);
        j.str("isa", d.isa);
        j.integer("threads", d.threads);
        j.num("gbs_all_threads", d.gbs_all, 2);
        j.num("gbs_one_thread", d.gbs_one, 2);
        j.close();
    }
    j.close_list();
    j.num("seconds", p.seconds, 1);
    j.close();
}

void model_json(Json& j, const Model& m) {
    j.open("model");
    j.str("name", m.name);
    j.str("arch", m.arch);
    j.str("expert_types", m.file_type);
    j.num("size_gib", m.size_gib, 2);
    j.integer("experts", m.n_expert);
    j.integer("experts_used", m.n_used);
    j.integer("moe_layers", m.n_moe);
    j.num("expert_mib", m.expert_mib, 3);
    j.close();
}

void pairs_json(Json& j, const char* key, const std::vector<std::pair<std::string, std::string>>& v) {
    j.open_list(key);
    for (const auto& [k, x] : v) {
        j.open_list();
        j.str(nullptr, k);
        j.str(nullptr, x);
        j.close_list();
    }
    j.close_list();
}

void run_json(Json& j, const Run& r) {
    j.open();
    j.str("name", r.name);
    j.boolean("one_token_a_call", r.single);
    j.integer("prompt_tokens", r.prompt_tokens);
    j.integer("reused", r.reused);
    j.integer("read", r.read);
    j.num("prompt_ms", r.prompt_ms, 1);
    j.num("prompt_tok_s", r.prompt_tok_s(), 1);
    j.integer("produced", r.produced);
    j.num("decode_ms", r.decode_ms, 1);
    j.num("decode_tok_s", r.decode_tok_s(), 2);
    j.integer("guessed", r.guessed);
    j.integer("guessed_right", r.guessed_right);
    j.integer("context_end", r.context_end);
    prof::profile_json(j, "profile", r.profile);
    j.close();
}

void header_json(Json& j, const Report& r, const char* kind) {
    j.str("kind", kind);
    j.integer("schema", 1);
    j.open("engine");
    j.str("version", r.engine_version);
    j.str("commit", r.commit);
    j.close();
    model_json(j, r.model);
    j.open("settings");
    pairs_json(j, "args", r.settings);
    pairs_json(j, "env", r.env);
    j.integer("max_context", r.max_context);
    j.close();
    j.open("hardware");
    gpu_json(j, r.gpu);
    host_json(j, r.host);
    j.close();
    j.integer("cached_experts", r.cached_experts);
    j.integer("total_experts", r.total_experts);
    j.num("cached_gib", r.cached_gib, 2);
}

}  // namespace

namespace {
const std::vector<std::string>* json_secrets = nullptr;   // to_json() / session_json() set it for Json::filter
std::string json_scrub(const std::string& v) { return json_secrets ? scrub(v, *json_secrets) : v; }
}  // namespace

std::string to_json(const Report& r) {
    const std::vector<std::string> secrets = private_strings();
    json_secrets = &secrets;
    Json j;
    j.filter = json_scrub;
    j.open();
    header_json(j, r, "strata-bench-report");
    j.str("tier", r.tier);
    j.str("prompts", r.prompt_source);
    j.boolean("direct_launches_counted", r.launches_wrapped);
    j.str("measures", "this build and these settings, one request at a time after a warm-up; decode a token a call = "
                      "the same answer with no guesses");
    probes_json(j, r.probes);
    j.open_list("runs");
    for (const Run& run : r.runs) run_json(j, run);
    j.close_list();
    j.open_list("long");
    for (const Run& run : r.long_points) run_json(j, run);
    j.close_list();
    if (!r.long_error.empty()) j.str("long_error", r.long_error);
    j.open_list("diagnosis");
    for (const Diagnosis& g : diagnose(r)) {
        j.open();
        j.str("phase", g.phase);
        j.str("verdict", g.verdict);
        j.num("ms", g.total_ms, 3);
        j.num("launches", g.launches, 1);
        j.num("direct_launches", g.direct, 1);
        j.num("replayed_nodes", g.nodes, 1);
        j.num("launch_ms_estimate", g.launch_ms, 3);
        j.num("gpu_gb", g.gpu_gb, 4);
        j.num("ideal_ms_at_probed_read", g.ideal_ms, 3);
        j.num("wait_cpu_ms", g.wait_cpu_ms, 3);
        j.num("wait_upload_ms", g.wait_upload_ms, 3);
        j.num("host_gap_ms", g.host_gap_ms, 3);
        j.num("host_round_trips", g.round_trips, 1);
        j.num("round_trip_idle_ms", g.round_trip_ms, 3);
        j.boolean("hit_rate_above_real_text", g.warm_cache);
        if (g.hit_rate >= 0) j.num("expert_hit_rate", g.hit_rate, 4); else j.null("expert_hit_rate");
        j.num("cpu_expert_gb", g.cpu_gb, 4);
        j.num("pcie_gb", g.pcie_gb, 4);
        j.open_list("families");
        for (const FamilyVerdict& v : g.families) {
            j.open();
            j.str("family", prof::fam_key(v.fam));
            j.num("ms", v.ms, 3);
            j.num("launches", v.launches, 1);
            j.num("launch_ms_estimate", v.launch_ms, 3);
            j.num("gb", v.gb, 4);
            j.num("gbs", v.gbs, 1);
            j.num("share_of_probed_read", v.read_share, 3);
            j.num("avg_kernel_us", v.avg_kernel_us, 2);
            j.boolean("tiny_kernels", v.tiny);
            j.str("verdict", v.verdict);
            j.close();
        }
        j.close_list();
        j.open_list("recover");
        for (const auto& [ms, what] : g.recover) {
            j.open();
            j.num("ms", ms, 3);
            j.str("by", what);
            j.close();
        }
        j.close_list();
        j.close();
    }
    j.close_list();
    j.open_list("explain");
    for (const Finding& f : explain(r)) {
        j.open();
        j.str("phase", f.phase);
        j.str("text", f.text);
        j.boolean("flag", f.flag);
        j.close();
    }
    j.close_list();
    j.open("seconds");
    j.num("probes", r.probe_s, 1);
    j.num("load", r.load_s, 1);
    j.num("warm_up", r.warm_s, 1);
    j.num("quick", r.quick_s, 1);
    j.num("long", r.long_s, 1);
    j.num("total", r.total_s, 1);
    j.close();
    j.close();
    json_secrets = nullptr;
    return j.s + "\n";   // (scrubbed value by value: a key that happens to be a name stays a key)
}

std::string session_json(const Report& r, const prof::Profile& p) {
    const std::vector<std::string> secrets = private_strings();
    json_secrets = &secrets;
    Json j;
    j.filter = json_scrub;
    j.open();
    header_json(j, r, "strata-session-profile");
    prof::profile_json(j, "profile", p);
    j.close();
    json_secrets = nullptr;
    return j.s + "\n";
}

namespace {

// A value in a table cell or a line: no cell breaks, no line breaks, no HTML, no headings, links or images.
std::string md_escape(const std::string& s) {
    std::string o;
    for (char c : s) {
        if (c == '\n' || c == '\r') o += ' ';
        else if (c == '<') o += "&lt;";
        else if (c == '>') o += "&gt;";
        else if (c == '|' || c == '#' || c == '!' || c == '[' || c == ']' || c == '*' || c == '_' || c == '`' ||
                 c == '\\')
            o += '\\', o += c;
        else o += c;
    }
    return o;
}

std::string num(double v, int digits) { return fmt(digits == 0 ? "%.0f" : digits == 1 ? "%.1f" : "%.2f", v); }

void profile_table(std::string& m, const Run& run, int phase) {
    const Acc& a = run.profile.split[phase];
    if (a.calls == 0) return;
    const double tl = a.timeline_ms(), tok = (double) std::max<int64_t>(a.tokens, 1);
    const bool prompt = phase == prof::kPrompt;
    m += "\n" + md_escape(run.name) + ", " + prof::phase_key(phase) + " (" + std::to_string(a.calls) + " split calls, " +
         std::to_string(a.tokens) + " tokens):\n\n";
    m += prompt ? "| Family | ms | Launches | Share |\n| --- | ---: | ---: | ---: |\n"
                : "| Family | ms a token | Launches a token | Share |\n| --- | ---: | ---: | ---: |\n";
    for (int f = 0; f < prof::kFams; ++f) {
        if (a.fam_ms[f] <= 0 && a.fam_launches[f] <= 0) continue;
        const double v = prompt ? a.fam_ms[f] : a.fam_ms[f] / tok;
        const double n = prompt ? a.fam_launches[f] : a.fam_launches[f] / tok;
        m += "| " + std::string(prof::fam_title(f)) + " | " + num(v, prompt ? 0 : 2) + " | " + num(n, prompt ? 0 : 1) +
             " | " + pct(tl > 0 ? a.fam_ms[f] / tl : 0) + " |\n";
    }
    m += "| (all) | " + num(prompt ? tl : tl / tok, prompt ? 0 : 2) + " | " +
         num(prompt ? a.launches() : a.launches() / tok, prompt ? 0 : 1) + " | 100% |\n";
    if (!prompt) {   // every token, the replayed ones too: what the card is handed for a token
        const Acc& all = run.profile.all[phase];
        if (all.tokens > 0 && all.launches() > 0)
            m += "\nLaunches a token over all " + std::to_string(all.tokens) + " tokens (replayed ones included): " +
                 num(all.launches() / (double) all.tokens, 0) + ".\n";
    }
}

}  // namespace

std::string to_markdown(const Report& r) {
    std::string m;
    const Gpu& g = r.gpu;
    const Host& h = r.host;
    m += "# strata-poly bench report (" + r.tier + ")\n\n";
    m += "Engine " + md_escape(r.engine_version) + " (" + md_escape(r.commit) + "), " + md_escape(r.model.name) + ", " + md_escape(g.name) +
         ".\n\n";
    m += "**What this measures:** this engine build with the settings listed below, on the prompts named under Results, "
         "one request at a time after a warm-up. Speeds from another build, other settings (`--kv`, `--window`, "
         "`--prompt-int8`, the `STRATA_*` switches), other prompts or a cold engine are not comparable with these. "
         "\"Decode, one token a call\" writes the same answer with no guesses (`--window 1` and no lookups): what the "
         "hardware does for a token. The plain decode column includes the settings' guesses, which depend on the text.\n\n";
    m += "## Hardware\n\n| Part | Value |\n| --- | --- |\n";
    m += "| GPU | " + md_escape(g.name) + " (" + md_escape(g.arch) + ", " + std::to_string(g.sms) + " " +
         (g.backend == "hip" ? "CUs" : "SMs") + ", " + num(g.vram_gib, 1) + " GiB) |\n";
    m += "| GPU driver | " + md_escape(g.driver) + ", runtime " + md_escape(g.runtime) + " |\n";
    m += "| PCIe | " +
         (g.pcie_gen > 0 ? "gen" + std::to_string(g.pcie_gen) + " x" + std::to_string(g.pcie_width) +
                               (g.pcie_gen_max > 0 ? " (up to gen" + std::to_string(g.pcie_gen_max) + " x" +
                                                         std::to_string(g.pcie_width_max) + ")"
                                                   : std::string())
                         : std::string("not known")) +
         " |\n";
    m += "| CPU | " + md_escape(h.cpu) + ", " + std::to_string(h.threads) + " threads (" +
         std::to_string(h.engine_threads) + " for the experts), " + h.isa + " |\n";
    m += "| RAM | " + num(h.ram_gib, 1) + " GiB |\n";
    m += "| OS | " + md_escape(h.os) + " |\n";
    m += "\n## Probes\n\n| Probe | GB/s |\n| --- | ---: |\n";
    m += "| H2D pinned | " + num(r.probes.h2d_pinned_gbs, 1) + " |\n";
    m += "| H2D pageable | " + num(r.probes.h2d_pageable_gbs, 1) + " |\n";
    m += "| D2H pinned | " + num(r.probes.d2h_pinned_gbs, 1) + " |\n";
    m += "| GPU memory (copy, read + write) | " + num(r.probes.gpu_mem_gbs, 0) +
         // the peak is the memory clock times the bus width as the driver reports them: on AMD the clock is not the
         // data rate (GDDR6 moves 8 bits a pin per clock), so a peak under the measured copy is left out
         (g.mem_peak_gbs > 0 && g.mem_peak_gbs >= r.probes.gpu_mem_gbs ? " (peak " + num(g.mem_peak_gbs, 0) + ")"
                                                                        : std::string()) + " |\n";
    if (r.probes.read_gbs > 0) m += "| GPU memory, read-only kernel | " + num(r.probes.read_gbs, 0) + " |\n";
    m += "| RAM read, " + std::to_string(r.probes.ram_threads) + " threads | " + num(r.probes.ram_read_gbs, 1) + " |\n";
    for (const DotRate& d : r.probes.dots)
        m += "| CPU expert dot " + d.type + " (" + d.kernel + ", " + d.isa + "), " + std::to_string(d.threads) +
             " threads / 1 | " + num(d.gbs_all, 1) + " / " + num(d.gbs_one, 1) + " |\n";
    if (r.probes.launch_us > 0)
        m += "\nLaunch: " + fmt("%.2f", r.probes.small_us) + " us a small dependent kernel back to back (" +
             fmt("%.2f", r.probes.small_node_us) + " us replayed, " + fmt("%.2f", r.probes.wide_node_us) + " us replayed across the card): a kernel's floor; " + fmt("%.2f", r.probes.launch_us) +
             " us an empty kernel back to back, " +
             fmt("%.2f", r.probes.graph_node_us) + " us a node of a replayed recording, " + fmt("%.1f", r.probes.sync_us) + " us a host round trip (kernel, "
             "event, wait), " + fmt("%.1f", r.probes.copy_sync_us) + " us a small copy to the host waited for; fp32 FMA " +
             num(r.probes.fma_gflops, 0) + " GFLOP/s.\n";
    m += "\n## Model and settings\n\n";
    m += md_escape(r.model.name) + " (" + md_escape(r.model.arch) + ", experts " + md_escape(r.model.file_type) + ", " +
         num(r.model.size_gib, 1) + " GiB); " + std::to_string(r.cached_experts) + " of " +
         std::to_string(r.total_experts) + " routed experts in VRAM (" + num(r.cached_gib, 1) +
         " GiB); context " + std::to_string(r.max_context) + "; loaded in " + num(r.load_s, 1) + " s.\n\n";
    m += "Arguments:";
    if (r.settings.empty()) m += " none";
    const auto code = [](std::string v) {   // inside a code span: no backtick, no line break
        for (char& c : v)
            if (c == '`' || c == '\n' || c == '\r') c = '\'';
        return v;
    };
    for (const auto& [k, v] : r.settings) m += " `" + code(k + (v.empty() ? "" : " " + v)) + "`";
    m += "\n\nEnvironment:";
    if (r.env.empty()) m += " none";
    for (const auto& [k, v] : r.env) m += " `" + code(k + "=" + v) + "`";
    m += "\n\n## Results\n\nGreedy. Prompts: " + md_escape(r.prompt_source) +
         ". Prompt tok/s counts the tokens read; decode tok/s the tokens after the first, with the settings' guesses "
         "and then again one token a call (the same prompt, read again from the conversation). The requests run one "
         "after the other after a warm-up, so the expert cache has followed what they use (its swaps): a cold engine's "
         "first answer is slower.\n\n";
    m += "| Prompt tokens | Prompt tok/s | Decode tok/s | Guesses right | Decode tok/s, one token a call |\n"
         "| ---: | ---: | ---: | ---: | ---: |\n";
    for (size_t i = 0; i < r.runs.size(); ++i) {
        const Run& run = r.runs[i];
        if (run.single) continue;
        const Run* one = i + 1 < r.runs.size() && r.runs[i + 1].single ? &r.runs[i + 1] : nullptr;
        m += "| " + std::to_string(run.read) + " | " + num(run.prompt_tok_s(), 0) + " | " + num(run.decode_tok_s(), 1) +
             " | " + (run.guessed > 0 ? std::to_string(run.guessed_right) + " of " + std::to_string(run.guessed) : "-") +
             " | " + (one ? num(one->decode_tok_s(), 1) : "-") + " |\n";
    }
    if (!r.long_points.empty() || !r.long_error.empty()) {
        m += "\n### Long context (one conversation, read in chunks; decode one token a call)\n\n";
        if (!r.long_error.empty()) m += "The long tier " + md_escape(r.long_error) + ".\n\n";
        m += "| Context | Chunk read | Chunk tok/s | Decode tok/s |\n| ---: | ---: | ---: | ---: |\n";
        for (const Run& run : r.long_points)
            m += "| " + std::to_string(run.context_end) + " | " + std::to_string(run.read) + " | " +
                 num(run.prompt_tok_s(), 0) + " | " + num(run.decode_tok_s(), 1) + " |\n";
    }
    m += "\n## Diagnosis\n\n";
    const auto diags = diagnose(r);
    if (diags.empty()) m += "Not enough split calls to diagnose.\n";
    for (const Diagnosis& g : diags) {
        m += "**" + md_escape(g.phase) + ": " + md_escape(g.verdict) + ".**\n\n";
        m += "- Roofline: " + fmt("%.2f", g.gpu_gb) + " GB " + g.unit + " (dense weights, KV rows, experts on the GPU) at the "
             "probed read bandwidth would take " + fmt("%.2f", g.ideal_ms) + " ms; it takes " + fmt("%.2f", g.total_ms) +
             " ms. Launches: " + fmt("%.0f", g.launches) + " " + g.unit + " (" + fmt("%.0f", g.direct) + " direct, " +
             fmt("%.0f", g.nodes) + " replayed), ~" + fmt("%.2f", g.launch_ms) + " ms at the probed floor of a small kernel.\n";
        if (g.hit_rate >= 0)
            m += "- Cache: " + pct(g.hit_rate) + " of the routed experts from VRAM (" + std::to_string(r.cached_experts) + " of " +
                 std::to_string(r.total_experts) + " held); " + fmt("%.3f", g.cpu_gb) + " GB " + g.unit +
                 " read by the CPU experts from RAM, " + fmt("%.3f", g.pcie_gb) + " GB over PCIe." +
                 (g.warm_cache ? " **Above 97%: this text keeps to the cached experts more than a real chat does (~87% "
                                 "on real text on an 8-16 GB card), so the misses' cost is understated here; "
                                 "`STRATA_PROFILE=1` on real chats, or `--text docs`, shows it.**"
                               : std::string()) +
                 "\n";
        if (g.round_trips > 0)
            m += "- Host round trips: " + fmt("%.0f", g.round_trips) + " " + g.unit + " (the host waits for a router's "
                 "scores, routes and hands the card its experts), the card idle ~" + fmt("%.2f", g.round_trip_ms) +
                 " ms meanwhile" + (r.probes.copy_sync_us > 0 ? " (the probed copy-and-wait alone: ~" +
             fmt("%.2f", g.round_trips * r.probes.copy_sync_us / 1000.0) + " ms)" : std::string()) + ".\n";
        if (!g.recover.empty()) {
            m += "- Most to recover:";
            for (size_t i = 0; i < g.recover.size(); ++i)
                m += std::string(i ? ";" : "") + " " + fmt("%.2f", g.recover[i].first) + " ms by " + md_escape(g.recover[i].second);
            m += " (estimates: launches at the probed latency, kernels at the probed read bandwidth).\n";
        }
        m += "\n| Family | ms | Launches | Launch ms (est.) | GB | GB/s | % of read | Verdict |\n"
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |\n";
        for (const FamilyVerdict& v : g.families)
            m += "| " + std::string(prof::fam_title(v.fam)) + " | " + num(v.ms, 2) + " | " + num(v.launches, 0) + " | " +
                 num(v.launch_ms, 2) + " | " + (v.gb > 0 ? fmt("%.3f", v.gb) : std::string("-")) + " | " +
                 (v.gb > 0 ? num(v.gbs, 0) : std::string("-")) + " | " + (v.gb > 0 ? pct(v.read_share) : std::string("-")) +
                 " | " + md_escape(v.verdict) + " |\n";
        m += "\n";
    }
    m += "Family rows are from the split calls (launched kernel by kernel); the verdict, roofline and launch lines from "
         "every call. Bytes are the engine's model: each dense matrix once a visit, the KV rows attention reads, an "
         "expert's weights each time a token routes to one on the card. Kernel time per launch is not measured, so "
         "\"tiny kernels\" is a family's average.\n";
    m += "\n## Explain\n\n";
    for (const Finding& f : explain(r)) m += std::string("- ") + (f.flag ? "**far off:** " : "") + f.phase + ": " + f.text + "\n";
    m += "\n<details><summary>Profile tables</summary>\n\nLaunches: the kernels the engine hands the card, " +
         std::string(prof::launches_wrapped() ? "its direct launches and the nodes of the recordings it replays"
                                              : "only the nodes of the recordings it replays (this build does not count "
                                                "direct launches)") +
         "; a BLAS call's own kernels are not counted.\n";
    for (const Run& run : r.runs) {
        profile_table(m, run, prof::kPrompt);
        profile_table(m, run, prof::kDecode);
    }
    for (const Run& run : r.long_points) profile_table(m, run, prof::kDecode);
    m += "\n</details>\n\nThe run took " + num(r.total_s, 0) + " s: probes " + num(r.probe_s, 0) + ", load " +
         num(r.load_s, 0) + ", warm-up " + num(r.warm_s, 0) + ", quick tier " + num(r.quick_s, 0) +
         (r.long_points.empty() ? std::string() : ", long tier " + num(r.long_s, 0)) + " s.\n";
    return scrub(m, private_strings());
}

}  // namespace strata::poly::bench
