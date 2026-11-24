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
    m += "| RAM read, " + std::to_string(r.probes.ram_threads) + " threads | " + num(r.probes.ram_read_gbs, 1) + " |\n";
    for (const DotRate& d : r.probes.dots)
        m += "| CPU expert dot " + d.type + " (" + d.kernel + ", " + d.isa + "), " + std::to_string(d.threads) +
             " threads / 1 | " + num(d.gbs_all, 1) + " / " + num(d.gbs_one, 1) + " |\n";
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
