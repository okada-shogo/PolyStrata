// src/poly/bench_report_test.cpp - --bench-report's report without a card: the profile's sums, the settings with
// their paths taken out, the explain section's flags, and a whole report scanned for this PC's and user's names.
#include "bench_report.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#ifdef _WIN32
#include <stdlib.h>
#endif

namespace B = strata::poly::bench;
namespace P = strata::poly::prof;

namespace {

int failed = 0;
#define CHECK(c)                                                                         \
    do {                                                                                 \
        if (!(c)) {                                                                      \
            std::fprintf(stderr, "bench_report_test:%d: CHECK(%s) failed\n", __LINE__, #c); \
            ++failed;                                                                    \
        }                                                                                \
    } while (0)

bool has(const std::string& s, const std::string& x) { return s.find(x) != std::string::npos; }

void set_env(const char* k, const char* v) {
#ifdef _WIN32
    _putenv_s(k, v);
#else
    setenv(k, v, 1);
#endif
}

// a call's figures: `ms` of family f, `tokens` tokens
P::Acc call(int f, double ms, int64_t tokens, double wall) {
    P::Acc a;
    a.fam_ms[f] = ms;
    a.tokens = tokens;
    a.calls = 1;
    a.wall_ms = wall;
    return a;
}

void test_profile() {
    P::Profile p;
    p.add_call(P::kPrompt, 0, true, call(P::kExpertsGpu, 100, 1024, 110));
    p.add_call(P::kPrompt, 1024, true, call(P::kWaitUpload, 50, 1024, 55));
    p.add_call(P::kDecode, 2048, true, call(P::kWaitCpu, 30, 1, 31));
    p.add_call(P::kDecode, 9000, false, call(P::kUnsplit, 40, 1, 41));
    CHECK(p.all[P::kPrompt].tokens == 2048);
    CHECK(p.all[P::kPrompt].timeline_ms() == 150);
    CHECK(p.split[P::kDecode].calls == 1 && p.all[P::kDecode].calls == 2);
    CHECK(p.by_ctx_all[P::kDecode].size() == 3);   // 0-2K, 2K-8K, 8K-32K
    CHECK(p.by_ctx_all[P::kDecode][1].calls == 1 && p.by_ctx_all[P::kDecode][2].calls == 1);
    CHECK(P::bucket_name(0) == "0K-2K" && P::bucket_name(2) == "8K-32K" && P::bucket_name(6) == "256K+");
    P::Acc la = call(P::kExpertsGpu, 10, 2, 11);
    la.fam_launches[P::kExpertsGpu] = 900;
    la.fam_launches[P::kNorms] = 700;
    p.add_call(P::kDecode, 100, true, la);
    CHECK(p.all[P::kDecode].launches() == 1600 && p.split[P::kDecode].fam_launches[P::kNorms] == 700);
    P::Json j;
    j.open();
    P::profile_json(j, "profile", p);
    j.close();
    CHECK(has(j.s, "\"experts_gpu\":100"));
    CHECK(has(j.s, "\"context\":\"8K-32K\""));
    CHECK(has(j.s, "\"launches_by_family\":{\"norms\":700,\"experts_gpu\":900}") && has(j.s, "\"launches_per_token\":"));
    CHECK(has(j.s, "\"share\":{\"experts_gpu\":0.6667,\"wait_upload\":0.3333}"));
    int depth = 0;
    for (char c : j.s) depth += c == '{' || c == '[' ? 1 : c == '}' || c == ']' ? -1 : 0;
    CHECK(depth == 0);
    p.reset();
    CHECK(p.all[P::kPrompt].calls == 0 && p.every == 16);
    // the JSON writer's strings and numbers
    P::Json q;
    q.open();
    q.str("s", "a\"b\\c\nd\x01");
    q.num("x", 2.50);
    q.num("y", -0.0001, 2);
    q.num("z", 1.0 / 0.0);
    q.close();
    CHECK(q.s == "{\"s\":\"a\\\"b\\\\c\\nd\\u0001\",\"x\":2.5,\"y\":0,\"z\":null}");
}

void test_settings() {
    const std::vector<std::string> argv{"--model",       "/home/someone/models/Inkling-2-Mini-IQ4_NL.gguf",
                                        "--expert-profile", "/home/someone/data/expert-profile.txt",
                                        "--max-context", "32768",
                                        "--kv",          "q8_0",
                                        "--tokens",      "1,2,3",
                                        "--draft-model", "D:\\models\\d.gguf",
                                        "--no-graphs",   "--cache-bias",
                                        "0.5",           "--pinned-session",
                                        "system.bin",    "--bench-report",
                                        "out/report"};
    const auto a = B::clean_args(argv);
    std::string all;
    for (const auto& [k, v] : a) all += k + "=" + v + " ";
    CHECK(!has(all, "--model") && !has(all, "someone") && !has(all, "Inkling") && !has(all, "models"));
    CHECK(has(all, "--expert-profile=<path>") && has(all, "--draft-model=<path>") && has(all, "--pinned-session=<path>"));
    CHECK(has(all, "--tokens=<ids>") && has(all, "--max-context=32768") && has(all, "--kv=q8_0"));
    CHECK(has(all, "--no-graphs= ") && has(all, "--cache-bias=0.5") && !has(all, "--bench-report"));
    std::vector<std::string> env_s{"STRATA_API_KEY=sk-secret", "STRATA_PROFILE_OUT=/home/someone/p.json",
                                   "STRATA_PROFILE=1", "PATH=/usr/bin", "STRATA_PF_I8MMA=0", "HOME=/home/someone"};
    std::vector<char*> envp;
    for (std::string& e : env_s) envp.push_back(e.data());
    envp.push_back(nullptr);
    const auto e = B::clean_env(envp.data());
    std::string ev;
    for (const auto& [k, v] : e) ev += k + "=" + v + " ";
    CHECK(has(ev, "STRATA_API_KEY=<hidden>") && !has(ev, "sk-secret"));
    CHECK(has(ev, "STRATA_PROFILE_OUT=<path>") && has(ev, "STRATA_PROFILE=1 ") && has(ev, "STRATA_PF_I8MMA=0"));
    CHECK(!has(ev, "PATH=") && !has(ev, "HOME") && !has(ev, "someone"));
    CHECK(B::looks_like_path("C:x") == false && B::looks_like_path("C:\\x") && B::looks_like_path("~/m") &&
          B::looks_like_path("m.gguf") && !B::looks_like_path("q8_0") && !B::looks_like_path("0.5"));
}

void test_scrub() {
    const std::string t = B::scrub("alice ran it in /home/alice/m on wonderbox; malice, alice_2 and Alice stay",
                                   {"/home/alice", "alice", "wonderbox", "ab"});
    CHECK(t == "<redacted> ran it in <redacted>/m on <redacted>; malice, alice_2 and Alice stay");
}

B::Report sample() {
    B::Report r;
    r.engine_version = "0.1.66";
    r.commit = "1a41ff673";
    r.model = {"Inkling 2 Mini", "inkling2_loop", "iq4_nl/iq4_nl", 17.6, 64, 8, 24, 3.375};
    r.gpu.name = "NVIDIA GeForce RTX 2070 SUPER";
    r.gpu.backend = "cuda";
    r.gpu.arch = "sm_75";
    r.gpu.sms = 40;
    r.gpu.vram_gib = 8;
    r.gpu.pcie_gen = 3;
    r.gpu.pcie_width = 16;
    r.host.cpu = "AMD Ryzen 5 3600XT 6-Core Processor";
    r.host.threads = 12;
    r.host.engine_threads = 8;
    r.host.ram_gib = 31.3;
    r.host.isa = "avx2";
    r.probes.h2d_pinned_gbs = 12.2;
    r.probes.h2d_pageable_gbs = 6.1;
    r.probes.ram_read_gbs = 38;
    r.probes.dots.push_back({"iq4_nl", "avx2", "avx2", 30, 6, 8});
    r.settings = {{"--max-context", "32768"}, {"--expert-profile", "<path>"}};
    B::Run pr;
    pr.name = "prompt 8192";
    pr.read = pr.prompt_tokens = 8192;
    pr.prompt_ms = 10000;
    pr.produced = 129;
    pr.decode_ms = 6400;
    P::Acc a;
    a.fam_ms[P::kWaitUpload] = 5000;   // half the time, 23 GB in 5 s = 4.6 GB/s against 12.2 probed
    a.fam_ms[P::kExpertsGpu] = 4000;
    a.fam_ms[P::kHostGap] = 1000;
    a.upload_bytes = 23e9;
    a.tokens = 8192;
    a.calls = 8;
    a.wall_ms = 10000;
    pr.profile.add_call(P::kPrompt, 0, true, a);
    P::Acc d;
    d.fam_ms[P::kWaitCpu] = 25;
    d.fam_ms[P::kExpertsGpu] = 20;
    d.fam_ms[P::kHostGap] = 5;
    d.tokens = 1;
    d.calls = 1;
    d.wall_ms = 50;
    d.cpu_ms = 24;
    d.cpu_experts = 200;   // 675 MiB at 24 ms: 29 GB/s
    for (int i = 0; i < 128; ++i) pr.profile.add_call(P::kDecode, 8192 + i, i % 16 == 0, d);
    r.runs.push_back(pr);
    r.total_s = 100;
    return r;
}

void test_explain() {
    B::Report r = sample();
    auto f = B::explain(r);
    bool upload_flag = false, decode_line = false, cpu_line = false;
    for (const B::Finding& x : f) {
        if (has(x.text, "waits for expert uploads")) upload_flag = x.flag && has(x.text, "12.2 GB/s pinned");
        if (x.phase == "decode" && has(x.text, "tok/s over")) decode_line = has(x.text, "50.0 ms a token");
        if (has(x.text, "CPU experts 24.0 ms a token")) cpu_line = !x.flag;   // near the probes: no flag
    }
    if (!(upload_flag && decode_line && cpu_line))
        for (const B::Finding& x : f) std::fprintf(stderr, "  [%d] %s: %s\n", (int) x.flag, x.phase.c_str(), x.text.c_str());
    CHECK(upload_flag && decode_line && cpu_line);
    r.probes.h2d_pinned_gbs = 6.0;   // 4.6 of 6.0: not far off
    for (const B::Finding& x : B::explain(r))
        if (has(x.text, "waits for expert uploads")) CHECK(!x.flag);
}

void test_report_privacy() {
    // names this "PC" has for the test; the report is scrubbed of whatever private_strings() finds
    set_env("USER", "zq_tester_9");
    set_env("HOME", "/home/zq_tester_9");
    set_env("HOSTNAME", "zq-host-box");
    B::Report r = sample();
    r.model.name = "zq_tester_9's Inkling";   // a model named after its user
    r.host.cpu = "CPU of zq-host-box";
    r.settings.push_back({"--note", "/home/zq_tester_9/x"});
    const std::string json = B::to_json(r), md = B::to_markdown(r);
    for (const std::string* s : {&json, &md}) {
        CHECK(!has(*s, "zq_tester_9") && !has(*s, "zq-host-box") && !has(*s, "/home/"));
        CHECK(has(*s, "<redacted>"));
        for (const std::string& x : B::private_strings()) CHECK(!has(*s, x));
    }
    CHECK(has(json, "\"kind\":\"strata-bench-report\"") && has(json, "\"explain\":[") && has(json, "\"probes\":{"));
    CHECK(has(md, "## Explain") && has(md, "| 8192 | 819 | ") && has(md, "**far off:**"));
    int depth = 0;
    bool in_str = false, esc = false;
    for (char c : json) {
        if (in_str) {
            if (esc) esc = false;
            else if (c == '\\') esc = true;
            else if (c == '"') in_str = false;
            continue;
        }
        if (c == '"') in_str = true;
        else if (c == '{' || c == '[') ++depth;
        else if (c == '}' || c == ']') --depth;
        CHECK(depth >= 0);
    }
    CHECK(depth == 0 && !in_str);
    const std::string sess = B::session_json(r, r.runs[0].profile);
    CHECK(has(sess, "\"kind\":\"strata-session-profile\"") && !has(sess, "zq_tester_9"));
    if (const char* dump = std::getenv("BENCH_REPORT_TEST_DUMP")) {   // for tools/test_bench_reports.py
        if (FILE* f = std::fopen(dump, "wb")) {
            std::fwrite(json.data(), 1, json.size(), f);
            std::fclose(f);
        }
    }
}

// A user whose name is a key of the JSON keeps the JSON whole: values are scrubbed, keys are not.  And a home folder
// with backslashes (Windows) is found before JSON escapes them.
void test_scrub_values_only() {
    set_env("USER", "share");
    set_env("HOME", "C:\\Users\\zq_win_home");
    B::Report r = sample();
    r.model.name = "share's model in C:\\Users\\zq_win_home\\m";
    const std::string json = B::to_json(r);
    CHECK(has(json, "\"share\":{"));   // the profile's shares stay
    CHECK(!has(json, "zq_win_home") && has(json, "<redacted>'s model"));
    set_env("USER", "zq_tester_9");
    set_env("HOME", "/home/zq_tester_9");
}

// A synthetic launch-bound card (not a measured one): ~1,580 kernels a 25 ms token at an 8.5 us floor each (~13 ms),
// experts a quarter of the time: the report must say launch-bound on its own.
void test_diagnosis_launch_bound() {
    B::Report r = sample();
    r.runs.clear();
    r.probes.launch_us = 8.5;
    r.probes.graph_node_us = 8.2;
    r.probes.small_us = 8.5, r.probes.small_node_us = 8.2;
    r.probes.read_gbs = 560;
    r.launches_wrapped = true;
    B::Run run;
    run.name = "prompt 2048, one token a call";
    run.single = true;
    run.produced = 129;
    run.decode_ms = 25.0 * 128;
    P::Acc unsplit;   // a replayed token: 25 ms, 1,580 kernels of recordings
    unsplit.fam_ms[P::kUnsplit] = 25;
    unsplit.fam_launches[P::kUnsplit] = 1580;
    unsplit.fam_graph_nodes[P::kUnsplit] = 1580;
    unsplit.fam_bytes[P::kDense] = 0.9e9;
    unsplit.fam_bytes[P::kExpertsGpu] = 0.5e9;
    unsplit.tokens = 1, unsplit.calls = 1, unsplit.wall_ms = 25;
    unsplit.gpu_experts = 300, unsplit.cpu_experts = 84;
    P::Acc split;     // a split token, launched kernel by kernel: the families
    split.fam_ms[P::kExpertsGpu] = 7;
    split.fam_launches[P::kExpertsGpu] = 400;
    split.fam_bytes[P::kExpertsGpu] = 0.5e9;
    split.fam_ms[P::kDense] = 6;
    split.fam_launches[P::kDense] = 300;
    split.fam_bytes[P::kDense] = 0.9e9;
    split.fam_ms[P::kNorms] = 8;
    split.fam_launches[P::kNorms] = 700;
    split.fam_ms[P::kAttention] = 3;
    split.fam_launches[P::kAttention] = 180;
    split.fam_ms[P::kWaitCpu] = 1;
    split.tokens = 1, split.calls = 1, split.wall_ms = 25;
    for (int i = 0; i < 128; ++i) run.profile.add_call(P::kDecode, 2048 + i, i % 16 == 0, i % 16 == 0 ? split : unsplit);
    r.runs.push_back(run);
    const auto d = B::diagnose(r);
    CHECK(!d.empty() && d[0].verdict.rfind("launch-bound", 0) == 0);
    if (!d.empty()) {
        CHECK(d[0].launch_ms > 10 && d[0].launch_ms < 16);
        CHECK(!d[0].recover.empty() && has(d[0].recover[0].second, "fewer launches"));
        bool norms_launch = false;
        for (const auto& v : d[0].families)
            if (v.fam == P::kNorms) norms_launch = has(v.verdict, "launch-bound") && v.tiny;
        CHECK(norms_launch);
        std::fprintf(stderr, "  diagnosis: %s\n", d[0].verdict.c_str());
    }
    const std::string md = B::to_markdown(r);
    CHECK(has(md, "## Diagnosis") && has(md, "launch-bound"));
}

// The RX 9070 XT measured under WSL2 (IQ4_NL, reserve 700): a small kernel's floor 3.1 us (2.2 us across the card),
// 1,491 kernels a 24.1 ms token (~4.6 ms), 48 host round trips with the card idle ~5 ms, 4.3 GB at 594 GB/s (7.3 ms):
// not launch-bound; the round trips come first among what to recover.
void test_diagnosis_rx9070xt_measured() {
    B::Report r = sample();
    r.runs.clear();
    r.probes.launch_us = 1.5, r.probes.graph_node_us = 1.4;
    r.probes.small_us = 3.1, r.probes.small_node_us = 3.1, r.probes.wide_node_us = 2.2;
    r.probes.read_gbs = 594;
    r.launches_wrapped = true;
    B::Run run;
    run.name = "prompt 2048, one token a call";
    run.single = true;
    P::Acc unsplit;
    unsplit.fam_ms[P::kUnsplit] = 24.1;
    unsplit.fam_launches[P::kUnsplit] = 1491;
    unsplit.fam_graph_nodes[P::kUnsplit] = 1246;
    unsplit.fam_bytes[P::kDense] = 2.217e9;
    unsplit.fam_bytes[P::kRouting] = 0.028e9;
    unsplit.fam_bytes[P::kAttention] = 0.132e9;
    unsplit.fam_bytes[P::kExpertsGpu] = 1.95e9;
    unsplit.tokens = 1, unsplit.calls = 1, unsplit.wall_ms = 24.1, unsplit.route_waits = 48;
    unsplit.gpu_experts = 381, unsplit.cpu_experts = 3;
    P::Acc split;
    const struct { int f; double ms, n, gb; } fams[] = {
        {P::kAttention, 2.8, 192, 0.132}, {P::kKvStore, 1.8, 384, 0}, {P::kDense, 5.9, 231, 2.217},
        {P::kNorms, 4.2, 343, 0},         {P::kRouting, 5.2, 96, 0.028}, {P::kExpertsGpu, 6.3, 240, 1.95},
        {P::kCopyH2D, 0.2, 5, 0},         {P::kWaitCpu, 0.3, 0, 0}};
    for (const auto& x : fams) split.fam_ms[x.f] = x.ms, split.fam_launches[x.f] = x.n, split.fam_bytes[x.f] = x.gb * 1e9;
    split.tokens = 1, split.calls = 1, split.wall_ms = 27, split.route_waits = 48;
    for (int i = 0; i < 128; ++i) run.profile.add_call(P::kDecode, 2048 + i, i % 16 == 0, i % 16 == 0 ? split : unsplit);
    r.runs.push_back(run);
    const auto d = B::diagnose(r);
    CHECK(!d.empty() && d[0].verdict.rfind("launch-bound", 0) != 0);
    if (!d.empty()) {
        CHECK(d[0].launch_ms > 4 && d[0].launch_ms < 5);
        CHECK(d[0].round_trips == 48 && d[0].round_trip_ms > 4);
        CHECK(!d[0].recover.empty() && has(d[0].recover[0].second, "host round trips"));
        std::fprintf(stderr, "  diagnosis: %s\n", d[0].verdict.c_str());
    }
}

// A card with everything in VRAM, replayed nodes at 1.5 us, a token of 2.2 GB at 300 GB/s probed: it must say
// bandwidth-bound, not launch-bound.  And kernels twice a launch long are not launch-bound (the launches hide).
void test_diagnosis_bandwidth_and_hidden_launches() {
    B::Report r = sample();
    r.runs.clear();
    r.probes.launch_us = 4.0;
    r.probes.graph_node_us = 1.5;
    r.probes.read_gbs = 300;   // 2.2 GB a token at 300 GB/s: 7.3 of 11 ms
    r.launches_wrapped = true;
    B::Run run;
    run.name = "prompt 2048, one token a call";
    run.single = true;
    P::Acc unsplit;
    unsplit.fam_ms[P::kUnsplit] = 11;
    unsplit.fam_launches[P::kUnsplit] = 900;
    unsplit.fam_graph_nodes[P::kUnsplit] = 900;
    unsplit.fam_bytes[P::kDense] = 1.2e9;
    unsplit.fam_bytes[P::kExpertsGpu] = 1.0e9;
    unsplit.tokens = 1, unsplit.calls = 1, unsplit.wall_ms = 11;
    unsplit.gpu_experts = 384;
    P::Acc split;
    split.fam_ms[P::kExpertsGpu] = 6;     // 200 launches of 30 us: the kernels hide the launches
    split.fam_launches[P::kExpertsGpu] = 200;
    split.fam_bytes[P::kExpertsGpu] = 1.0e9;
    split.fam_ms[P::kDense] = 5;
    split.fam_launches[P::kDense] = 300;  // 16.7 us a launch against 4 us: not launch-bound
    split.fam_bytes[P::kDense] = 1.2e9;
    split.tokens = 1, split.calls = 1, split.wall_ms = 13;
    for (int i = 0; i < 128; ++i) run.profile.add_call(P::kDecode, 2048 + i, i % 16 == 0, i % 16 == 0 ? split : unsplit);
    r.runs.push_back(run);
    const auto d = B::diagnose(r);
    CHECK(!d.empty() && d[0].verdict.rfind("bandwidth-bound", 0) == 0);
    for (const auto& v : d.empty() ? std::vector<B::FamilyVerdict>{} : d[0].families)
        CHECK(v.verdict.find("launch-bound") == std::string::npos);
    if (!d.empty()) std::fprintf(stderr, "  diagnosis: %s\n", d[0].verdict.c_str());
}

// A prompt: experts counted once a visit (in VRAM, copied in), the KV rows a tile of queries at a time
void test_diagnosis_prompt() {
    B::Report r = sample();
    r.probes.launch_us = 4.0;
    r.probes.graph_node_us = 1.5;
    r.probes.read_gbs = 400;
    r.launches_wrapped = true;
    P::Acc a;
    a.fam_ms[P::kExpertsGpu] = 2000;
    a.fam_launches[P::kExpertsGpu] = 20000;
    a.fam_ms[P::kWaitUpload] = 300;
    a.fam_bytes[P::kExpertsGpu] = 3.0e9;
    a.fam_bytes[P::kDense] = 0.9e9;
    a.fam_ms[P::kDense] = 300;
    a.fam_launches[P::kDense] = 2000;
    a.gpu_experts = 400, a.staged_experts = 2000, a.cpu_experts = 0;
    a.tokens = 2048, a.calls = 2, a.wall_ms = 2600;
    r.runs[0].profile = P::Profile{};
    r.runs[0].read = 2048;
    r.runs[0].profile.add_call(P::kPrompt, 0, true, a);
    const auto d = B::diagnose(r);
    bool found = false;
    for (const auto& g : d)
        if (g.phase.rfind("prompt", 0) == 0) {
            found = true;
            CHECK(g.hit_rate > 0.16 && g.hit_rate < 0.17);   // 400 of 2,400 a visit from VRAM
            CHECK(g.verdict.rfind("launch-bound", 0) != 0);
            std::fprintf(stderr, "  prompt diagnosis: %s\n", g.verdict.c_str());
        }
    CHECK(found);
}

}  // namespace

int main() {
    test_profile();
    test_settings();
    test_scrub();
    test_explain();
    test_report_privacy();
    test_scrub_values_only();
    test_diagnosis_launch_bound();
    test_diagnosis_rx9070xt_measured();
    test_diagnosis_bandwidth_and_hidden_launches();
    test_diagnosis_prompt();
    if (failed) {
        std::fprintf(stderr, "bench_report_test: %d checks failed\n", failed);
        return 1;
    }
    std::printf("bench_report_test: ok\n");
    return 0;
}
