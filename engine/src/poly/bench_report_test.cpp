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

}  // namespace

int main() {
    test_profile();
    test_settings();
    test_scrub();
    test_explain();
    test_report_privacy();
    test_scrub_values_only();
    if (failed) {
        std::fprintf(stderr, "bench_report_test: %d checks failed\n", failed);
        return 1;
    }
    std::printf("bench_report_test: ok\n");
    return 0;
}
