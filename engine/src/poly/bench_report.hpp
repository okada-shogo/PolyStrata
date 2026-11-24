// src/poly/bench_report.hpp - strata-poly --bench-report: what the report holds, the hardware probes, and the JSON and
// Markdown it is written as.
//
// The report is meant to be posted: it holds the card, the CPU, the engine's version, the settings, the speeds, the
// event profile (profile.hpp) and the probes, and nothing that names the person or the PC.  No path (a setting's value
// that looks like one is written as <path>), no user or host name, no serial number or bus address, no prompt (the
// prompts are random tokens from a fixed seed).  scrub() takes out of the finished text every string of
// private_strings() as a last line of defence, and bench_report_test.cpp checks a report for them.
//
// The probes (bench_probe.cpp) are measured before the model loads, so the card's memory is free for them.
#pragma once

#include "profile.hpp"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace strata::poly::bench {

struct Gpu {
    std::string name, arch, driver, runtime, backend;   // backend: "cuda" or "hip"
    int sms = 0;                                         // SMs (NVIDIA) or CUs (AMD)
    double vram_gib = 0;
    double mem_peak_gbs = 0;                             // from the memory clock and bus width, 0 if unknown
    int pcie_gen = -1, pcie_width = -1, pcie_gen_max = -1, pcie_width_max = -1;   // -1: not known
    std::string pcie_source;   // "nvml", "setupapi", "sysfs" or "" (not known)
};

struct Host {
    std::string cpu, os, isa;   // isa: the CPU experts' instruction set (strata::quant::best_isa())
    std::vector<std::string> flags;
    int threads = 0;            // the CPUs this process may use
    int engine_threads = 0;     // the engine's CPU expert threads
    double ram_gib = 0;
};

struct DotRate {
    std::string type, kernel, isa;
    double gbs_all = 0, gbs_one = 0;   // GB/s of weights, all the engine's threads and one
    int threads = 0;
};

struct Probes {
    double h2d_pinned_gbs = 0, h2d_pageable_gbs = 0, d2h_pinned_gbs = 0, gpu_mem_gbs = 0, ram_read_gbs = 0;
    int ram_threads = 0;
    std::vector<DotRate> dots;
    double seconds = 0;
};

struct Model {
    std::string name, arch, file_type;
    double size_gib = 0;
    int n_expert = 0, n_used = 0, n_moe = 0;
    double expert_mib = 0;   // one routed expert's gate, up and down
};

/// One request of the run: a prompt read (some of it may be the conversation's already) and an answer.
struct Run {
    std::string name;
    bool single = false;   // its answer one token a call, no guesses
    int64_t prompt_tokens = 0, reused = 0, read = 0, produced = 0, context_end = 0;
    double prompt_ms = 0, decode_ms = 0;
    int64_t guessed = 0, guessed_right = 0;
    prof::Profile profile;

    double prompt_tok_s() const { return prompt_ms > 0 ? 1000.0 * (double) read / prompt_ms : 0.0; }
    /// The first token is the prompt's (as Session::Result counts it): the rate of the ones after it.
    double decode_tok_s() const { return decode_ms > 0 && produced > 1 ? 1000.0 * (double) (produced - 1) / decode_ms : 0.0; }
};

struct Finding {
    std::string phase, text;
    bool flag = false;   // far from what the probes say the hardware does
};

struct Report {
    std::string engine_version = "dev", commit = "unknown", tier = "quick";
    std::string prompt_source = "random tokens of the vocabulary";
    Model model;
    std::vector<std::pair<std::string, std::string>> settings;   // the engine's arguments, paths taken out
    std::vector<std::pair<std::string, std::string>> env;        // STRATA_* settings, paths taken out
    Gpu gpu;
    Host host;
    Probes probes;
    double load_s = 0, total_s = 0, probe_s = 0, warm_s = 0, quick_s = 0, long_s = 0;   // seconds
    int cached_experts = 0, total_experts = 0, max_context = 0;
    double cached_gib = 0;
    std::vector<Run> runs;         // the quick tier's requests
    std::vector<Run> long_points;  // --long: the checkpoints of the one long conversation
    std::string long_error;        // why the long tier stopped early ("" when it did not)
};

/// The engine's arguments as (option, value) pairs: a value that looks like a path or a file is "<path>", and
/// --model's is left out (Report::model says what the file is).
std::vector<std::pair<std::string, std::string>> clean_args(const std::vector<std::string>& argv);
/// The STRATA_* variables set in this process, with values that look like paths as "<path>"; the API key's never.
std::vector<std::pair<std::string, std::string>> clean_env(char** envp);
/// True for a value that names a file or a folder.
bool looks_like_path(const std::string& v);

/// This user's and this PC's names and folders as they could show up in a text: user and host names, the home
/// and working folders.  Strings under four characters are left out (they would match everywhere).
std::vector<std::string> private_strings();
/// `text` with every one of `secrets` replaced by <redacted>.
std::string scrub(std::string text, const std::vector<std::string>& secrets);

/// The explain section: each phase against what the probes measured.
std::vector<Finding> explain(const Report& r);

/// The commit the engine was built from (git describe at configure time), or "unknown".
const char* build_commit();

std::string to_json(const Report& r);
std::string to_markdown(const Report& r);

/// A request's profile written alone (STRATA_PROFILE=1 in normal use): hardware without probes, and the profile.
std::string session_json(const Report& r, const prof::Profile& p);

// ---- bench_probe.cpp (needs the card) ----
Gpu gpu_info(int device);
Host host_info();
/// The copies and memory reads, ~10 s; before the model is loaded.  `log` gets a line a probe (may be null).
void probe_hardware(Probes& p, int device, void (*log)(const char*));
/// The CPU experts' dot product for each (ggml type id, row length) on `threads` threads and one, ~2 s a type.
void probe_dots(Probes& p, const std::vector<std::pair<int, int>>& type_cols, int threads);

}  // namespace strata::poly::bench
