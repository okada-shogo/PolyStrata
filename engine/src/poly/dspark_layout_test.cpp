// src/poly/dspark_layout_test.cpp - dspark_layout_problem() on the names of the two drafters Unsloth publishes for
// Inkling 2 Mini: dspark-Inkling-2-Mini-Q8_0_Old.gguf (Unsloth's layout, the one the engine reads) and the
// repository's root dspark-Inkling-2-Mini-Q8_0.gguf since 2026-11-23 (llama.cpp #32218's layout).
//
//   g++ -std=c++20 src/poly/dspark_layout_test.cpp -o dspark_layout_test && ./dspark_layout_test
#include "dspark_layout.hpp"

#include <cstdio>
#include <cstdlib>
#include <set>
#include <string>

namespace {

int checks = 0;
void check(bool ok, const char* what) {
    ++checks;
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        std::exit(1);
    }
}

std::string problem(const std::set<std::string>& keys, const std::set<std::string>& tensors) {
    return strata::poly::dspark_layout_problem([&](const char* k) { return keys.count(k) > 0; },
                                               [&](const char* t) { return tensors.count(t) > 0; });
}

}  // namespace

int main() {
    // the names the two files differ in (their headers, read 2026-11-25); the rest they share
    const std::set<std::string> shared_keys = {"general.architecture", "dflash.block_size", "dflash.block_count",
                                               "dflash.attention.sliding_window", "dflash.target_layers"};
    const std::set<std::string> shared_tensors = {"markov_w1.weight", "joint_proj.weight", "mask_embd.weight",
                                                  "blk.0.attn_q.weight", "output_norm.weight"};
    auto unsloth_keys = shared_keys, unsloth_tensors = shared_tensors;
    unsloth_keys.insert({"dflash.dspark.markov_rank", "dflash.vocab_size", "dflash.vocab_size_unpadded",
                         "dflash.attention.causal"});
    unsloth_tensors.insert({"dspark_hidden_out.weight", "dspark_hidden_out.bias", "dspark_fuse.weight",
                            "dspark_norm.weight", "blk.0.dflare_norm.weight"});
    auto mainline_keys = shared_keys, mainline_tensors = shared_tensors;
    mainline_keys.insert({"dflash.has_markov_context", "tokenizer.ggml.padding_token_id"});
    mainline_tensors.insert({"markov_w2.weight", "markov_w2.bias", "markov_ctx_mix.weight", "markov_ctx_norm.weight",
                             "token_embd_norm.weight", "blk.0.ctx_mix.weight", "blk.0.ctx_norm.weight"});

    check(problem(unsloth_keys, unsloth_tensors).empty(), "Unsloth's layout is read");
    const std::string m = problem(mainline_keys, mainline_tensors);
    check(m.find("#32218") != std::string::npos, "the mainline layout is named");
    check(m.find("dspark-Inkling-2-Mini-Q8_0_Old.gguf") != std::string::npos, "the file to take is named");
    // either sign of the mainline layout is enough; a file with neither is left to ds_open()'s own checks
    auto only_w2 = shared_tensors;
    only_w2.insert("markov_w2.weight");
    check(!problem(shared_keys, only_w2).empty(), "markov_w2 alone");
    auto only_ctx = shared_keys;
    only_ctx.insert("dflash.has_markov_context");
    check(!problem(only_ctx, shared_tensors).empty(), "has_markov_context alone");
    check(problem(shared_keys, shared_tensors).empty(), "neither: ds_open() names the missing key");
    std::printf("dspark_layout_test: %d checks passed\n", checks);
    return 0;
}
