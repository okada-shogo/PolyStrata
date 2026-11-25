// src/poly/dspark_layout.hpp - which DSpark drafter layout a file has, from its names alone.  Unsloth's
// Inkling-2-Mini-GGUF repository has two: its own (dflash.dspark.markov_rank, dspark_hidden_out), the one
// Inkling2::ds_open() reads, and since 2026-11-23 at the repository's root the layout of llama.cpp's DSpark pull
// request #32218 (markov_w2 over the whole vocabulary, ctx_mix a layer, no markov_rank).  The same name is now the
// second one, so a file of it is refused with the name of the file to fetch instead.
#pragma once

#include <functional>
#include <string>

namespace strata::poly {

// "" unless the file is a drafter of llama.cpp #32218's layout; then what to use instead.
inline std::string dspark_layout_problem(const std::function<bool(const char*)>& has_key,
                                         const std::function<bool(const char*)>& has_tensor) {
    if (has_key("dflash.dspark.markov_rank")) return "";
    if (!has_tensor("markov_w2.weight") && !has_key("dflash.has_markov_context")) return "";
    return "this drafter is in the layout of llama.cpp's DSpark pull request #32218 (markov_w2, no "
           "dflash.dspark.markov_rank), which this engine does not read; take Unsloth's own layout, "
           "dspark-Inkling-2-Mini-Q8_0_Old.gguf (402,278,560 bytes) from unsloth/Inkling-2-Mini-GGUF";
}

}  // namespace strata::poly
