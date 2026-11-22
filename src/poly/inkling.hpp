// src/poly/inkling.hpp - Thinking Machines' Inkling 2 Mini (llama.cpp architecture "inkling2_loop"): 128 experts,
// 8 a token, 24 layers of which the middle twelve run three times.
//
// No rotary embedding: a layer's attention adds a bias by the distance between query and key, a learned table of
// 16 per distance that the query's own 16 coefficients a head weigh, and each query has a sink logit of its own.
// Five layers in six attend a window of 1,024 positions; every sixth attends everything, with its queries scaled up
// slowly past 8,192 positions.  Keys, values, the attention's output and the experts' sum each go through a short
// causal convolution of four taps.
//
// The residual stream starts afresh every six layers ("attention residuals"): what a block of six started from is
// kept, and the input of an attention, an FFN or the output is a learned mix of those and of the stream.
//
// Layers 6 to 17 are the core, run three times in a row ("passes").  Each pass starts from what layer 5 gave; from
// the second on, every core layer also gets a message from the pass before (a mix of what that pass left), which
// a small gate of its own blends into the attention's and the FFN's inputs.  Every run of a layer has its own
// attention cache and convolution state.
//
// In the engine a run of a layer is a "visit": 48 of them in the order they run (layers_, n_layer_), each with the
// weights of its physical layer.  On several cards a card has a run of physical layers, and the loop's are on one
// card: a card starts at layer 1 to 6 or 18 to 22.  The reference for the math is llama.cpp's
// src/models/inkling2.cpp and inkling2-loop.cpp (29b52d398b).
#pragma once

#include "engine.hpp"
#include "strata/poly/inkling_kernels.hpp"

#include <algorithm>

namespace strata::poly {

class Inkling final : public Engine {
public:
    /// For a layer split (Engine::open()): the file's layers (n_layer: its block_count, which in Unsloth's files
    /// counts every visit) as physical ones, and the loop's [keep_lo, keep_hi), which runs on one card: a later visit
    /// runs the first one's experts (Layer::alias) and weights.
    static void split_layers(const Shards& file, int& n_layer, int& keep_lo, int& keep_hi);
    size_t card_row_bytes() const override;
    void save_card_rows(int64_t first, int64_t n, uint8_t* dst) override;
    void load_card_rows(const uint8_t* src, int64_t from, int64_t count, int64_t first, int64_t n) override;

protected:
    void load_geometry() override;
    void load_layer(int il) override;
    void begin_pass() override;
    size_t recurrent_floats() const override;
    void alloc_caches() override;
    void assign_scratch() override;
    bool sees() const override { return true; }   // pictures (strata-vision's rows) in place of tokens
    /// step()'s rows in one head: the products, mixes and controllers for all of them, and each row's caches,
    /// convolution states and position from its own sequence (attn(), post_conv()).  The rows' positions go to the
    /// kernels by value, so a step is not recorded: with --batch Inkling sets no Layer::fixed, and step() records
    /// only those.
    bool batch_heads() const override { return true; }
    /// A call's launches that take the position read its first one from at_ when recorded (attn()), a token's and a
    /// window's alike; a full visit's attention is recorded once for each room() it is sized for, a window visit's
    /// for the context.
    int head_key(int il, int n) const override;
    /// A call's visits are recorded only as one, when its routed experts are all on the card (Engine::run()).
    bool layer_graphs() const override { return false; }
    /// A thread a core: on a Ryzen 7 7800X3D (8 cores, 16 CPUs) beside an RX 9070 XT, IQ4_NL wrote 45.5 to 47.2
    /// tokens/s with 8 threads and 42.1 to 44.7 with the 11 of 70%; on a Ryzen 5 3600XT beside an RTX 2070 Super,
    /// 6, 8, 11 and 12 threads were the same.
    bool thread_a_core() const override { return true; }
    /// A window of guesses is routed on the card (Engine::moe_card()) with every expert in VRAM, so that it is
    /// recorded as one too: --window 5 to 8 have more rows than kSumRows.
    bool windows_on_card() const override { return true; }
    int widest() const override { return std::max({n_head_ * head_dim_, 2 * n_embd_, s_ff_}); }
    int streams() const override { return kStreams + (spark_ ? kTaps : 0); }
    void layer_head(Layer& L, int il, int n) override;
    void layer_tail(Layer& L, int il, int n) override;
    void output_rows(int first, int rows, float* h) override;
    void route(int il, const float* logits, int32_t token, int* ids, float* w, float* tmp) const override;
    bool route_card(int il, int n, int32_t* ids, float* w) override;
    void router_scores(const Layer& L, int n) override;
    void moe_shared(const Layer& L, int il, int n) override;
    /// A token's (fused()), outside a window of guesses: the same for every call of one token, so that a recording
    /// of moe_card() serves run() and step() alike.
    bool card_sum_in_tail(int n) const override { return n == 1 && !opt_.dump && opt_.window == 1; }
    bool act_rows_q8_1(const float* gate_up, void* q8, int rows, const ExpertScales* es) override;
    void finish_logits(float* logits, int rows) override;
    void shift_rows(ptrdiff_t by) override;
    size_t extra_bytes() const override;
    void load_drafter() override;
    void drafter_rows(int64_t pos, int n, const float* h) override;
    int draft_block(int32_t token, int most, Likeliest* out) override;
    void save_layer(int il, int64_t pos, uint8_t* dst) override;
    void load_extra(const uint8_t* src, int64_t pos) override;
    void save_last(uint8_t* dst, int64_t pos) override;
    const Names* tensor_alias(const char* name) const override;
    const Names* meta_alias(const char* key) const override;
    int physical(int v) const override { return phys_of(v); }

private:
    /// The residual streams of a token in r_, n_embd_ floats each: the stream itself, the four blocks' starts
    /// (the normed embedding, layer 5's output, layer 11's of this pass, layer 17's of the last pass), and what
    /// the pass before left for the messages (its layer 11's and 17's outputs).
    /// With the drafter, three more: the streams entering the visits it reads (kTap0 on).
    enum Stream { kCur = 0, kBlock0 = 1, kBank = 5, kBankOut = 6, kStreams = 7, kTap0 = 7, kTaps = 3 };

    /// A visit's weights (its physical layer's: a later visit has the first one's pointers) and its caches.
    struct Block {
        int phys = 0;     ///< the physical layer
        int pass = 0;     ///< 0: the prelude, the epilogue and the core's first pass; 1, 2: the core's later passes
        bool swa = false;
        int extent = 0;   ///< the distances its bias table has
        Dev wq, wk, wv, wo, wgate, wrel, wsink, rel_bias, q_clip, rel_clip;
        Dev k_conv, v_conv, post_conv, ffn_post_conv;
        Dev q_norm, k_norm, v_norm;       ///< a full layer's three input norms (a window layer's is Layer::attn_norm)
        Dev k_conv_proj, v_conv_proj;     ///< a full layer's input-dependent taps
        Dev attn_query, ffn_query;        ///< the depth mix's queries: 1 (window) or 3 (full) for the attention
        Dev msg_query;                    ///< a core layer's: the messages' 4 queries
        Dev ctrl_down[4], ctrl_gain[4], ctrl_forget[4];   ///< a core layer's gates: q, k, v, ffn
        std::vector<float> probs_b;       ///< the bias the routed experts' pick adds to the gates
        Dev probs_b_dev;
        float* conv = nullptr;            ///< in state_: K, V, attention-out, FFN-out convolutions' 3 past inputs
        /// The caches, a row a position (max_context of them, or ring_ in a window visit): fp16 keys and values
        /// (planes 0, 1), or under --kv q8_0 the keys' codes and scales and the values' codes and scales (0 to 3), a
        /// Q8_0 block's fp16 scale for every 32 codes (store_q8()).
        uint8_t* plane[4] = {};
    };

    bool q8() const { return opt_.kv_format == KvFmt::kQ8_0; }
    int planes() const { return q8() ? 4 : 2; }
    size_t plane_row(int i) const;   ///< a row's bytes in plane i
    int phys_of(int v) const;
    int pass_of(int v) const;
    /// The blocks a layer's mix has before its attention (before) or its FFN (after its block is pushed).
    static int blocks_before(int ip) { return ip == 0 ? 0 : (ip - 1) / kBlockSize + 1; }
    static int blocks_at(int ip) { return ip / kBlockSize + 1; }
    static constexpr int kBlockSize = 6;
    /// loop.count's bound: the released model runs 3 passes.  Nothing else limits it (each pass more is 12 visits'
    /// caches and state); a larger value is taken for a bad file.
    static constexpr int kMaxPasses = 8;

    float* stream(int k) const { return r_ + (size_t) k * n_embd_; }
    /// Where a layer works on the stream of a call of n tokens: r_ itself for one, else res_ (gathered by the head,
    /// scattered back by the tail).
    float* work_stream(int n) const { return n == 1 ? stream(kCur) : res_; }
    void gather(int k, float* dst, int n, int first = 0);
    void scatter(int k, const float* src, int n);
    void copy_stream(int from, int to, int n);
    void attn(Block& B, int il, const float* xq, const float* xk, const float* xv, const void* const* q8, int n,
              bool fuse);
    /// A token alone (not step()'s rows, no --dump-layers) goes in fewer launches, the same bits: the visit before's
    /// tail convolution with the first mix, K and V's store with the clips, the attention's gate with its output,
    /// three input norms in one launch, the attention's convolution with the FFN's mix and, with the experts on the
    /// card, their sum with the tail's convolution.
    bool fused(int n) const { return n == 1 && !rows_of_ && !opt_.dump; }
    int64_t attn_room(const Block& B, int n) const;
    void post_conv(const float* x, float* state, const Dev& w, float* acc, int n, bool fresh, const Row* rows);
    void conv_snaps(const float* x, const float* state, int channels, int n);
    void dump(const char* name, int il, const float* dev, int width, int n);
    int slot_of(int v) const;   ///< llama.cpp's layer index of visit v (its cache slot), for the dumps
    template <class F>
    void window_runs(int64_t pos, F&& f) const;

    std::vector<Block> blocks_;   ///< one a visit, as layers_
    int n_phys_ = 0, core_start_ = 0, core_end_ = 0, n_passes_ = 0, rank_ = 0;
    int n_head_ = 0, n_kv_ = 0, head_dim_ = 0, n_rel_ = 0, extent_ = 0, n_conv_ = 4;
    int n_vocab_real_ = 0;
    float qk_clip_ = 0, rel_clip_ = 0, softcap_ = 0, w_scale_ = 0;
    float norm_eps_ = 1e-6f;      ///< the depth mix's and the controls' (the weighted norms take rms_eps_)
    float tau_alpha_ = 0;
    int tau_floor_ = 0;
    int window_ = 0;              ///< the tokens a window layer attends, the token itself among them
    int ring_ = 0;                ///< rows of a window layer's caches: the window and a call's tokens
    int keep_ = 0;                ///< rows of a window layer a sequence's state keeps
    Dev tok_norm_, out_query_;
    size_t conv_floats_ = 0;      ///< one visit's convolution state

    // device scratch, rows_ tokens each
    float *res_ = nullptr;                 ///< the stream (kCur) from a layer's head to its tail
    const Row* tail_rows_ = nullptr;       ///< step()'s rows (rows_of_) from a layer's head to its tail
    InklingConvIn tail_conv_;              ///< a token's tail convolution, left to the next visit's head (fused())
    bool tail_wait_ = false;               ///< tail_conv_ is still to be done
    float *blk_[4] = {};                   ///< the blocks, gathered from r_
    float *bank_ = nullptr, *bank_out_ = nullptr;
    float *msg_[4] = {};                   ///< a core layer's messages: q, k, v, ffn
    float *mix_[3] = {};                   ///< the attention's inputs out of the depth mix
    float *in_[3] = {};                    ///< the normed (and controlled) q, k, v inputs
    float *ctl_[3] = {}, *ctl_work_ = nullptr;
    void* inq_[3] = {};                    ///< in_'s Q8_1 copies
    void* attq_ = nullptr;                 ///< a token's gated attention output's Q8_1 copy (fused())
    float* neg_inf_ = nullptr;             ///< -inf for the padded vocabulary's scores
    void* mix_work_ = nullptr;             ///< inkling_mix_conv()'s blocks' meeting place
    float *q_ = nullptr, *kraw_ = nullptr, *k_ = nullptr, *vraw_ = nullptr, *v_ = nullptr, *rel_ = nullptr,
          *sink_ = nullptr, *gate_ = nullptr, *ktap_ = nullptr, *vtap_ = nullptr, *att_ = nullptr, *o_ = nullptr;
    float* awork_ = nullptr;
    size_t awork_n_ = 0;
    bool int8_dot_ = false;   ///< STRATA_KV_INT8: decode attention by __dp4a on the q8_0 cache
    /// unsloth's layout: architecture "inkling2", block_count the visits, loop_{attn,k,v,mlp}_*
    bool unsloth_ = false;
    void* msg_mem_ = nullptr;              ///< that layout's message queries, a core layer's [2304, 4] each
    static constexpr int kFusedRows = 16;  ///< the most tokens the controllers' and the router's fused kernels take

    // ---- the DSpark drafter: --draft-model, a "dflash" file (inkling.cpp) ----
    /// A layer of the drafter: dense, pre-norm, its attention's keys and values of the model's rows (k_ctx, v_ctx)
    /// and of the block's own in one cache, a row a position (a ring of d_ring_ rows in a window layer).
    struct SparkLayer {
        bool swa = false;
        Dev attn_norm, wq, wk, wv, wo, q_norm, k_norm, k_ctx, v_ctx, k_ctx_norm, ctx_norm, ffn_norm, gate, up, down;
        uint16_t* k = nullptr;
        uint16_t* v = nullptr;
    };
    void load_spark_shape();
    size_t spark_row() const { return (size_t) s_kv_ * s_dim_ * sizeof(uint16_t); }   ///< a key's or a value's row
    size_t spark_extra() const;
    template <class F>
    void spark_window(int64_t pos, F&& f) const;
    void spark_store(const SparkLayer& D, const float* k, const float* v, int64_t pos, int n);
    void spark_rows(int64_t pos, int n, const float* const* x, const int64_t* stride);
    int spark_block(int32_t token, int64_t pos0, int most, Likeliest* out);
    bool spark_ = false;
    std::vector<SparkLayer> spark_layers_;
    int taps_[kTaps] = {};        ///< the visits whose input streams it reads; the model's output is its fourth input
    int s_head_ = 0, s_kv_ = 0, s_dim_ = 0, s_ff_ = 0, s_rank_ = 0, s_block_ = 0, s_window_ = 0, s_ring_ = 0;
    float s_eps_ = 0, s_theta_ = 0, s_cap_ = 0;
    float fuse_[4][4] = {};       ///< softmax(dspark_fuse): a row for each layer's context and one for the head's
    Dev s_out_norm_, s_mask_, s_ctx_norm_, s_joint_, s_joint_b_, s_hout_, s_hout_b_;
    const uint16_t* markov_ = nullptr;   ///< markov_w1 (BF16, rank a token), read in the file
    float* s_ctx_[4] = {};               ///< a call's rows fused: each layer's context and the head's
    float* s_rnn_ = nullptr;             ///< the head's context of the last s_block_ + 1 positions, a ring (seq_alloc)
    float *s_x_ = nullptr, *s_xn_ = nullptr, *s_q_ = nullptr, *s_k_ = nullptr, *s_v_ = nullptr, *s_qr_ = nullptr,
          *s_kr_ = nullptr, *s_att_ = nullptr, *s_g_ = nullptr, *s_u_ = nullptr, *s_cur_ = nullptr,
          *s_base_ = nullptr, *s_bias_ = nullptr, *s_in_ = nullptr, *s_jo_ = nullptr, *s_state_ = nullptr,
          *s_latent_ = nullptr, *s_corr_ = nullptr, *s_work_ = nullptr;
    float* h_w1_ = nullptr;              ///< pinned: a markov_w1 row as floats, then a token's embedding
    Likeliest* h_pick_ = nullptr;        ///< pinned: a slot's guess

public:
    /// Test only (poly_dspark_parity): the drafter takes n rows of the model from position pos as features (host,
    /// a row the four inputs of inkling_dspark_fuse() one after the other), then guesses `most` tokens after
    /// `token` at pos0 into tokens; logits gets each guess's scores (most rows of n_vocab_real() floats, host).
    void spark_rows_for_test(int64_t pos, int n, const float* features);
    int spark_block_for_test(int32_t token, int64_t pos0, int most, int32_t* tokens, float* logits);
    /// Test only: a conversation's state at pos and its rows before pos saved (save_state(), save_rows()), the
    /// drafter's caches zeroed, and the two loaded back, as a conversation cache's turn does.
    void spark_reload_for_test(int64_t pos);
    int n_vocab_real() const { return n_vocab_real_; }
};

}  // namespace strata::poly
