#include "gpt/gpt2.hpp"
#include "tensor/linalg.hpp"

#include <string>
#include <cstring>
#include <chrono>

namespace quetzal::gpt {

gpt2::gpt2(const weights_manager& wm, const model_config& cfg) :
    tok_emb(wm.get("tok_emb.weight")),
    rope(cfg.max_seq_len, cfg.d_model / cfg.n_head),
    ln_f_w(wm.get("ln_f.weight")),
    ln_f_b(wm.get("ln_f.bias")),
    lm_head_pre_transposed(wm.get("tok_emb.weight").transpose(0, 1).contiguous()) {
    for (std::size_t i = 0; i < cfg.n_layer; ++i) {
        std::string prefix = "blocks." + std::to_string(i) + ".";
        blocks.emplace_back(
            cfg.d_model, cfg.n_head, rope,
            wm.get(prefix + "ln1.weight"), wm.get(prefix + "ln1.bias"),
            wm.get(prefix + "ln2.weight"), wm.get(prefix + "ln2.bias"),
            wm.get(prefix + "ffn.gate_proj.weight"),
            wm.get(prefix + "ffn.up_proj.weight"),
            wm.get(prefix + "ffn.down_proj.weight"),
            wm.get(prefix + "attn.Wq.weight"),
            wm.get(prefix + "attn.Wk.weight"),
            wm.get(prefix + "attn.Wv.weight"),
            wm.get(prefix + "attn.Wo.weight")
        );
    }
}

tensor::tensor<float> gpt2::forward(const std::vector<std::uint32_t>& indices) const {
    auto h = tensor::embedding_gather<float>(tok_emb, indices);
    for (const auto& block : blocks) {
        h = block.forward(h);
    }

    h = tensor::layernorm<float>(h, ln_f_w, ln_f_b);
    auto logits = tensor::matmul_2d<float>(h, lm_head_pre_transposed);
    return logits;
}

tensor::tensor<float> gpt2::forward_perf(const std::vector<std::uint32_t>& indices,
                                         util::perf_info& pi) const {
    using clk = std::chrono::high_resolution_clock;

    auto total_begin = clk::now();

    auto h = tensor::embedding_gather<float>(tok_emb, indices);
    for (const auto& block : blocks) {
        auto start = clk::now();
        h = block.forward(h);
        auto end = clk::now();
        auto dur = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        pi.transformer_perf.push_back(dur / 1000.f);
    }

    auto start = clk::now();
    h = tensor::layernorm<float>(h, ln_f_w, ln_f_b);
    auto end = clk::now();
    auto dur = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    pi.layernorm_perf = dur / 1000.f;

    start = clk::now();
    auto logits = tensor::matmul_2d<float>(h, lm_head_pre_transposed);
    end = clk::now();
    dur = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    pi.logits_calc_perf = dur / 1000.f;

    auto total_end = clk::now();
    dur = std::chrono::duration_cast<std::chrono::microseconds>(total_end - total_begin).count();
    pi.total_perf = dur / 1000.f;

    return logits;
}

tensor::tensor<float> gpt2::forward_write_ppm(const std::vector<std::uint32_t>& indices,
                                              util::ppm_writer& pw) const {
    auto h = tensor::embedding_gather<float>(tok_emb, indices);
    for (const auto& block : blocks) {
        auto h_old = h;
        h = block.forward(h);
        auto delta = tensor::sub<float>(h, h_old);
        pw.add(delta);
    }

    h = tensor::layernorm<float>(h, ln_f_w, ln_f_b);
    pw.add(h);
    auto logits = tensor::matmul_2d<float>(h, lm_head_pre_transposed);
    return logits;
}

}
