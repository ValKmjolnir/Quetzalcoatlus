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
            cfg.d_model, cfg.n_head, cfg.max_seq_len, rope,
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
    h = tensor::last_stride(h);
    h = h.reshape({1, h.total_size()});

    auto logits = tensor::matmul_2d<float>(h, lm_head_pre_transposed);
    logits = logits.reshape({logits.total_size()});
    return logits;
}

tensor::tensor<float> gpt2::prefill(const std::vector<std::uint32_t>& indices) {
    auto h = tensor::embedding_gather<float>(tok_emb, indices);
    for (auto& block : blocks) {
        h = block.prefill(h);
    }

    h = tensor::layernorm<float>(h, ln_f_w, ln_f_b);
    h = tensor::last_stride(h);
    h = h.reshape({1, h.total_size()});

    auto logits = tensor::matmul_2d<float>(h, lm_head_pre_transposed);
    logits = logits.reshape({logits.total_size()});
    return logits;
}

tensor::tensor<float> gpt2::decode(std::uint32_t token) {
    std::vector<std::uint32_t> indices = {token};
    auto h = tensor::embedding_gather<float>(tok_emb, indices);
    for (auto& block : blocks) {
        h = block.decode(h);
    }

    h = tensor::layernorm<float>(h, ln_f_w, ln_f_b);
    auto logits = tensor::matmul_2d<float>(h, lm_head_pre_transposed);
    logits = logits.reshape({logits.total_size()});
    return logits;
}

tensor::tensor<float> gpt2::decode_perf(std::uint32_t token, util::perf_info& pi) {
    using clk = std::chrono::high_resolution_clock;

    auto total_begin = clk::now();

    std::vector<std::uint32_t> indices = {token};
    auto h = tensor::embedding_gather<float>(tok_emb, indices);
    for (auto& block : blocks) {
        util::transformer_perf_info tpi;
        h = block.decode_perf(h, tpi);
        pi.transformer_perf.push_back(tpi);
    }

    auto start = clk::now();
    h = tensor::layernorm<float>(h, ln_f_w, ln_f_b);
    auto end = clk::now();
    pi.layernorm_perf = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    start = clk::now();
    auto logits = tensor::matmul_2d<float>(h, lm_head_pre_transposed);
    logits = logits.reshape({logits.total_size()});
    end = clk::now();
    pi.logits_calc_perf = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    auto total_end = clk::now();
    auto dur = std::chrono::duration_cast<std::chrono::microseconds>(total_end - total_begin).count();
    pi.total_perf = dur / 1000.f;

    return logits;
}

tensor::tensor<float> gpt2::forward_write_ppm(const std::vector<std::uint32_t>& indices,
                                              util::ppm_writer& pw) const {
    auto h = tensor::embedding_gather<float>(tok_emb, indices);
    for (const auto& block : blocks) {
        auto h_old = h;
        h = block.forward(h);
        auto delta = h - h_old;
        pw.add(delta);
    }

    h = tensor::layernorm<float>(h, ln_f_w, ln_f_b);
    pw.add(h);
    h = tensor::last_stride(h);
    h = h.reshape({1, h.total_size()});

    auto logits = tensor::matmul_2d<float>(h, lm_head_pre_transposed);
    logits = logits.reshape({logits.total_size()});
    return logits;
}

}
