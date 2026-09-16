#include "gpt/gpt2.hpp"
#include "tensor/linalg.hpp"

#include <string>
#include <cstring>

namespace quetzal::gpt {

gpt2::gpt2(const weights_manager& wm) :
    tok_emb(wm.get("tok_emb.weight")),
    ln_f_w(wm.get("ln_f.weight")),
    ln_f_b(wm.get("ln_f.bias")),
    lm_head(wm.get("tok_emb.weight")) {
    for (int i = 0; i < 30; ++i) {
        std::string prefix = "blocks." + std::to_string(i) + ".";
        blocks.emplace_back(
            352, 11,
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
    auto logits = tensor::matmul_2d<float>(h, lm_head.transpose(0, 1));
    return logits;
}

}
