#pragma once

#include "tensor/tensor.hpp"
#include "tensor/rope.hpp"
#include "tensor/weights_manager.hpp"
#include "model/riverstrike/transformer.hpp"
#include "model/config.hpp"
#include "util/ppm.hpp"
#include "util/perf_info.hpp"

#include <vector>

namespace quetzal::gpt {

// first version of riverstrike model, also named jishui, which means "击水"
// name chosen from a famous poem《沁园春·长沙》: "曾记否，到中流击水，浪遏飞舟。"
class riverstrike_jishui_v1 {
private:
    tensor::tensor<float> tok_emb;
    tensor::rope<float> rope;
    std::vector<transformer> blocks;
    tensor::tensor<float> ln_f_w;
    tensor::tensor<float> ln_f_b;
    tensor::tensor<float> lm_head_pre_transposed;

public:
    riverstrike_jishui_v1(const weights_manager& wm, const model_config& cfg);
    ~riverstrike_jishui_v1() = default;
    tensor::tensor<float> forward(const std::vector<std::uint32_t>& indices) const;
    tensor::tensor<float> prefill(const std::vector<std::uint32_t>& indices);
    tensor::tensor<float> decode(std::uint32_t token);
    tensor::tensor<float> decode_perf(std::uint32_t token, util::perf_info& pi);
    tensor::tensor<float> forward_write_ppm(const std::vector<std::uint32_t>& indices,
                                            util::ppm_writer& pw) const;
    const auto& get_blocks() const { return blocks; }
    const auto& get_tok_emb() const { return tok_emb; }
};

}
