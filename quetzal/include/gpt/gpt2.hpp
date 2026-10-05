#pragma once

#include "tensor/tensor.hpp"
#include "tensor/rope.hpp"
#include "tensor/weights_manager.hpp"
#include "gpt/transformer.hpp"
#include "gpt/config.hpp"
#include "util/ppm.hpp"
#include "util/perf_info.hpp"

#include <vector>

namespace quetzal::gpt {

class gpt2 {
private:
    tensor::tensor<float> tok_emb;
    tensor::rope<float> rope;
    std::vector<transformer> blocks;
    tensor::tensor<float> ln_f_w;
    tensor::tensor<float> ln_f_b;
    tensor::tensor<float> lm_head_pre_transposed;

public:
    gpt2(const weights_manager& wm, const model_config& cfg);
    ~gpt2() = default;
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
