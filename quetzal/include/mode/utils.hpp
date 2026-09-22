#pragma once

#include "tensor/tensor.hpp"
#include "gpt/config.hpp"
#include "util/cli.hpp"

#include <vector>
#include <iostream>

namespace quetzal::mode {

void info_dump(std::ostream& os,
               const util::cli& cli,
               const gpt::model_config& cfg);

gpt::model_config quetzal_gpt2_50M_config();

void visualize_topk(const std::vector<std::string>& vocab,
                    const tensor::tensor<float>& logits,
                    std::size_t k);

}
