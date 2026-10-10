#pragma once

#include "tensor/tensor.hpp"
#include "model/config.hpp"
#include "util/cli.hpp"

#include <vector>
#include <iostream>

namespace quetzal::mode {

void info_dump(std::ostream& os,
               const util::cli& cli,
               const gpt::model_config& cfg);

gpt::model_config riverstrike_jishui_v1_50M_config();
gpt::model_config riverstrike_jishui_v1_200M_config();

void visualize_topk(const std::vector<std::string>& vocab,
                    const tensor::tensor<float>& logits,
                    std::size_t k);

}
