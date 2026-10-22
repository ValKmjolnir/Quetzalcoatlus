#pragma once

#include "tensor/tensor.hpp"

#include <vector>
#include <cstdint>

namespace quetzal::mode {

class repetition_penalty {
private:
    float repetition_penalty_;

public:
    repetition_penalty(float repetition_penalty_param) :
        repetition_penalty_(repetition_penalty_param) {}

    void apply(const std::vector<std::uint32_t>& indices,
               tensor::tensor<float>& logits,
               std::size_t prompt_end);
};

}
