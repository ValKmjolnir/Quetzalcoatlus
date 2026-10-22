#include "engine/repetition_penalty.hpp"

namespace quetzal::mode {

void repetition_penalty::apply(const std::vector<std::uint32_t>& indices,
                               tensor::tensor<float>& logits,
                               std::size_t prompt_end) {
    if (repetition_penalty_ <= 1.0f || indices.size() <= prompt_end) {
        return;
    }
    for (std::size_t i = prompt_end; i < indices.size(); ++i) {
        if (seen.count(indices[i])) {
            continue;
        }
        seen.insert(indices[i]);
        logits.data()[indices[i]] /= repetition_penalty_;
    }
}

}
