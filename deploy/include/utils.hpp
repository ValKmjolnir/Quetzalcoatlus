#pragma once

#include "tensor.hpp"

namespace quetzal::utils {

template <typename T>
void debug_init(quetzal::tensor::tensor<T>& input, T div_base = T(1)) {
    std::size_t n = input.total_size();
    for (std::size_t i = 0; i < n; i++) {
        input.data()[i] = i / div_base;
    }
}

}
