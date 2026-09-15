#pragma once

#include "tensor/tensor.hpp"

namespace quetzal::gpt {

class multi_head_attention {
private:
    std::size_t d_model_;
    std::size_t n_head_;
    tensor::tensor<float> Wq_;
    tensor::tensor<float> Wk_;
    tensor::tensor<float> Wv_;
    tensor::tensor<float> Wo_;

public:
    multi_head_attention(std::size_t d_model,
                         std::size_t n_head,
                         const tensor::tensor<float>& Wq,
                         const tensor::tensor<float>& Wk,
                         const tensor::tensor<float>& Wv,
                         const tensor::tensor<float>& Wo) :
        d_model_(d_model), n_head_(n_head),
        Wq_(Wq), Wk_(Wk), Wv_(Wv), Wo_(Wo) {}

    tensor::tensor<float> forward(const tensor::tensor<float>& x) const;
};

}
