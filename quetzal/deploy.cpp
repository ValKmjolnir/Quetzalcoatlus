#include "tensor/weights_manager.hpp"
#include "tensor/linalg.hpp"
#include "gpt/gpt2.hpp"
#include "bbpe/bin_reader.hpp"
#include "bbpe/tokenizer.hpp"

#include <cstdint>
#include <iostream>
#include <random>
#include <limits>
#include <cmath>

quetzal::tensor::tensor<float> last_stride(const quetzal::tensor::tensor<float>& t) {
    auto length = t.shape().back();
    quetzal::tensor::tensor<float> res({length});
    std::memcpy(res.data(), t.data() + t.total_size() - length, length * sizeof(float));
    return res;
}

void topk_mask(quetzal::tensor::tensor<float>& t, std::size_t k) {
    auto topk = quetzal::tensor::topk<float>(t, k);
    for (std::size_t i = 0; i < t.total_size(); ++i) {
        if (t.data()[i] >= topk) {
            t.data()[i] = -std::numeric_limits<float>::infinity();
        }
    }
}

int main(int argc, const char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: deploy <weights.bin> <tokenizer.bin>\n";
        return -1;
    }

    quetzal::weights_manager wm(argv[1]);
    quetzal::bbpe::bin_reader br(argv[2]);
    quetzal::bbpe::tokenizer tokenizer(br);

    quetzal::gpt::gpt2 model(wm);
    std::cout << "[Info] model ready" << std::endl;

    std::string input;
    std::cout << ">>> ";
    std::cin >> input;
    std::vector<std::uint32_t> indices = tokenizer.encode(input);

    auto logits = last_stride(model.forward(indices));
    topk_mask(logits, 10);
    auto topk = quetzal::tensor::softmax<float>(logits);

    std::mt19937_64 gen(std::random_device{}());
    auto index = quetzal::tensor::multinomial<float>(topk, gen);
    std::cout << "[Quetzal] index: " << index << " | "
              << tokenizer.decode({std::uint32_t(index)}) << std::endl;
    return 0;
}
