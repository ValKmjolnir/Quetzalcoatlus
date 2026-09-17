#include "tensor/weights_manager.hpp"
#include "tensor/linalg.hpp"
#include "gpt/gpt2.hpp"
#include "bbpe/bin_reader.hpp"
#include "bbpe/tokenizer.hpp"
#include "util/ppm.hpp"

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
        if (t.data()[i] < topk) {
            t.data()[i] = -std::numeric_limits<float>::infinity();
        }
    }
}

std::string build_prompt(const std::string& input) {
    std::string prompt = "<|im_start|>system\n";
    prompt += "You are a helpful assistant.<|im_end|>\n";
    prompt += "<|im_start>user\n" + input + "<|im_end|>\n";
    prompt += "<|im_start|>assistant\n";
    return prompt;
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
    std::mt19937_64 gen(std::random_device{}());
    std::cout << "[Info] model ready" << std::endl;

    std::string input;
    std::cout << ">>> ";
    std::cin >> input;
    std::string prompt = build_prompt(input);
    std::vector<std::uint32_t> indices = tokenizer.encode(prompt);

    const auto im_end = br.get_vocab_index().at("<|im_end|>");
    std::uint32_t index = 0;
    std::uint32_t count = 0;
    while (index != im_end && indices.size() < 100) {
        quetzal::util::ppm_writer pw("output." + std::to_string(count) + ".ppm",
                                     352 * 2 + 352 / 11 * 2, (indices.size() + 1) * 31);
        auto logits = last_stride(model.forward_write_ppm(indices, pw));
        auto temperature = 0.8f;
        logits = quetzal::tensor::div<float>(logits, temperature);
        topk_mask(logits, 30);
        auto topk = quetzal::tensor::softmax<float>(logits);
        index = quetzal::tensor::multinomial<float>(topk, gen);
        indices.push_back(index);
        ++count;
        std::cout << " | thinking... " << "/\\"[indices.size() & 1] << "\r" << std::flush;
    }
    std::cout << " | done        " << std::endl;
    std::cout << " | [Quetzal] " << tokenizer.decode(indices) << std::endl;
    return 0;
}
