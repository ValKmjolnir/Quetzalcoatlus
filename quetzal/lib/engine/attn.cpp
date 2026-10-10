#include "engine/attn.hpp"
#include "engine/utils.hpp"

#include "tensor/linalg.hpp"
#include "tensor/weights_manager.hpp"
#include "bbpe/tokenizer.hpp"
#include "model/riverstrike/riverstrike.hpp"
#include "util/utf8.hpp"

#include <cmath>

namespace quetzal::mode {

void attn_mode(const util::cli& cli) {
    weights_manager wm(cli.get_weight_file_path());
    bbpe::bin_reader br(cli.get_tokenizer_file_path());
    bbpe::tokenizer tokenizer(br);

    gpt::model_config cfg = riverstrike_jishui_v1_50M_config();
    gpt::riverstrike_jishui_v1 model(wm, cfg);
    std::mt19937_64 gen(42);

    std::string prompt = "今天中午吃什么？";
    auto indices = tokenizer.encode(prompt);
    auto input = tensor::embedding_gather(model.get_tok_emb(), indices);

    auto transformer = model.get_blocks().front();
    auto attn = transformer.get_attention();
    input = tensor::layernorm(input,
                              transformer.get_layernorm1_weight(),
                              transformer.get_layernorm1_bias());
    auto attn_mat = attn.forward_attn(input);
    QUETZAL_ASSERT(attn_mat.shape().size() == 3, "attn_mat.shape().size() != 3");

    std::uint32_t max_len = 0;
    for (auto i : indices) {
        max_len = (std::max)(max_len, utf8::utf8_str_wcwidth(br.get_vocab()[i]));
    }

    for (std::size_t i = 0; i < attn_mat.shape()[0]; ++i) {
        std::cout << "head [" << i << "]:" << std::endl;
        for (std::size_t j = 0; j < attn_mat.shape()[1]; ++j) {
            auto str = br.get_vocab()[indices[j]];
            auto len = utf8::utf8_str_wcwidth(str);
            for (std::size_t k = 0; k < max_len - len; ++k) {
                str.push_back(' ');
            }
            printf("\033[38;5;127m|%5u \033[0m\033[34;1m", indices[j]);
            utf8::print(std::cout, str);
            printf("\033[0m ");
            if ((j + 1) % 4 == 0) {
                std::cout << std::endl;
            }
        }
        if (indices.size() % 4 != 0) {
            std::cout << std::endl;
        }
        std::cout << "      ";
        for (std::size_t j = 0; j < attn_mat.shape()[1]; ++j) {
            printf("\033[38;5;127m%5u \033[0m", indices[j]);
        }
        std::cout << std::endl;
        for (std::size_t j = 0; j < attn_mat.shape()[1]; ++j) {
            printf("\033[38;5;127m%5u \033[0m", indices[j]);
            for (std::size_t k = 0; k < attn_mat.shape()[2]; ++k) {
                float item = attn_mat[i][j][k];
                if (0.0 < item && item < 0.5) {
                    std::cout << "\033[38;5;51m";
                } else if (item > 0.5) {
                    std::cout << "\033[48;5;51m";
                } else if (item < -0.5) {
                    std::cout << "\033[48;5;162m";
                } else {
                    std::cout << "\033[38;5;162m";
                }
                printf("%5.2f \033[0m", item);
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }
}

}