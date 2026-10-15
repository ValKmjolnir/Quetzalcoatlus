#include "engine/utils.hpp"
#include "util/utf8.hpp"

#include <algorithm>

namespace quetzal::mode {

void info_dump(std::ostream& os,
               const util::cli& cli,
               const gpt::model_config& cfg) {
    cli.logo(os);
    os << "[Info] version    : " << cli.version() << "\n";
    os << "[Info] mode       : " << (cli.is_chat_mode() ? "chat" : "experiment") << std::endl;
    os << "[Info] model path : " << cli.get_model_file_path() << std::endl;
    os << "[Info] model      : " << cfg.model_name << "\n\n";
}

void visualize_topk(const std::vector<std::string>& vocab,
                    const tensor::tensor<float>& logits,
                    std::size_t k) {
    struct topk_pair {
        const std::string* content;
        float score;
    };
    std::vector<topk_pair> topk;
    for (std::size_t i = 0; i < logits.shape()[0]; ++i) {
        topk.push_back({&vocab[i], logits.data()[i] * 100.f});
    }

    std::sort(topk.begin(), topk.end(), [](const topk_pair& a, const topk_pair& b) {
        return a.score > b.score;
    });

    std::uint32_t max_len = 0;
    for (std::size_t i = 0; i < k; ++i) {
        max_len = (std::max)(max_len, utf8::utf8_str_wcwidth(*topk[i].content));
    }

    std::cout << "[Info] TopK (k = " << k << "):" << std::endl;
    for (std::size_t i = 0; i < k; ++i) {
        std::printf("%2lu. ", i + 1);
        utf8::print(std::cout, topk[i].content->c_str());
        std::uint32_t pad_len = max_len - utf8::utf8_str_wcwidth(*topk[i].content);
        for (std::uint32_t j = 0; j < pad_len; ++j) {
            std::cout << " ";
        }
        std::printf(": %6.2f%% | ", topk[i].score);
        for (int j = 0; j < int(topk[i].score) / 2; ++j) {
            std::cout << "█";
        }
        std::cout << std::endl;
    }
}

}
