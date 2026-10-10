#include "engine/utils.hpp"
#include "util/utf8.hpp"

#include <algorithm>

namespace quetzal::mode {

static void logo_dump(std::ostream& os) {
    os << "\n";
    os << "      __   _  _  ____  ____  ____   __   __   \n";
    os << "     /  \\ / )( \\(  __)(_  _)(__  ) / _\\ (  )  \n";
    os << "    (  O )) \\/ ( ) _)   )(   / _/ /    \\/ (_/\\\n";
    os << "     \\__\\)\\____/(____) (__) (____)\\_/\\_/\\____/\n";
    os << "      ___  __    __  ____  __    _  _  ____   \n";
    os << "     / __)/  \\  / _\\(_  _)(  )  / )( \\/ ___)  \n";
    os << "    ( (__(  O )/    \\ )(  / (_/\\) \\/ (\\___ \\  \n";
    os << "     \\___)\\__/ \\_/\\_/(__) \\____/\\____/(____/  \n\n";
}

void info_dump(std::ostream& os,
               const util::cli& cli,
               const gpt::model_config& cfg) {
    logo_dump(os);
    os << "[Info] mode           : " << (cli.is_chat_mode() ? "chat" : "experiment") << std::endl;
    os << "[Info] weight path    : " << cli.get_weight_file_path() << std::endl;
    os << "[Info] tokenizer path : " << cli.get_tokenizer_file_path() << std::endl;
    os << "[Info] model          : " << cfg.model_name << "\n\n";
}

gpt::model_config riverstrike_jishui_v1_50M_config() {
    return gpt::model_config {
        "riverstrike-jishui-v1-50M-quetzal", 352, 11, 30, 1024,
        "你是一个聊天助手，请自然、友好地和用户聊天。"
    };
}

gpt::model_config riverstrike_jishui_v1_200M_config() {
    return gpt::model_config {
        "riverstrike-jishui-v1-200M-Base", 704, 11, 30, 2048,
        "汝乃古文助手。"
    };
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
        for (int j = 0; j < int(topk[i].score); ++j) {
            std::cout << "█";
        }
        std::cout << std::endl;
    }
}

}
