#include "mode/utils.hpp"
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
    os << "[Info] mode: " << (cli.is_chat_mode() ? "chat" : "experiment") << std::endl;
    os << "[Info] model weight ready: " << cli.get_weight_file_path() << std::endl;
    os << "[Info] tokenizer ready: " << cli.get_tokenizer_file_path() << std::endl;
    os << "[Info] model ready: " << cfg.model_name << std::endl;
}

gpt::model_config quetzal_gpt2_50M_config() {
    return gpt::model_config {"quetzal-gpt2-50M", 352, 11, 30, 1024};
}

void visualize_topk(const std::vector<std::string>& vocab,
                    const tensor::tensor<float>& logits,
                    std::size_t k) {
    struct topk_pair {
        const char* content;
        float score;
    };
    std::vector<topk_pair> topk;
    for (std::size_t i = 0; i < logits.shape()[0]; ++i) {
        topk.push_back({vocab[i].data(), logits.data()[i] * 100.f});
    }
    std::sort(topk.begin(), topk.end(), [](const topk_pair& a, const topk_pair& b) {
        return a.score > b.score;
    });

    std::cout << "[Info] TopK (k = " << k << "):" << std::endl;
    for (std::size_t i = 0; i < k; ++i) {
        std::printf("%2lu. ", i + 1);
        utf8::print(std::cout, topk[i].content);
        std::printf(": %.2f%%\t| ", topk[i].score);
        for (int j = 0; j < int(topk[i].score); ++j) {
            std::cout << "█";
        }
        std::cout << std::endl;
    }
}

}
