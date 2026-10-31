#include "engine/utils.hpp"
#include "util/utf8.hpp"

#include <algorithm>

namespace quetzal::mode {

void info_dump(std::ostream& os,
               const util::cli& cli,
               const gpt::model_config& cfg,
               std::size_t max_seq_len) {
    cli.logo(os);
    os << "[Info] version    : " << cli.version() << std::endl;
    os << "[Info] mode       : " << cli.get_mode_name() << std::endl;
    os << "[Info] model path : " << cli.get_model_file_path() << std::endl;
    os << "[Info] model      : " << cfg.model_name << std::endl;
    os << "[Info] context    : ";
    if (max_seq_len < 1024) {
        os << max_seq_len << " tokens";
    } else {
        os << max_seq_len / 1024 << " k tokens";
    }
    os << std::endl;
    os << "[Info] openmp     : "
       << (tensor::omp_enabled ? "\033[32m[enabled]" : "\033[31m[disabled]")
       << "\033[0m";
    if (tensor::omp_enabled) {
        os << " " << tensor::get_omp_version();
        os << " (max thread " << tensor::get_omp_threads() << ")" << std::endl;
    }
    os << std::endl;
    if (cli.is_chat_mode()) {
        os << "[Info] input \033[32m/help\033[0m for help\n\n";
    }
}

void visualize_topk(const std::vector<std::string>& vocab,
                    const tensor::tensor<float>& logits,
                    std::size_t k,
                    std::uint32_t chosen) {
    struct topk_pair {
        const std::string* content;
        float score;
        std::size_t index;
    };
    std::vector<topk_pair> topk;
    for (std::size_t i = 0; i < logits.shape()[0]; ++i) {
        topk.push_back({&vocab[i], logits.data()[i] * 100.f, i});
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
        if (topk[i].index == chosen) {
            std::cout << "\033[32m";
        }
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
        if (topk[i].index == chosen) {
            std::cout << "\033[0m";
        }
        std::cout << std::endl;
    }
}

}
