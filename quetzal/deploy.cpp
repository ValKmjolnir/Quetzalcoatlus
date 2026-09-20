#include "tensor/weights_manager.hpp"
#include "tensor/linalg.hpp"
#include "gpt/gpt2.hpp"
#include "bbpe/bin_reader.hpp"
#include "bbpe/tokenizer.hpp"
#include "util/ppm.hpp"
#include "util/utf8.hpp"
#include "util/chat_message.hpp"
#include "util/cli.hpp"

#include <cstdint>
#include <iostream>
#include <random>
#include <limits>
#include <cmath>

void logo_dump(std::ostream& os) {
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
               const quetzal::util::cli& cli,
               const quetzal::gpt::model_config& cfg) {
    logo_dump(os);
    os << "[Info] mode: " << (cli.is_chat_mode() ? "chat" : "experiment") << std::endl;
    os << "[Info] model weight ready: " << cli.get_weight_file_path() << std::endl;
    os << "[Info] tokenizer ready: " << cli.get_tokenizer_file_path() << std::endl;
    os << "[Info] model ready: " << cfg.model_name << std::endl;
}

quetzal::gpt::model_config quetzal_gpt2_50M_config() {
    return quetzal::gpt::model_config {"quetzal-gpt2-50M", 352, 11, 30, 1024};
}

void chat_mode(const quetzal::util::cli& cli) {
    quetzal::weights_manager wm(cli.get_weight_file_path());
    quetzal::bbpe::bin_reader br(cli.get_tokenizer_file_path());
    quetzal::bbpe::tokenizer tokenizer(br);

    quetzal::gpt::model_config cfg = quetzal_gpt2_50M_config();
    quetzal::gpt::gpt2 model(wm, cfg);
    std::mt19937_64 gen(std::random_device{}());

    quetzal::util::message_manager mm(tokenizer);
    mm.push("system", "You are a helpful assistant.");

    quetzal::utf8::utf8_stream_decoder decoder;

    info_dump(std::cout, cli, cfg);

    const auto im_end = br.get_vocab_index().at("<|im_end|>");
    while (true) {
        std::string input;
        std::cout << ">>> " << std::flush;
        std::getline(std::cin, input);
        mm.push("user", input);

        std::string prompt = mm.build(cfg.max_seq_len);
        std::vector<std::uint32_t> indices = tokenizer.encode(prompt);
        std::size_t index = indices.back();
        std::string output_content = "";

        std::cout << "[Quetzal] ";
        while (indices.size() < cfg.max_seq_len) {
            auto logits = last_stride(model.forward(indices));
            const auto temperature = 0.8f;
            logits = quetzal::tensor::div<float>(logits, temperature);
            quetzal::tensor::apply_topk_mask(logits, 30);
            auto topk = quetzal::tensor::softmax<float>(logits);
            index = quetzal::tensor::multinomial<float>(topk, gen);
            if (index == im_end) {
                break;
            }
            indices.push_back(index);
            auto token = decoder.feed(br.get_vocab()[index]);
            output_content += token;
            std::cout << token << std::flush;
        }
        std::cout << std::endl;
        mm.push("assistant", output_content);
    }
}

void visualize_topk(const std::vector<std::string>& vocab,
                    const quetzal::tensor::tensor<float>& logits,
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
        quetzal::utf8::print(std::cout, topk[i].content);
        std::printf(": %.2f%%\t| ", topk[i].score);
        for (int j = 0; j < int(topk[i].score); ++j) {
            std::cout << "█";
        }
        std::cout << std::endl;
    }
}

void experiment_mode(const quetzal::util::cli& cli) {
    quetzal::weights_manager wm(cli.get_weight_file_path());
    quetzal::bbpe::bin_reader br(cli.get_tokenizer_file_path());
    quetzal::bbpe::tokenizer tokenizer(br);

    quetzal::gpt::model_config cfg = quetzal_gpt2_50M_config();
    quetzal::gpt::gpt2 model(wm, cfg);
    std::mt19937_64 gen(42);

    quetzal::util::message_manager mm(tokenizer);
    mm.push("system", "You are a helpful assistant.");

    info_dump(std::cout, cli, cfg);

    std::string input;
    std::cout << ">>> ";
    std::getline(std::cin, input);
    mm.push("user", input);

    std::string prompt = mm.build(cfg.max_seq_len);
    std::vector<std::uint32_t> indices = tokenizer.encode(prompt);
    std::vector<std::uint32_t> output_indices;

    const auto im_end = br.get_vocab_index().at("<|im_end|>");
    std::uint32_t index = 0;
    std::uint32_t count = 0;
    while (indices.size() < 100) {
        quetzal::util::ppm_writer pw(
            "output." + std::to_string(count) + ".ppm",
            cfg.d_model * 2,
            (indices.size() + 1) * (cfg.n_layer + 1)
        );
        auto logits = last_stride(model.forward_write_ppm(indices, pw));
        auto temperature = 0.8f;
        logits = quetzal::tensor::div<float>(logits, temperature);
        quetzal::tensor::apply_topk_mask(logits, 30);
        auto topk = quetzal::tensor::softmax<float>(logits);
        visualize_topk(br.get_vocab(), topk, 5);
        index = quetzal::tensor::multinomial<float>(topk, gen);
        if (index == im_end) {
            break;
        }
        indices.push_back(index);
        output_indices.push_back(index);
        ++count;
    }
    std::cout << std::endl;
    std::cout << "[Quetzal] " << tokenizer.decode(output_indices) << std::endl;
}

int main(int argc, const char* argv[]) {
    quetzal::util::cli cli(argc, argv);

    if (cli.is_chat_mode()) {
        chat_mode(cli);
        return 0;
    }

    experiment_mode(cli);
    return 0;
}
