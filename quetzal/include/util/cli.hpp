#pragma once

#include <cstdlib>
#include <string>
#include <iostream>

namespace quetzal::util {

class cli {
public:
    enum class mode {
        chat,
        experimental,
        attn,
        perf
    };

private:
    std::string executable_name;
    std::string weight_file_path;
    std::string tokenizer_file_path;
    mode run_mode = mode::experimental;
    float temperature = 0.8f;
    float repetition_penalty = 1.15f;
    std::size_t top_k = 20;

private:
    void help(std::ostream& out) {
        out << "Usage: " << executable_name
            << " <model_file_path> <tokenizer_file_path>"
            << std::endl;
        out << "Options:" << std::endl;
        out << "  --chat | enable chat mode" << std::endl;
        out << "  --expr | enable experimental mode (default)" << std::endl;
        out << "  --attn | enable attn visualization mode" << std::endl;
        out << "  --perf | enable performance debug mode" << std::endl;
        out << "  --help | print this help message" << std::endl;
    }

    void report_and_exit() {
        help(std::cerr);
        std::exit(1);
    }

public:
    cli(int argc, const char** argv) {
        executable_name = argv[0];
        if (argc < 3) {
            report_and_exit();
        }
        weight_file_path = argv[1];
        tokenizer_file_path = argv[2];

        for (int i = 3; i < argc; i++) {
            if (std::string(argv[i]) == "--chat") {
                run_mode = mode::chat;
            } else if (std::string(argv[i]) == "--expr") {
                run_mode = mode::experimental;
            } else if (std::string(argv[i]) == "--attn") {
                run_mode = mode::attn;
            } else if (std::string(argv[i]) == "--perf") {
                run_mode = mode::perf;
            } else {
                report_and_exit();
            }
        }
    }

    const std::string& get_weight_file_path() const {
        return weight_file_path;
    }
    const std::string& get_tokenizer_file_path() const {
        return tokenizer_file_path;
    }
    bool is_chat_mode() const {
        return run_mode == mode::chat;
    }
    bool is_experimental_mode() const {
        return run_mode == mode::experimental;
    }
    bool is_attn_mode() const {
        return run_mode == mode::attn;
    }
    bool is_perf_mode() const {
        return run_mode == mode::perf;
    }
    float get_temperature() const {
        return temperature;
    }
    float get_repetition_penalty() const {
        return repetition_penalty;
    }
    std::size_t get_top_k() const {
        return top_k;
    }
};

}
