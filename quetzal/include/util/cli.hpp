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
    std::string model_file_path;
    mode run_mode = mode::chat;
    float temperature = 0.8f;
    float repetition_penalty = 1.15f;
    std::size_t top_k = 20;
    bool disable_system_prompt = false;

private:
    void help(std::ostream& out) {
        out << "Usage: " << executable_name
            << " <model_file_path>" << std::endl;
        out << "Options:" << std::endl;
        out << "  --chat                  | enable chat mode (default)" << std::endl;
        out << "  --expr                  | enable experimental mode" << std::endl;
        out << "  --attn                  | enable attn visualization mode" << std::endl;
        out << "  --perf                  | enable performance debug mode" << std::endl;
        out << "  --disable-system-prompt | disable system prompt" << std::endl;
        out << "  --help                  | print this help message" << std::endl;
    }

    void report_and_exit() {
        help(std::cerr);
        std::exit(1);
    }

public:
    cli(int argc, const char** argv) {
        executable_name = argv[0];
        if (argc < 2) {
            report_and_exit();
        }
        model_file_path = argv[1];

        for (int i = 2; i < argc; i++) {
            if (std::string(argv[i]) == "--chat") {
                run_mode = mode::chat;
            } else if (std::string(argv[i]) == "--expr") {
                run_mode = mode::experimental;
            } else if (std::string(argv[i]) == "--attn") {
                run_mode = mode::attn;
            } else if (std::string(argv[i]) == "--perf") {
                run_mode = mode::perf;
            } else if (std::string(argv[i]) == "--disable-system-prompt") {
                disable_system_prompt = true;
            } else {
                report_and_exit();
            }
        }
    }

    const std::string& get_model_file_path() const {
        return model_file_path;
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
    bool system_prompt_disabled() const {
        return disable_system_prompt;
    }
};

}
