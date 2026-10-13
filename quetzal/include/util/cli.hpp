#pragma once

#include <cstdlib>
#include <string>
#include <iostream>

namespace quetzal::util {

class cli {
public:
    enum class mode {
        unknown,
        chat,
        experimental,
        attn,
        perf
    };

private:
    std::string executable_name;
    std::string model_file_path;
    mode run_mode = mode::unknown;
    float temperature = 0.8f;
    float repetition_penalty = 1.15f;
    std::size_t top_k = 20;
    float top_p = 0.9f;
    bool disable_system_prompt = false;

private:
    void help(std::ostream& out) const;

    void report_and_exit(std::ostream& out, int exit_code = 1) const {
        logo(out);
        help(out);
        std::exit(exit_code);
    }

public:
    cli(int argc, const char** argv);
    const char* version() const;
    void logo(std::ostream& os) const;
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
    float get_top_p() const {
        return top_p;
    }
    bool system_prompt_disabled() const {
        return disable_system_prompt;
    }
};

}
