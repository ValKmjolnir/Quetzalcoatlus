#pragma once

#include <cstdlib>
#include <string>
#include <iostream>

namespace quetzal::util {

class cli {
private:
    std::string executable_name;
    std::string weight_file_path;
    std::string tokenizer_file_path;
    bool chat_mode = false;

private:
    void help(std::ostream& out) {
        out << "Usage: " << executable_name
            << " <model_file_path> <tokenizer_file_path>"
            << std::endl;
        out << "Options:" << std::endl;
        out << "  --chat | enable chat mode" << std::endl;
    }

    void report_and_exit() {
        std::cerr << "Usage: " << executable_name
                  << " <model_file_path> <tokenizer_file_path>"
                  << std::endl;
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
                chat_mode = true;
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
        return chat_mode;
    }
};

}
