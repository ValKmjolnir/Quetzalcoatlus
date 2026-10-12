#include "util/cli.hpp"

namespace quetzal::util {

cli::cli(int argc, const char** argv) {
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

const char* cli::version() const {
    // version 1.0.0 we finished the inference engine, with simple kv-cache
    return "1.0.0";
}

}
