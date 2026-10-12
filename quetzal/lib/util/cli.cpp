#include "util/cli.hpp"

#include <vector>
#include <random>

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

void cli::logo(std::ostream& os) const {
    std::string logo_content = "\n"
    "      __   _  _  ____  ____  ____   __   __           \n"
    "     /  \\ / )( \\(  __)(_  _)(__  ) / _\\ (  )       \n"
    "    (  O )) \\/ ( ) _)   )(   / _/ /    \\/ (_/\\     \n"
    "     \\__\\)\\____/(____) (__) (____)\\_/\\_/\\____/  \n"
    "      ___  __    __  ____  __    _  _  ____           \n"
    "     / __)/  \\  / _\\(_  _)(  )  / )( \\/ ___)       \n"
    "    ( (__(  O )/    \\ )(  / (_/\\) \\/ (\\___ \\     \n"
    "     \\___)\\__/ \\_/\\_/(__) \\____/\\____/(____/  \n\n";

    std::vector<const char*> colors = {
        "\033[38;5;51m",
        "\033[38;5;162m",
        "\033[38;5;162m",
        "\033[38;5;162m"
    };

    std::mt19937_64 gen(std::random_device{}());
    std::uniform_int_distribution<std::size_t> dist(0, colors.size() - 1);
    for (auto c : logo_content) {
        if (c == '\n' || c == ' ') {
            os << c;
            continue;
        }
        os << colors[dist(gen)] << c << "\033[0m";
    }
}

}
