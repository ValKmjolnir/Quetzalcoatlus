#include "tensor/omp.hpp"
#include "util/cli.hpp"

#include <vector>
#include <cstring>

namespace quetzal::util {

static bool streq(const char* a, const char* b) {
    return strcmp(a, b) == 0;
}

cli::cli(int argc, const char** argv) {
    executable_name = argv[0];
    if (argc < 2) {
        report_and_exit(std::cerr, 1);
    }
    if (streq(argv[1], "--help") || streq(argv[1], "-h")) {
        report_and_exit(std::cout, 0);
    } else if (streq(argv[1], "--version") || streq(argv[1], "-v")) {
        std::cout << "Quetzalcoatlus Inference Engine v" << version()
                  << " (OpenMP "
                  << (tensor::omp_enabled ? "enabled" : "disabled")
                  << ")" << std::endl;
        std::exit(0);
    }
    model_file_path = argv[1];

    for (int i = 2; i < argc; i++) {
        if (streq(argv[i], "--help") || streq(argv[i], "-h")) {
            report_and_exit(std::cout, 0);
        } else if (streq(argv[1], "--version") || streq(argv[1], "-v")) {
            std::cout << "Quetzal v" << version()
                      << (tensor::omp_enabled ? " (OpenMP)" : "") << std::endl;
            std::exit(0);
        }

        if (streq(argv[i], "--chat") || streq(argv[i], "-c")) {
            run_mode = mode::chat;
        } else if (streq(argv[i], "--expr") || streq(argv[i], "-e")) {
            run_mode = mode::experimental;
        } else if (streq(argv[i], "--attn") || streq(argv[i], "-a")) {
            run_mode = mode::attn;
        } else if (streq(argv[i], "--perf") || streq(argv[i], "-p")) {
            run_mode = mode::perf;
        } else if (streq(argv[i], "--debug")) {
            show_debug_info = true;
        } else if (streq(argv[i], "--disable-system-prompt")) {
            disable_system_prompt = true;
        } else {
            report_and_exit(std::cerr, 1);
        }
    }

    // default chat mode
    if (run_mode == mode::unknown) {
        run_mode = mode::chat;
    }
}

void cli::help(std::ostream& out) const {
    out << "Usage: " << executable_name
        << " <model_file_path>" << std::endl;
    out << "Options:" << std::endl;
    out << "  -c, --chat              | enable chat mode (default)" << std::endl;
    out << "  -e, --expr              | enable experimental mode" << std::endl;
    out << "  -a, --attn              | enable attn visualization mode" << std::endl;
    out << "  -p, --perf              | enable performance debug mode" << std::endl;
    out << "  -h, --help              | print help message" << std::endl;
    out << "  -v, --version           | print version" << std::endl;
    out << "      --debug             | enable debug info" << std::endl;
    out << "  --disable-system-prompt | disable system prompt" << std::endl;
}

const char* cli::version() const {
    // version 1.0.0 we finished the inference engine, with simple kv-cache
    return "1.0.1";
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

    os << "\033[38;5;51m" << logo_content << "\033[0m";
}

}
