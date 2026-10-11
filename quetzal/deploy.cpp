#include "engine/attn.hpp"
#include "engine/chat.hpp"
#include "engine/experiment.hpp"
#include "engine/performance.hpp"

#ifdef _WIN32
#include <windows.h>
#endif

void windows_console_enable_utf8() {
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);
#endif
}

int main(int argc, const char* argv[]) {
    windows_console_enable_utf8();

    quetzal::util::cli cli(argc, argv);

    try {
        if (cli.is_chat_mode()) {
            quetzal::mode::chat_mode(cli);
            return 0;
        } else if (cli.is_attn_mode()) {
            quetzal::mode::attn_mode(cli);
            return 0;
        } else if (cli.is_perf_mode()) {
            quetzal::mode::perf_mode(cli);
            return 0;
        }

        quetzal::mode::experiment_mode(cli);
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "[Error] " << e.what() << std::endl;
        return 1;
    }
}
