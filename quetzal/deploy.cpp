#include "engine/attn.hpp"
#include "engine/chat.hpp"
#include "engine/experiment.hpp"
#include "engine/performance.hpp"


int main(int argc, const char* argv[]) {
    quetzal::util::cli cli(argc, argv);

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
}
