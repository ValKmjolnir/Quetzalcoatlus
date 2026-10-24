#include "engine/chat.hpp"
#include "engine/utils.hpp"
#include "engine/repetition_penalty.hpp"

#include "tensor/linalg.hpp"
#include "model/qgpt_bin_reader.hpp"
#include "model/riverstrike/riverstrike.hpp"
#include "util/chat_message.hpp"
#include "util/utf8.hpp"
#include "util/timestamp.hpp"

#include <unordered_set>

namespace quetzal::mode {

static void chat_mode_help() {
    std::cout << "[Quetzal] chat mode help:\n"
              << "     /exit    - exit chat mode\n"
              << "  /q /quit    - exit chat mode\n"
              << "  /c /clear   - clear chat history\n"
              << "     /context - show context usage\n"
              << "  /h /help    - show help message\n";
}

void chat_mode(const quetzal::util::cli& cli) {
    gpt::qgpt_bin_reader qbr(cli.get_model_file_path(), cli.enable_debug_info());
    const auto& wm = qbr.get_weights_manager();
    const auto& br = qbr.get_bin_reader();
    bbpe::tokenizer tokenizer(br);
    repetition_penalty rp(cli.get_repetition_penalty());

    const auto& cfg = qbr.get_model_config();
    gpt::riverstrike_jishui_v1 model(wm, cfg);
    std::mt19937_64 gen(std::random_device{}());

    util::message_manager mm(tokenizer);
    if (!cli.system_prompt_disabled()) {
        mm.push("system", cfg.default_system_prompt);
    }

    utf8::utf8_stream_decoder decoder;

    info_dump(std::cout, cli, cfg);

    const auto im_end = br.get_vocab_index().at("<|im_end|>");
    while (true) {
        std::string input;
        std::cout << ">>> " << std::flush;
        std::getline(std::cin, input);

        if (input == "/exit" || input == "/quit" || input == "/q") {
            break;
        } else if (input == "/c" || input == "/clear") {
            mm.clear();
            continue;
        } else if (input == "/h" || input == "/help") {
            chat_mode_help();
            continue;
        } else if (input == "/context") {
            const auto context_used = mm.get_token_count();
            std::cout << "[Quetzal-context] "
                      << context_used * 100.f / cfg.max_seq_len << "% used ("
                      << context_used << " tokens)\n";
            continue;
        }

        mm.push("user", input);

        // reserve 256 token for reply
        std::string prompt = mm.build(cfg.max_seq_len - 256, cli.enable_debug_info());
        std::vector<std::uint32_t> indices = tokenizer.encode(prompt);
        std::size_t index = indices.back();
        std::size_t prompt_end = indices.size();
        std::string output_content = "";
        bool prefill_executed = false;

        if (indices.size() >= cfg.max_seq_len) {
            std::cout << "[Quetzal] warning: too long prompt: " << indices.size() << "\n\n";
        }

        util::timestamp ts;
        float TTFT = 0.f;
        ts.stamp();

        std::cout << "\n[Quetzal] ";
        while (indices.size() < cfg.max_seq_len) {
            auto logits = prefill_executed ? model.decode(index) : model.prefill(indices);
            if (!prefill_executed) {
                TTFT = ts.elapsed_micro_seconds().count() / 1000000.f;
                ts.stamp();
            }
            prefill_executed = true;

            rp.apply(indices, logits, prompt_end);
            logits = logits / cli.get_temperature();
            tensor::apply_topk_mask(logits, cli.get_top_k());
            tensor::apply_topp_mask(logits, cli.get_top_p());
            auto topk = tensor::softmax<float>(logits);
            index = tensor::multinomial<float>(topk, gen);
            if (index == im_end) {
                break;
            }

            indices.push_back(index);
            auto token = decoder.feed(br.get_vocab()[index]);
            output_content += token;
            std::cout << token << std::flush;
        }
        if (indices.size() >= cfg.max_seq_len) {
            std::cout << "\n[Quetzal] context out of range, truncated.\n";
        }

        auto dur = ts.elapsed_micro_seconds().count() / 1000000.f;

        std::cout << "\n[Quetzal-perf] TTFT: " << TTFT << "s, decode: ";
        std::cout << (indices.size() - prompt_end) * 1.f / dur << " token/s\n";
        mm.push("assistant", output_content);
    }
}

}
