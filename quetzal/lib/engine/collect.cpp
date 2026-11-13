#include "engine/mode.hpp"
#include "engine/utils.hpp"
#include "engine/runtime.hpp"

#include "tensor/linalg.hpp"
#include "util/chat_message.hpp"
#include "util/utf8.hpp"
#include "util/timestamp.hpp"
#include "util/fs.hpp"

#include <optional>
#include <unordered_set>

namespace quetzal::mode {

void collect_mode(const quetzal::util::cli& cli) {
    runtime::runtime rt(cli);
    util::log_manager logger(cli.enable_debug_info());
    auto& model = rt.get_model();

    std::mt19937_64 gen = create_generator(cli.get_random_seed());

    util::message_manager mm(rt.get_tokenizer());
    if (!cli.system_prompt_disabled()) {
        mm.push("system",
            cli.get_system_prompt().empty()
                ? rt.get_model_config().default_system_prompt
                : cli.get_system_prompt()
        );
    }

    utf8::utf8_stream_decoder decoder;

    const auto im_end = rt.get_vocab_index("<|im_end|>");
    for (int iter = 0; iter < 8; ++iter) {
        mm.push("user", cli.get_collect_mode_prompt());

        // reserve 256 token for reply
        std::string prompt = mm.build(rt.get_max_seq_len() - 256, cli.enable_debug_info());
        std::vector<std::uint32_t> indices = rt.get_tokenizer().encode(prompt);
        std::size_t index = indices.back();
        std::size_t prompt_end = indices.size();
        std::string output_content = "";
        bool prefill_executed = false;

        if (indices.size() >= rt.get_max_seq_len()) {
            std::cout << "[Info] warning: too long prompt: " << indices.size() << "\n\n";
        }

        while (indices.size() < rt.get_max_seq_len()) {
            auto logits = prefill_executed ? model.decode(index) : model.prefill(indices);
            prefill_executed = true;

            rt.get_penalty_applier().apply(indices, logits, prompt_end);
            logits = logits / cli.get_temperature();
            tensor::apply_topk_mask(logits, cli.get_top_k());
            tensor::apply_topp_mask(logits, cli.get_top_p());
            auto topk = tensor::softmax<float>(logits);
            index = tensor::multinomial<float>(topk, gen);
            if (index == im_end) {
                break;
            }

            indices.push_back(index);
            auto token = decoder.feed(rt.get_vocab(index));
            output_content += token;
        }
        if (indices.size() >= rt.get_max_seq_len()) {
            std::cout << "\n[Info] context out of range ("
                      << indices.size() << " >= " << rt.get_max_seq_len()
                      << "), truncated.\n";
        }

        mm.push("assistant", output_content);
        logger.log(mm.generate_log());
        mm.clear();
        std::cout << "[Info] collecting (" << iter + 1 << "/8)...\n";
    }
}

}
