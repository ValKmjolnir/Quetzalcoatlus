#include "mode/chat.hpp"
#include "mode/utils.hpp"

#include "tensor/linalg.hpp"
#include "tensor/weights_manager.hpp"
#include "bbpe/tokenizer.hpp"
#include "gpt/gpt2.hpp"
#include "util/chat_message.hpp"
#include "util/utf8.hpp"
#include "util/timestamp.hpp"

#include <unordered_set>

namespace quetzal::mode {

static void apply_repetition_penalty(const std::vector<std::uint32_t>& indices,
                                     tensor::tensor<float>& logits,
                                     std::size_t prompt_end,
                                     float repetition_penalty) {
    if (repetition_penalty <= 1.0f || indices.size() <= prompt_end) {
        return;
    }
    std::unordered_set<std::uint32_t> seen;
    for (std::size_t i = prompt_end; i < indices.size(); ++i) {
        if (seen.count(indices[i])) {
            continue;
        }
        seen.insert(indices[i]);
        logits.data()[indices[i]] /= repetition_penalty;
    }
}

void chat_mode(const quetzal::util::cli& cli) {
    quetzal::weights_manager wm(cli.get_weight_file_path());
    quetzal::bbpe::bin_reader br(cli.get_tokenizer_file_path());
    quetzal::bbpe::tokenizer tokenizer(br);

    quetzal::gpt::model_config cfg = quetzal_gpt2_50M_config();
    quetzal::gpt::gpt2 model(wm, cfg);
    std::mt19937_64 gen(std::random_device{}());

    quetzal::util::message_manager mm(tokenizer);
    mm.push("system", "You are a helpful assistant.");

    quetzal::utf8::utf8_stream_decoder decoder;

    info_dump(std::cout, cli, cfg);

    const auto im_end = br.get_vocab_index().at("<|im_end|>");
    while (true) {
        std::string input;
        std::cout << ">>> " << std::flush;
        std::getline(std::cin, input);

        if (input == "/exit" || input == "/quit") {
            break;
        } else if (input == "/clear") {
            mm.clear();
            continue;
        }

        mm.push("user", input);

        // reserve 256 token for reply
        std::string prompt = mm.build(cfg.max_seq_len - 256);
        std::vector<std::uint32_t> indices = tokenizer.encode(prompt);
        std::size_t index = indices.back();
        std::size_t prompt_end = indices.size();
        std::string output_content = "";
        bool prefill_executed = false;

        if (indices.size() >= cfg.max_seq_len) {
            std::cout << "[Quetzal] warning: too long prompt: " << indices.size() << "\n\n";
        }

        util::timestamp ts;
        ts.stamp();

        std::cout << "\n[Quetzal] ";
        while (indices.size() < cfg.max_seq_len) {
            auto logits = prefill_executed ? model.decode(index) : model.prefill(indices);
            prefill_executed = true;
            apply_repetition_penalty(indices, logits, prompt_end, cli.get_repetition_penalty());
            logits = logits / cli.get_temperature();
            quetzal::tensor::apply_topk_mask(logits, cli.get_top_k());
            auto topk = quetzal::tensor::softmax<float>(logits);
            index = quetzal::tensor::multinomial<float>(topk, gen);
            if (index == im_end) {
                break;
            }
            indices.push_back(index);
            auto token = decoder.feed(br.get_vocab()[index]);
            output_content += token;
            std::cout << token << std::flush;
        }

        auto dur = ts.elapsed_micro_seconds().count() / 1000000.f;

        std::cout << "\n[Quetzal-perf] ";
        std::cout << (indices.size() - prompt_end) * 1.f / dur << " token/s\n";
        mm.push("assistant", output_content);
        const auto context_used = mm.get_token_count();
        std::cout << "[Quetzal-context] "
                  << context_used * 100.f / cfg.max_seq_len << "% used ("
                  << context_used << " tokens)\n\n";
    }
}

}
