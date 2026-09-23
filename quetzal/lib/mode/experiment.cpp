#include "mode/experiment.hpp"
#include "mode/utils.hpp"

#include "tensor/linalg.hpp"
#include "tensor/weights_manager.hpp"
#include "bbpe/tokenizer.hpp"
#include "gpt/gpt2.hpp"
#include "util/chat_message.hpp"
#include "util/utf8.hpp"

namespace quetzal::mode {

void experiment_mode(const quetzal::util::cli& cli) {
    quetzal::weights_manager wm(cli.get_weight_file_path());
    quetzal::bbpe::bin_reader br(cli.get_tokenizer_file_path());
    quetzal::bbpe::tokenizer tokenizer(br);

    quetzal::gpt::model_config cfg = quetzal_gpt2_50M_config();
    quetzal::gpt::gpt2 model(wm, cfg);
    std::mt19937_64 gen(42);

    quetzal::util::message_manager mm(tokenizer);
    mm.push("system", "You are a helpful assistant.");
    mm.push("user", "你好！");

    info_dump(std::cout, cli, cfg);

    std::string prompt = mm.build(cfg.max_seq_len);
    std::vector<std::uint32_t> indices = tokenizer.encode(prompt);
    std::vector<std::uint32_t> output_indices;

    const auto im_end = br.get_vocab_index().at("<|im_end|>");
    std::uint32_t index = 0;
    std::uint32_t count = 0;
    while (indices.size() < 70) {
        quetzal::util::ppm_writer pw(
            "output." + std::to_string(count) + ".ppm",
            cfg.d_model * 2,
            (indices.size() + 1) * (cfg.n_layer + 1)
        );
        auto logits = quetzal::tensor::last_stride(model.forward_write_ppm(indices, pw));
        pw.write();

        logits = quetzal::tensor::div<float>(logits, cli.get_temperature());
        quetzal::tensor::apply_topk_mask(logits, cli.get_top_k());
        auto topk = quetzal::tensor::softmax<float>(logits);
        visualize_topk(br.get_vocab(), topk, 5);
        index = quetzal::tensor::multinomial<float>(topk, gen);
        if (index == im_end) {
            break;
        }
        indices.push_back(index);
        output_indices.push_back(index);
        ++count;
    }
    std::cout << std::endl;
    std::cout << "[Quetzal] " << tokenizer.decode(output_indices) << std::endl;
}

}
