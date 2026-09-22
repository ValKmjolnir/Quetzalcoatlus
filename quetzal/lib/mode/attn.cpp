#include "mode/attn.hpp"
#include "mode/utils.hpp"

#include "tensor/linalg.hpp"
#include "tensor/weights_manager.hpp"
#include "bbpe/tokenizer.hpp"
#include "gpt/gpt2.hpp"
#include "util/chat_message.hpp"
#include "util/utf8.hpp"

namespace quetzal::mode {

void attn_mode(const quetzal::util::cli& cli) {
    quetzal::weights_manager wm(cli.get_weight_file_path());
    quetzal::bbpe::bin_reader br(cli.get_tokenizer_file_path());
    quetzal::bbpe::tokenizer tokenizer(br);

    quetzal::gpt::model_config cfg = quetzal_gpt2_50M_config();
    quetzal::gpt::gpt2 model(wm, cfg);
    std::mt19937_64 gen(42);

    quetzal::util::message_manager mm(tokenizer);
    mm.push("system", "You are a helpful assistant.");
    mm.push("user", "你好，今天感觉怎么样？");

    std::string prompt = mm.build(cfg.max_seq_len);
    auto indices = tokenizer.encode(prompt);
    auto input = quetzal::tensor::embedding_gather(model.get_tok_emb(), indices);

    auto transformer = model.get_blocks().front();
    auto attn = transformer.get_attention();
    input = quetzal::tensor::layernorm(input,
                                       transformer.get_layernorm1_weight(),
                                       transformer.get_layernorm1_bias());
    auto attn_mat = attn.forward_attn(input);

    attn_mat.dump(std::cout);
    // QUETZAL_ASSERT(attn_mat.shape().size() == 3, "attn_mat.shape().size() != 3");

    // auto head = attn_mat.shape()[0];
    // auto seq_len = attn_mat.shape()[1];
    // QUETZAL_ASSERT(attn_mat.shape()[1] == attn_mat.shape()[2],
    //                "attn_mat.shape()[1] != attn_mat.shape()[2]");
    // for (std::size_t i = 0; i < head; ++i) {
    //     auto offset = i * seq_len * seq_len;
    // }
}

}