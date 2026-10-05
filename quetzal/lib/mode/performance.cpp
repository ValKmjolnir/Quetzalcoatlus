#include "mode/performance.hpp"
#include "mode/utils.hpp"

#include "tensor/linalg.hpp"
#include "tensor/weights_manager.hpp"
#include "bbpe/tokenizer.hpp"
#include "gpt/gpt2.hpp"
#include "util/chat_message.hpp"
#include "util/perf_info.hpp"
#include "util/timestamp.hpp"

#include <fstream>
#include <ctime>
#include <string>

namespace quetzal::mode {

static std::string format_time(std::time_t t) {
    std::tm tm {};
#ifndef _WIN32
    localtime_r(&t, &tm);
#else
    localtime_s(&tm, &t);
#endif
    char buf[16];
    std::strftime(buf, sizeof(buf), "%Y%m%d%H%M%S", &tm);
    return buf;
}

void perf_mode(const quetzal::util::cli& cli) {
    std::cout << "[Info] performance mode\n";
    quetzal::weights_manager wm(cli.get_weight_file_path());
    quetzal::bbpe::bin_reader br(cli.get_tokenizer_file_path());
    quetzal::bbpe::tokenizer tokenizer(br);

    quetzal::gpt::model_config cfg = quetzal_gpt2_50M_config();
    quetzal::gpt::gpt2 model(wm, cfg);
    std::mt19937_64 gen(42);

    quetzal::util::message_manager mm(tokenizer);
    mm.push("system", "You are a helpful assistant.");
    mm.push("user", "你好，今天感觉怎么样？");
    mm.push("assistant", "你好，今天感觉不错。");
    mm.push("user", "今天中午准备吃点什么？");
    mm.push("assistant", "今天中午准备吃烤牛排。");
    mm.push("user", "今天晚上准备吃点什么？");
    mm.push("assistant", "今天晚上准备吃烤鸡。");
    mm.push("user", "那吃的是相当不错了");

    std::string prompt = mm.build(cfg.max_seq_len);
    std::vector<std::uint32_t> indices = tokenizer.encode(prompt);
    util::timestamp ts;

    // warm up for 5 cycles
    for (int i = 0; i < 5; ++i) {
        ts.stamp();
        auto logits = model.forward(indices);
        logits = logits / cli.get_temperature();
        quetzal::tensor::apply_topk_mask(logits, cli.get_top_k());
        auto topk = quetzal::tensor::softmax<float>(logits);
        auto index = quetzal::tensor::multinomial(topk, gen);
        indices.push_back(index);
        std::cout << "[Info] performance: warmup " << i + 1 << " cycle(s) "
                  << ts.elapsed_milli_seconds().count() << " ms\n";
    }

    std::string file = cfg.model_name + "-" + format_time(std::time(nullptr)) + ".perf.txt";
    std::ofstream perf_file_output(file);
    // perf for 5 cycles
    for (int i = 0; i < 5; ++i) {
        std::cout << "[Info] performance: test " << i + 1 << " cycle(s)\n";
        util::perf_info pi;
        pi.indices_length = indices.size();
        auto logits = model.forward_perf(indices, pi);
        
        ts.stamp();
        logits = logits / cli.get_temperature();
        quetzal::tensor::apply_topk_mask(logits, cli.get_top_k());
        auto topk = quetzal::tensor::softmax<float>(logits);
        auto index = quetzal::tensor::multinomial(topk, gen);
        pi.token_choose_perf = ts.elapsed_micro_seconds();
        indices.push_back(index);

        pi.dump(perf_file_output);
    }
    std::cout << "[Info] performance: written to " << file << "\n";
}

}
