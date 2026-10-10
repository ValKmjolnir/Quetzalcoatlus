#pragma once

#include "tensor/weights_manager.hpp"
#include "bbpe/bin_reader.hpp"
#include "bbpe/tokenizer.hpp"
#include "model/config.hpp"
#include "util/cli.hpp"

namespace quetzal {

class runtime_context {
private:
    weights_manager wm;
    bbpe::bin_reader br;
    bbpe::tokenizer tk;
    gpt::model_config cfg;

    float temperature;
    float repetition_penalty;

public:
    runtime_context(const util::cli& cli,
                    const gpt::model_config& mc);
    const weights_manager& get_weights_manager() const { return wm; }
    const bbpe::bin_reader& get_bbpe_bin() const { return br; }
    const bbpe::tokenizer& get_tokenizer() const { return tk; }
    const gpt::model_config& get_model_config() const { return cfg; }
    float get_temperature() const {
        return temperature;
    }
    float get_repetition_penalty() const {
        return repetition_penalty;
    }
};

}
