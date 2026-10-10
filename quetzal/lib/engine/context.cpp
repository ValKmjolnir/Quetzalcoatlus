#include "engine/context.hpp"

#include <iostream>

namespace quetzal {

runtime_context::runtime_context(const util::cli& cli,
                                 const gpt::model_config& mc):
    wm(cli.get_weight_file_path()),
    br(cli.get_tokenizer_file_path()),
    tk(br), cfg(mc),
    temperature(cli.get_temperature()),
    repetition_penalty(cli.get_repetition_penalty()) {}

}
