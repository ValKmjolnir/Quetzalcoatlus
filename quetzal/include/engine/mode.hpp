#pragma once

#include "util/cli.hpp"

namespace quetzal::mode {

void attn_mode(const util::cli& cli);
void chat_mode(const util::cli& cli);
void experiment_mode(const util::cli& cli);
void perf_mode(const util::cli& cli);
void collect_mode(const util::cli& cli);

}