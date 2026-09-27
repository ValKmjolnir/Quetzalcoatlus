#pragma once

#include <iostream>
#include <vector>
#include <chrono>

namespace quetzal::util {

struct transformer_perf_info {
    std::chrono::microseconds attn_time;
    std::chrono::microseconds ffn_time;
    float total_time = 0.f;
};

struct perf_info {
    std::size_t indices_length = 0;
    std::vector<transformer_perf_info> transformer_perf;
    float layernorm_perf = 0.f;
    float logits_calc_perf = 0.f;
    float total_perf = 0.f;

    void dump(std::ostream&) const;
};

}
