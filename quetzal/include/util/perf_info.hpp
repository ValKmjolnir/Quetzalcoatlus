#pragma once

#include <iostream>
#include <vector>

namespace quetzal::util {

struct perf_info {
    std::size_t indices_length = 0;
    std::vector<float> transformer_perf;
    float layernorm_perf = 0.f;
    float logits_calc_perf = 0.f;
    float total_perf = 0.f;

    void dump(std::ostream&) const;
};

}
