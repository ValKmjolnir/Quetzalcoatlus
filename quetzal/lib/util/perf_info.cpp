#include "util/perf_info.hpp"

namespace quetzal::util {

void perf_info::dump(std::ostream& os) const {
    os << "===========================\n";
    os << "Performance (indices length: " << indices_length << "):\n";
    os << " - Transformer Block:\n";
    for (std::size_t i = 0; i < transformer_perf.size(); ++i) {
        os << "  - block." << i << ": " << transformer_perf[i] << " ms ";
        os << "(" << transformer_perf[i] * 100.f / total_perf << "%)\n";
    }
    os << " - Layernorm: " << layernorm_perf << " ms ";
    os << "(" << layernorm_perf * 100.f / total_perf << "%)\n";
    os << " - Logits   : " << logits_calc_perf << " ms ";
    os << "(" << logits_calc_perf * 100.f / total_perf << "%)\n";
    os << " - Total    : " << total_perf << " ms\n";
}

}
