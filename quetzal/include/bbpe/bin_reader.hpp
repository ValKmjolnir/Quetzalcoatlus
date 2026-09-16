#pragma once

#include "util/densemap.hpp"
#include "bbpe/index_pair.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace quetzal::bbpe {

class bin_reader {
private:
    util::densemap<std::string, std::uint32_t> vocab_index;
    std::vector<std::string> vocab;

    std::vector<std::string> special_vocab;

    std::vector<index_pair> merge_pairs;

public:
    bin_reader(const std::string& path);

    const util::densemap<std::string, std::uint32_t>& get_vocab_index() const { return vocab_index; }
    const std::vector<std::string>& get_vocab() const { return vocab; }
    const std::vector<std::string>& get_special_vocab() const { return special_vocab; }
    const std::vector<index_pair>& get_merges() const { return merge_pairs; }
};

}
