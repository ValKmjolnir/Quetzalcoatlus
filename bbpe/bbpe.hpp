#pragma once

#include "util/densemap.hpp"
#include "util/index_pair.hpp"

#include <string>
#include <cstdint>
#include <vector>

namespace bbpe {

class BBPE {
private:
    util::densemap<std::string, std::uint32_t> vocab_index;
    std::vector<std::string> vocab;

    util::densemap<index_pair, std::uint32_t> pair_index;
    std::vector<index_pair> merge_pairs;

private:
    void init();
    void read_to_text(std::string& text, std::istream& ifs);
    void replace(std::vector<std::uint32_t>&, index_pair);
    bool single_merge(std::vector<std::uint32_t>&);
    void merge(const std::string& text);

public:
    BBPE(const std::string& path);
    void dump() const;
    std::vector<std::uint32_t> encode(const std::string& text) const;
    std::string decode(const std::vector<std::uint32_t>& indices) const;
};

}
