#pragma once

#include "util/densemap.hpp"

#include <string>
#include <cstdint>
#include <vector>

namespace bbpe {

struct index_pair {
    std::uint32_t left;
    std::uint32_t right;

    index_pair() : left(0), right(0) {}
    index_pair(std::uint32_t l, std::uint32_t r) : left(l), right(r) {}

    bool operator==(const index_pair& rhs) const {
        return left == rhs.left && right == rhs.right;
    }
};

} // namespace bbpe

template <>
struct std::hash<bbpe::index_pair> {
    std::size_t operator()(const bbpe::index_pair& p) const noexcept {
        auto l = static_cast<std::size_t>(p.left);
        auto r = static_cast<std::size_t>(p.right);
        l ^= r + 0x9e3779b97f4a7c15ULL + (l << 6) + (l >> 2);
        return l;
    }
};

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
    bool single_merge(std::vector<std::uint32_t>&, const std::string& text);
    void merge(const std::string& text);

public:
    BBPE(const std::string& path);
    void dump() const;
    std::vector<std::uint32_t> encode(const std::string& text) const;
    std::string decode(const std::vector<std::uint32_t>& indices) const;
};

}
