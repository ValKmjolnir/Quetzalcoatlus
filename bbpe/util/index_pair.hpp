#pragma once

#include <cstdint>
#include <utility>

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
