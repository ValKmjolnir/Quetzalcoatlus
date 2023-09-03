#pragma once

#include <vector>
#include <set>

namespace quetza {

class engine {
private:
    using line = std::vector<uint64_t>;
    using table = std::set<line>;

public:
    void test() const;
};

}