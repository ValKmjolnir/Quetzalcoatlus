#pragma once

namespace quetzal::tensor {

class NTK_aware_scaling {
private:
    int factor_ = 1;

public:
    NTK_aware_scaling(int factor) : factor_(factor) {}
    int get_factor() const { return factor_; }
};

}
