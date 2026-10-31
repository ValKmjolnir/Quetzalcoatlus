#pragma once

namespace quetzal::tensor {

class NTK_aware_scaling {
private:
    float factor_ = 1.f;

public:
    NTK_aware_scaling(float factor) : factor_(factor) {}
    float get_factor() const { return factor_; }
};

}
