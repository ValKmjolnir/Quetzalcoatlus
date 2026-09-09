#include "weights.hpp"

#include <cstdint>
#include <iostream>

int main(int argc, const char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: qgpt-load <weights.bin> [tensor_name]\n";
        return -1;
    }

    const auto tensors = quetzal::load_weights(argv[1]);

    // with a tensor name, dump its leading values for sanity checking
    if (argc >= 3) {
        auto it = tensors.find(argv[2]);
        if (it == tensors.end()) {
            std::cerr << "tensor not found: " << argv[2] << "\n";
            return -1;
        }
        const auto& t = it->second;
        std::cout << t.name << "  [";
        for (std::size_t i = 0; i < t.shape.size(); ++i) {
            if (i) {
                std::cout << ", ";
            }
            std::cout << t.shape[i];
        }
        std::cout << "]\n";
        t.to_tensor().dump(std::cout);
        return 0;
    }

    std::size_t total = 0;
    for (const auto& [name, t] : tensors) {
        std::size_t n = t.numel();
        total += n;
        std::cout << name << "  [";
        for (std::size_t i = 0; i < t.shape.size(); ++i) {
            if (i) {
                std::cout << ", ";
            }
            std::cout << t.shape[i];
        }
        std::cout << "]  (" << n << ")\n";
    }

    std::cout << "\n========================================================\n";
    std::cout << "total: " << tensors.size() << " tensors, "
              << total << " params (" << (total * sizeof(float) / 1024 / 1024)
              << " MB fp32)\n";
    return 0;
}
