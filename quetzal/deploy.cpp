#include "tensor/weights_manager.hpp"

#include <cstdint>
#include <iostream>

int main(int argc, const char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: deploy <weights.bin> [tensor_name]\n";
        return -1;
    }

    quetzal::weights_manager wm(argv[1]);

    // with a tensor name, dump its leading values for sanity checking
    if (argc >= 3) {
        if (!wm.has(argv[2])) {
            std::cerr << "tensor not found: " << argv[2] << "\n";
            return -1;
        }
        const auto& t = wm.get(argv[2]);
        std::cout << argv[2] << "  [";
        for (std::size_t i = 0; i < t.shape().size(); ++i) {
            if (i) {
                std::cout << ", ";
            }
            std::cout << t.shape()[i];
        }
        std::cout << "]\n";
        t.dump(std::cout);
        return 0;
    }

    std::size_t total = 0;
    for (const auto& [name, t] : wm.weights()) {
        std::size_t n = t.total_size();
        total += n;
        std::cout << name << "  [";
        for (std::size_t i = 0; i < t.shape().size(); ++i) {
            if (i) {
                std::cout << ", ";
            }
            std::cout << t.shape()[i];
        }
        std::cout << "]  (" << n << ")\n";
    }

    std::cout << "\n========================================================\n";
    std::cout << "total: " << wm.weights().size() << " tensors, "
              << total << " params (" << (total * sizeof(float) / 1024 / 1024)
              << " MB fp32)\n";
    return 0;
}
