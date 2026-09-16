#include "tensor/weights_manager.hpp"
#include "gpt/gpt2.hpp"
#include "bbpe/bin_reader.hpp"
#include "bbpe/tokenizer.hpp"

#include <cstdint>
#include <iostream>

int main(int argc, const char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: deploy <weights.bin> <tokenizer.bin>\n";
        return -1;
    }

    quetzal::weights_manager wm(argv[1]);
    quetzal::bbpe::bin_reader br(argv[2]);
    quetzal::bbpe::tokenizer tokenizer(br);

    quetzal::gpt::gpt2 model(wm);
    std::cout << "[Info] model ready" << std::endl;

    std::string input;
    std::cout << ">>> ";
    std::cin >> input;
    std::vector<std::uint32_t> indices = tokenizer.encode(input);

    auto res = model.forward(indices);
    auto stride = res.shape().back();
    res.dump_info(std::cout);
    res.dump(std::cout);
    return 0;
}
