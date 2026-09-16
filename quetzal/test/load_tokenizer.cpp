#include "bbpe/bin_reader.hpp"

#include <iostream>

int main(int argc, const char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: load_tokenizer <tokenizer.bin>\n";
        return -1;
    }

    quetzal::bbpe::bin_reader br(argv[1]);

    const auto& vocab_map = br.get_vocab_index();
    const auto& vocab = br.get_vocab();
    const auto& merges = br.get_merges();

    bool verify = true;
    for (const auto& [key, value] : vocab_map) {
        if (vocab[value] != key) {
            verify = false;
            std::cerr << "[Error] vocab_map and vocab are inconsistent\n";
        }
    }
    if (verify) {
        std::cout << "[Info] Vocab map and vocab are consistent\n";
    }

    for (const auto& [lhs, rhs] : merges) {
        std::cout << "[Info] " << lhs << " : " << rhs << "\t|"
                  << vocab[lhs] << " " << vocab[rhs] << "\n";
    }
    return 0;
}