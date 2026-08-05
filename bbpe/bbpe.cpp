#include "bbpe.hpp"
#include "util/utf8.hpp"
#include "util/byte_to_unicode.hpp"

#include <fstream>
#include <iostream>

namespace bbpe {

void BBPE::init() {
    for (std::uint32_t i = 0; i < 256; i++) {
        auto c = static_cast<char>(i);
        vocab_index.insert(std::string(1, c), vocab_index.size());
        vocab.push_back(std::string(1, c));
    }
}

void BBPE::read_to_text(std::string& text, std::istream& ifs) {
    while (ifs) {
        std::string line;
        std::getline(ifs, line);

        if (line.empty()) {
            continue;
        }

        text += line;
    }
}

void BBPE::replace(std::vector<std::uint32_t>& src, index_pair p) {
    std::vector<std::uint32_t> new_src;

    for (std::size_t i = 0; i < src.size(); i++) {
        if (src[i] == p.left && i + 1 < src.size() && src[i + 1] == p.right) {
            auto new_str = vocab[p.left] + vocab[p.right];
            new_src.push_back(vocab_index.at(new_str));
            i++;
        } else {
            new_src.push_back(src[i]);
        }
    }

    src = new_src;
}

bool BBPE::single_merge(std::vector<std::uint32_t>& src, const std::string& text) {
    util::densemap<index_pair, std::uint32_t> merge;
    for (std::size_t i = 0; i < src.size(); i++) {
        if (i + 1 < src.size()) {
            auto key = index_pair(src[i], src[i + 1]);
            if (merge.contains(key)) {
                merge[key]++;
            } else {
                merge.insert(key, 1);
            }
        }
    }

    std::uint32_t max_count = 0;
    for (const auto& [key, value] : merge) {
        if (value > max_count) {
            max_count = value;
        }
    }

    if (max_count < 2) {
        return false;
    }

    for (const auto& [key, value] : merge) {
        if (value != max_count) {
            continue;
        }

        auto new_str = vocab[key.left] + vocab[key.right];
        if (vocab_index.contains(new_str)) {
            continue;
        }
        vocab_index.insert(new_str, vocab_index.size());
        vocab.push_back(new_str);
        pair_index.insert(key, pair_index.size());
        merge_pairs.push_back(key);
        replace(src, key);
        return true;
    }

    return false;
}

void BBPE::merge(const std::string& text) {
    auto src = encode(text);

    while (single_merge(src, text));
}

BBPE::BBPE(const std::string& path) {
    init();

    std::ifstream ifs(path);
    if (!ifs) {
        std::cerr << "cannot open " << path << std::endl;
        return;
    }

    std::string text;
    read_to_text(text, ifs);

    merge(text);
}

void BBPE::dump() const {
    std::vector<std::string> table;
    table.resize(vocab.size());
    for (const auto& [key, value] : vocab_index) {
        table[value] = key;
    }

    util::converter cvt;

    for (const auto& key : table) {
        std::cout << "[" << vocab_index.at(key) << "] ";
        utf8::print(std::cout, key) << " -> [";

        auto encoded = cvt.encode(key);
        std::cout << encoded << "] [";

        auto decoded = cvt.decode(encoded);
        utf8::print(std::cout, decoded) << "]\n";
    }
}

std::vector<std::uint32_t> BBPE::encode(const std::string& text) const {
    std::vector<std::uint32_t> result;
    for (auto c : text) {
        auto index = vocab_index.at(std::string(1, c));
        result.push_back(index);
    }

    if (merge_pairs.empty()) {
        return result;
    }

    for (std::size_t i = 0; i < merge_pairs.size(); i++) {
        auto [left, right] = merge_pairs[i];
        auto new_id = 256 + i;
        for (std::size_t pos = 0; pos + 1 < result.size(); ) {
            if (result[pos] == left && result[pos + 1] == right) {
                result[pos] = new_id;
                result.erase(result.begin() + pos + 1);
            } else {
                pos++;
            }
        }
    }

    return result;
}

std::string BBPE::decode(const std::vector<std::uint32_t>& indices) const {
    std::string result;
    for (auto i : indices) {
        result += vocab[i];
    }
    return result;
}

}
