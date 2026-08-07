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
    auto new_str = vocab[p.left] + vocab[p.right];
    auto id = vocab_index.at(new_str);

    std::size_t write = 0;
    for (std::size_t read = 0; read < src.size(); read ++) {
        if (src[read] == p.left && read + 1 < src.size() && src[read + 1] == p.right) {
            src[write ++] = id;
            read ++;
        } else {
            src[write ++] = src[read];
        }
    }

    src.resize(write);
}

bool BBPE::single_merge(std::vector<std::uint32_t>& src) {
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

        if (pair_index.contains(key)) {
            continue;
        }

        auto new_str = vocab[key.left] + vocab[key.right];

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
    auto prev = src.size();

    std::cout << "src: " << src.size();
    std::cout << " text: " << text.length() / 1024.0 / 1024.0 << "M\n";

    std::uint64_t count = 0;
    while (single_merge(src)) {
        count ++;
        if (count % 50 == 0) {
            std::cout << "diff: " << prev - src.size();
            std::cout << " src: " << src.size();
            std::cout << " vocab: " << vocab.size() << "\n";
            prev = src.size();
        }
    }
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

void BBPE::dump(std::ostream& os) const {
    std::vector<std::string> table;
    table.resize(vocab.size());
    for (const auto& [key, value] : vocab_index) {
        table[value] = key;
    }

    util::converter cvt;

    for (const auto& key : table) {
        os << "[" << vocab_index.at(key) << "] ";
        utf8::print(os, key) << "\n";

        auto encoded = cvt.encode(key);
        os << "  encoded: " << encoded << "\n";

        auto decoded = cvt.decode(encoded);
        os << "  decoded: ";
        utf8::print(os, decoded) << "\n";
    }
}

void BBPE::dump_json(std::ostream& os) const {
    auto raw = [&](const std::string& s) {
        for (auto c : s) {
            if (c == '\"') {
                os << "\\\"";
            } else if (c == '\\') {
                os << "\\\\";
            } else {
                os << c;
            }
        }
    };

    util::converter cvt;

    os << "{\n";
    os << "  \"version\": \"1.0\",\n";
    os << "  \"added_tokens\": [],\n";
    os << "  \"model\": {\n";
    os << "    \"vocab\": {\n";
    for (const auto& i : vocab) {
        os << "      \"";

        auto encoded = cvt.encode(i);
        raw(encoded);

        os << "\": " << vocab_index.at(i);
        if (i != vocab.back()) {
            os << ",";
        }
        os << "\n";
    }
    os << "    },\n";
    os << "    \"merges\": [\n";
    for (const auto& p : merge_pairs) {
        auto l = cvt.encode(vocab[p.left]);
        auto r = cvt.encode(vocab[p.right]);
        os << "      [ \"";
        raw(l);
        os << "\", \"";
        raw(r);
        os << "\" ]";
        if (p != merge_pairs.back()) {
            os << ",";
        }
        os << "\n";
    }
    os << "    ]\n";
    os << "  }\n";
    os << "}\n";
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
