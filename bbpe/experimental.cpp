#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cstdint>

#include "util/densemap.hpp"

class frequency_collector {
private:
    bbpe::util::densemap<std::string, std::uint32_t> freq;
    const std::vector<std::string> seperators = {
        " ", ",", ".", "?", "!",     // ASCII
        "，", "、", "。", "！", "？" // UTF-8 ZH
    };

private:
    void add_segment(const std::string& seg) {
        if (seg.empty()) {
            return;
        }
        if (freq.contains(seg)) {
            freq[seg] += 1;
        } else {
            freq.insert(seg, 1);
        }
    }

    int find_seperator(const std::string&, int);

public:
    frequency_collector() {
        for (auto& sep : seperators) {
            add_segment(sep);
        }
    }
    void split_line(const std::string&);
    void dump() const;
    std::size_t total_stored_chars() const {
        std::size_t total = 0;
        for (const auto& [seg, fr] : freq) {
            total += seg.size();
        }
        return total;
    }
    std::size_t total_stored_segments() const {
        return freq.size();
    }
};

int frequency_collector::find_seperator(const std::string& line, int start) {
    for (const auto& sep : seperators) {
        if (line.compare(start, sep.size(), sep) == 0) {
            add_segment(sep);
            return sep.size();
        }
    }
    return -1;
}

void frequency_collector::split_line(const std::string& line) {
    std::vector<std::string> segments;

    std::string segment = "";
    for (std::size_t i = 0; i < line.size(); ++i) {
        auto c = line[i];
        auto sep_size = find_seperator(line, i);
        if (sep_size != -1) {
            add_segment(segment);
            segment.clear();
            i += sep_size - 1;
        } else {
            segment.push_back(c);
        }
    }
    add_segment(segment);
}

void frequency_collector::dump() const {
    for (const auto& [seg, fr] : freq) {
        std::cout << seg << " " << fr << "\n";
    }
}

void collect_segment_frequency(const char* filename) {
    std::ifstream in(filename);
    if (!in) {
        std::cerr << "Cannot open " << filename << "\n";
        return;
    }

    frequency_collector collector;

    std::uint64_t line_count = 0;
    std::uint64_t total_chars = 0;
    std::string line;
    while (std::getline(in, line)) {
        line_count += 1;
        total_chars += line.size();
        while (!line.empty() && (line.back() == '\n' || line.back() == '\r')) {
            line.pop_back();
        }
        collector.split_line(line);
        if (line_count % 1000000 == 0) {
            std::cout << "Processed " << line_count << " lines\n";
        }
    }

    collector.dump();
    std::cout << "Total chars: " << total_chars << "\n"
              << "Total stored chars: " << collector.total_stored_chars() << "\n"
              << "Total stored chars ratio: " << collector.total_stored_chars() / (double)total_chars << "\n"
              << "Total segments: " << collector.total_stored_segments() << "\n";
}

int main(int argc, const char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: bbpe <input text>\n";
        return -1;
    }

    collect_segment_frequency(argv[1]);
    return 0;
}
