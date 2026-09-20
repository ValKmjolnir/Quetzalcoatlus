#pragma once

#include "bbpe/tokenizer.hpp"

#include <string>
#include <vector>

namespace quetzal::util {

class chat_message {
private:
    std::string role_;
    std::string content_;
    std::size_t token_count_;

public:
    chat_message(const std::string& role,
                 const std::string& content,
                 std::size_t token_count):
                 role_(role), content_(content), token_count_(token_count) {}

    std::string to_prompt_segment() const {
        return "<|im_start|>" + role_ + "\n" + content_ + "<|im_end|>\n";
    }
    std::size_t token_count() const {
        return token_count_;
    }
};

class message_manager {
private:
    std::string prompt_end;
    std::size_t prompt_end_token_count = 0;
    std::vector<chat_message> messages;
    const bbpe::tokenizer& tokenizer;

public:
    message_manager(const bbpe::tokenizer& tok): tokenizer(tok) {
        prompt_end = "<|im_start|>assistant\n";
        prompt_end_token_count = tokenizer.encode(prompt_end).size();
    }
    void push(const std::string& role, const std::string& content) {
        messages.emplace_back(role, content, tokenizer.encode(content).size());
    }
    std::string build(std::size_t max_token_count) {
        std::size_t token_count = prompt_end_token_count;
        for (const auto& message : messages) {
            token_count += message.token_count();
        }
        while (token_count >= max_token_count && !messages.empty()) {
            messages.erase(messages.begin());
            token_count = prompt_end_token_count;
            for (const auto& message : messages) {
                token_count += message.token_count();
            }
        }
        std::string prompt;
        for (const auto& message : messages) {
            prompt += message.to_prompt_segment();
        }
        return prompt + prompt_end;
    }
};

}
