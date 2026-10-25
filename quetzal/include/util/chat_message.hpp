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
    const std::string& get_role() const { return role_; }
    const std::string& get_content() const { return content_; }
    std::string to_prompt_segment() const {
        return "<|im_start|>" + role_ + "\n" + content_ + "<|im_end|>\n";
    }
    std::size_t token_count() const {
        return token_count_;
    }
};

class log_manager {
private:
    std::string session_file;

private:
    const char* get_session_dir();
    void create_quetzal_dir(bool debug);
    std::string generate_session_file_name();

public:
    log_manager(bool debug = false);
    void log(const std::string& message);
};

class message_manager {
private:
    std::string prompt_end;
    std::size_t prompt_end_token_count = 0;
    std::vector<chat_message> messages;
    const bbpe::tokenizer& tokenizer;

public:
    message_manager(const bbpe::tokenizer& tok);
    void push(const std::string& role, const std::string& content);
    void clear();
    std::size_t get_token_count() const;
    std::string build(std::size_t max_token_count, bool debug_mode = false);
    std::string generate_log() const;
};

}
