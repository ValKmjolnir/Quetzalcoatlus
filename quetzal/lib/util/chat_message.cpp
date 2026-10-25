#include "util/chat_message.hpp"
#include "util/fs.hpp"
#include "util/timestamp.hpp"

namespace quetzal::util {

const char* log_manager::get_session_dir() {
    return util::is_windows()
        ? ".quetzal\\sessions"
        : ".quetzal/sessions";
}

void log_manager::create_quetzal_dir(bool debug) {
    if (!util::dir_exists(".quetzal")) {
        util::create_dir(".quetzal");
        if (debug) {
            std::cout << "[Debug] log_manager: created <.quetzal> directory\n";
        }
    } else if (debug) {
        std::cout << "[Debug] log_manager: <.quetzal> directory exists\n";
    }

    const char* session_dir = get_session_dir();
    if (!util::dir_exists(session_dir)) {
        util::create_dir(session_dir);
        if (debug) {
            std::cout << "[Debug] log_manager: created <.quetzal/sessions> directory\n";
        }
    } else if (debug) {
        std::cout << "[Debug] log_manager: <.quetzal/sessions> directory exists\n";
    }
}

std::string log_manager::generate_session_file_name() {
    std::string session_file_name = get_session_dir();
    session_file_name += util::is_windows() ? "\\" : "/";
    session_file_name += "session-";
    session_file_name += format_time(std::time(nullptr));
    session_file_name += "-cache.jsonl";
    return session_file_name;
}

log_manager::log_manager(bool debug)
    : session_file(generate_session_file_name()) {
    create_quetzal_dir(debug);
}

void log_manager::log(const std::string& message) {
    std::ofstream os(session_file, std::ios::app);
    os << message << std::endl;
}

message_manager::message_manager(const bbpe::tokenizer& tok)
    : tokenizer(tok) {
    prompt_end = "<|im_start|>assistant\n";
    prompt_end_token_count = tokenizer.encode(prompt_end).size();
}

void message_manager::push(const std::string& role,
                           const std::string& content) {
    auto real_content = "<|im_start|>" + role + "\n" + content + "<|im_end|>\n";
    messages.emplace_back(role, content, tokenizer.encode(real_content).size());
}

void message_manager::clear() {
    if (messages.size() > 1 && messages[0].get_role() == "system") {
        auto system_prompt = messages[0];
        messages.clear();
        messages.push_back(system_prompt);
    } else {
        messages.clear();
    }
}

bool message_manager::empty() const {
    if (messages.empty()) {
        return true;
    }
    if (messages.size() == 1 && messages[0].get_role() == "system") {
        return true;
    }
    return false;
}

std::size_t message_manager::get_token_count() const {
    std::size_t token_count = 0;
    for (const auto& message : messages) {
        token_count += message.token_count();
    }
    return token_count;
}

std::string message_manager::build(std::size_t max_token_count,
                                   bool debug_mode) {
    std::size_t token_count = prompt_end_token_count + get_token_count();
    bool cleared = false;
    while (token_count >= max_token_count && messages.size() > 1) {
        cleared = true;
        if (messages[0].get_role() == "system") {
            messages.erase(messages.begin() + 1);
        } else {
            messages.erase(messages.begin());
        }
        token_count = prompt_end_token_count + get_token_count();
    }
    if (cleared && debug_mode) {
        std::cout << "\n[Debug] message_manager: prompt list cleared, "
                    << "token count: " << token_count << std::endl;
    }
    std::string prompt;
    for (const auto& message : messages) {
        prompt += message.to_prompt_segment();
    }
    return prompt + prompt_end;
}

std::string message_manager::generate_log() const {
    std::string log = "{\"timestamp\":\"" + format_time(std::time(nullptr)) + "\",";
    log += "\"messages\":[";
    for (const auto& message : messages) {
        log += "{\"role\": \"" + message.get_role() + "\", ";
        log += "\"content\": \"";
        for (const auto& c : message.get_content()) {
            if (c == '"') {
                log += "\\\"";
            } else if (c == '\\') {
                log += "\\\\";
            } else if (c == '\n') {
                log += "\\n";
            } else if (c == '\r') {
                log += "\\r";
            } else if (c == '\t') {
                log += "\\t";
            } else {
                log += c;
            }
        }
        log += "\"},";
    }
    if (log.back() == ',') {
        log.pop_back();
    }
    log += "]}";
    return log;
}

}
