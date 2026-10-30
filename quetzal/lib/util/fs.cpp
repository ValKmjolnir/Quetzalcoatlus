#include "util/fs.hpp"

namespace quetzal::util {

bool is_windows() {
#ifdef _WIN32
    return true;
#else
    return false;
#endif
}

bool dir_exists(const char* path) {
#ifdef _MSC_VER
    struct _stat st;
    if (_stat(path, &st) == 0) {
        return (st.st_mode & S_IFDIR) != 0;
    }
    return false;
#else
    struct stat st;
    if (stat(path, &st) == 0) {
        return S_ISDIR(st.st_mode);
    }
    return false;
#endif
}

bool create_dir(const char* path) {
#ifdef _WIN32
    return _mkdir(path) == 0;
#else
    return mkdir(path, 0755) == 0;
#endif
}

bool dir_exists(const std::string& path) {
    return dir_exists(path.c_str());
}

bool create_dir(const std::string& path) {
    return create_dir(path.c_str());
}

}
