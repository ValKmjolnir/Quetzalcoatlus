#pragma once

#ifndef _MSC_VER
#include <unistd.h>
#include <dirent.h>
#else
#define _CRT_SECURE_NO_DEPRECATE 1
#define _CRT_NONSTDC_NO_DEPRECATE 1
#include <io.h>
#include <direct.h>
#endif

#include <sys/stat.h>
#include <string>

namespace quetzal::util {

bool is_windows();
bool dir_exists(const char* path);
bool create_dir(const char* path);
bool dir_exists(const std::string& path);
bool create_dir(const std::string& path);

}
