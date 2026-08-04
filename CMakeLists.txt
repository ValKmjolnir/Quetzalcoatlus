cmake_minimum_required(VERSION 3.10)

project(bbpe VERSION 0.1)

message("CMAKE_HOST_SYSTEM_NAME: ${CMAKE_HOST_SYSTEM_NAME}")

# -std=c++17 -Wshadow -Wall
if (MSVC)
    set(CMAKE_CXX_STANDARD 20)
else()
    set(CMAKE_CXX_STANDARD 17)
endif()
set(CMAKE_CXX_STANDARD_REQUIRED True)

# set compile option
if (NOT MSVC)
    add_compile_options(-fPIC)
endif()
add_compile_options(-I ${CMAKE_SOURCE_DIR})
if (APPLE)
    add_compile_options(-mmacosx-version-min=10.15)
endif()

# build colgm used object
set(BBPE
    ${CMAKE_SOURCE_DIR}/src/util/utf8.cpp
    ${CMAKE_SOURCE_DIR}/src/bbpe.cpp)

add_library(bbpe-lib STATIC ${BBPE})
target_include_directories(bbpe-lib PRIVATE ${CMAKE_SOURCE_DIR}/src)

# build colgm
add_executable(bbpe ${CMAKE_SOURCE_DIR}/src/main.cpp)
target_link_libraries(bbpe bbpe-lib)
