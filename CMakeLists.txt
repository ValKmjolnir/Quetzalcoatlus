cmake_minimum_required(VERSION 3.10)

project(bbpe VERSION 0.1)

message("CMAKE_HOST_SYSTEM_NAME: ${CMAKE_HOST_SYSTEM_NAME}")

if (MSVC)
    set(CMAKE_CXX_STANDARD 20)
else()
    set(CMAKE_CXX_STANDARD 17)
endif()
set(CMAKE_CXX_STANDARD_REQUIRED True)

# set compile option
if (NOT MSVC)
    add_compile_options(-fPIC -Wshadow -Wall)
else()
    add_compile_options(/W4)
endif()
add_compile_options(-I ${CMAKE_SOURCE_DIR})
if (APPLE)
    add_compile_options(-mmacosx-version-min=10.15)
endif()

# build colgm used object
set(BBPE
    ${CMAKE_SOURCE_DIR}/bbpe/util/byte_to_unicode.cpp
    ${CMAKE_SOURCE_DIR}/bbpe/util/utf8.cpp
    ${CMAKE_SOURCE_DIR}/bbpe/bbpe.cpp)

add_library(bbpe-lib STATIC ${BBPE})
target_include_directories(bbpe-lib PRIVATE ${CMAKE_SOURCE_DIR}/bbpe)

# build colgm
add_executable(bbpe ${CMAKE_SOURCE_DIR}/bbpe/main.cpp)
target_link_libraries(bbpe bbpe-lib)
add_executable(experimental ${CMAKE_SOURCE_DIR}/bbpe/experimental.cpp)
target_link_libraries(experimental bbpe-lib)

# deploy: QGPT weights loader (Python -> C++ bridge)
add_subdirectory(deploy)
