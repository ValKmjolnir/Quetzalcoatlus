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
add_compile_options(-I ${CMAKE_SOURCE_DIR}/include)
if (APPLE)
    add_compile_options(-mmacosx-version-min=15)
endif()

# build tokenizer used object
add_library(bbpe-lib STATIC
            ${CMAKE_SOURCE_DIR}/lib/util/byte_to_unicode.cpp
            ${CMAKE_SOURCE_DIR}/lib/util/utf8.cpp
            ${CMAKE_SOURCE_DIR}/lib/bbpe/bin_reader.cpp
            ${CMAKE_SOURCE_DIR}/lib/bbpe/tokenizer.cpp)
target_include_directories(bbpe-lib PRIVATE ${CMAKE_SOURCE_DIR}/include)

# find openmp
find_package(OpenMP)

# build tokenizer
add_executable(tokenizer ${CMAKE_SOURCE_DIR}/tokenizer.cpp)
target_link_libraries(tokenizer bbpe-lib)

add_library(quetzal STATIC
            ${CMAKE_SOURCE_DIR}/lib/gpt/attention.cpp
            ${CMAKE_SOURCE_DIR}/lib/gpt/transformer.cpp
            ${CMAKE_SOURCE_DIR}/lib/gpt/gpt2.cpp
            ${CMAKE_SOURCE_DIR}/lib/tensor/weights_manager.cpp
            ${CMAKE_SOURCE_DIR}/lib/tensor/tensor.cpp)
target_include_directories(quetzal PUBLIC ${CMAKE_SOURCE_DIR}/include)

if (OpenMP_CXX_FOUND)
    message(STATUS "OpenMP found")
    target_link_libraries(quetzal PUBLIC OpenMP::OpenMP_CXX)
    target_compile_definitions(quetzal PUBLIC QGPT_USE_OPENMP)
else()
    message(STATUS "OpenMP not found")
endif()

# build deploy
add_executable(deploy ${CMAKE_SOURCE_DIR}/deploy.cpp)
target_link_libraries(deploy quetzal bbpe-lib)

# build test
add_executable(tensor_test ${CMAKE_SOURCE_DIR}/test/tensor_test.cpp)
add_executable(load_weights ${CMAKE_SOURCE_DIR}/test/load_weights.cpp)
add_executable(load_tokenizer ${CMAKE_SOURCE_DIR}/test/load_tokenizer.cpp)
target_link_libraries(tensor_test quetzal)
target_link_libraries(load_weights quetzal)
target_link_libraries(load_tokenizer quetzal bbpe-lib)
