cmake_minimum_required(VERSION 3.17)
project(dgemm)

set(CMAKE_C_FLAGS "-std=c99")
set(CMAKE_C_COMPILER=icc)


set(SOURCE_FILES ./lib/dgemm-blocked.c)
        #./lib/dgemm-naive.c ./lib/dgemm-tuned-a.c ./lib/dgemm-tuned-b.c
        #./lib/search_optimal.c)

add_library(dgemm_lib SHARED ${SOURCE_FILES})

add_executable(dgemm benchmark.c)

TARGET_LINK_LIBRARIES(dgemm dgemm_lib)
