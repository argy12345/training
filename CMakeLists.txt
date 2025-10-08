cmake_minimum_required(VERSION 4.0)
project(training_models)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(PROJECT_EXECUTABLE_NAME "training_models")
add_executable(${PROJECT_EXECUTABLE_NAME} main.cpp)


find_package(dlib REQUIRED)
target_link_libraries(${PROJECT_EXECUTABLE_NAME} PRIVATE dlib::dlib)

set(LIGHTGBM_INCLUDE_DIR "/home/argy/libs/LightGBM/include")
set(LIGHTGBM_LIB_FILE "/home/argy/libs/LightGBM/lib_lightgbm.so")

# Include headers (visibility keyword required)
target_include_directories(${PROJECT_EXECUTABLE_NAME} PRIVATE "${LIGHTGBM_INCLUDE_DIR}")

# Link directly to the compiled library file
target_link_libraries(${PROJECT_EXECUTABLE_NAME} PRIVATE "${LIGHTGBM_LIB_FILE}")

# Optional: threading
find_package(Threads REQUIRED)
target_link_libraries(${PROJECT_EXECUTABLE_NAME} PRIVATE Threads::Threads)
