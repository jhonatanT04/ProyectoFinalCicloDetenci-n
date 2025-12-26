cmake_minimum_required(VERSION 3.16)
project(AppIMA LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)
find_package(ITK REQUIRED)
include(${ITK_USE_FILE})

find_package(CURL REQUIRED)
find_package(nlohmann_json REQUIRED)

find_path(WEBSOCKETPP_INCLUDE_DIR
    websocketpp/config/asio_no_tls.hpp
)


add_executable(main
    main.cpp
)

target_include_directories(main PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    ${WEBSOCKETPP_INCLUDE_DIR}
)

target_compile_definitions(main PRIVATE
    ASIO_STANDALONE
)

target_link_libraries(main
    ${OpenCV_LIBS}
    ${ITK_LIBRARIES}
    CURL::libcurl
    nlohmann_json::nlohmann_json
    pthread
)
