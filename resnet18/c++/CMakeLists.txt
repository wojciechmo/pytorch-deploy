cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(resnet_pred)

find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)

add_executable(app main.cpp)
target_link_libraries(app "${TORCH_LIBRARIES}")
target_link_libraries(app ${OpenCV_LIBS})
set_property(TARGET app PROPERTY CXX_STANDARD 11)


