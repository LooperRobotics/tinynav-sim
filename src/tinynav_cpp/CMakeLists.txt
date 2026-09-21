cmake_minimum_required(VERSION 3.16)
project(tinynav_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE RelWithDebInfo)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(message_filters REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(Ceres REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_library(NVINFER_LIB nvinfer REQUIRED)
find_library(NVONNXPARSER_LIB nvonnxparser)

# ---------------------------------------------------------------- GTSAM (optional)
# The image ships a full gtsam 4.3a1 tree (/3rdparty/gtsam: sources + build with
# libgtsam + libgtsam_unstable + the python bindings perception fixtures use).
# When present, perception's factor-graph refinement links it; machines without
# the tree build fine and the node degrades to the chained-PnP v1 pipeline
# (same rule as the TRT engines).
set(TINYNAV_GTSAM_ROOT "/3rdparty/gtsam" CACHE PATH "gtsam source+build tree")
if(EXISTS "${TINYNAV_GTSAM_ROOT}/build/gtsam/libgtsam.so")
  file(GLOB GTSAM_SOURCES CONFIGURE_DEPENDS src/gtsam/*.cpp)
  add_library(tinynav_gtsam STATIC ${GTSAM_SOURCES})
  target_include_directories(tinynav_gtsam PUBLIC include
    ${TINYNAV_GTSAM_ROOT} ${TINYNAV_GTSAM_ROOT}/build)
  # the image's gtsam is built with the TBB allocator (CMakeCache)
  target_link_libraries(tinynav_gtsam PUBLIC tinynav_core tbb
    ${TINYNAV_GTSAM_ROOT}/build/gtsam/libgtsam.so
    ${TINYNAV_GTSAM_ROOT}/build/gtsam_unstable/libgtsam_unstable.so)
  message(STATUS "tinynav: GTSAM factor-graph refinement enabled (${TINYNAV_GTSAM_ROOT})")
else()
  message(STATUS "tinynav: no gtsam build tree at ${TINYNAV_GTSAM_ROOT} — perception stays v1")
endif()

# ---------------------------------------------------------------- core libs
# Pure C++ (no ROS): math / planning / mapping / kernels ported from
# reference/tinynav. GLOB so new translation units need no CMake edit.
file(GLOB_RECURSE CORE_SOURCES CONFIGURE_DEPENDS
  src/core/*.cpp src/kernels/*.cpp src/planning/*.cpp src/mapping/*.cpp)
add_library(tinynav_core STATIC ${CORE_SOURCES})
target_include_directories(tinynav_core PUBLIC include ${EIGEN3_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})
target_link_libraries(tinynav_core PUBLIC ${OpenCV_LIBS} Eigen3::Eigen yaml-cpp ${CERES_LIBRARIES})

# ---------------------------------------------------------------- TensorRT
file(GLOB TRT_SOURCES CONFIGURE_DEPENDS src/trt/*.cpp)
add_library(tinynav_trt STATIC ${TRT_SOURCES})
target_include_directories(tinynav_trt PUBLIC include ${CUDAToolkit_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})
target_link_libraries(tinynav_trt PUBLIC tinynav_core ${NVINFER_LIB} CUDA::cudart ${OpenCV_LIBS})

# ---------------------------------------------------------------- components
set(COMPONENT_TARGETS)
foreach(comp planning mapping perception imu_propagator)
  add_library(${comp}_component SHARED src/components/${comp}_component.cpp)
  target_include_directories(${comp}_component PUBLIC include)
  target_link_libraries(${comp}_component tinynav_core tinynav_trt)
  if(TARGET tinynav_gtsam AND comp STREQUAL perception)
    target_link_libraries(${comp}_component tinynav_gtsam)
    # rpath must live on the final shared object: a static lib's INSTALL_RPATH
    # never reaches the consumer's libgtsam.so.4 lookup.
    set_property(TARGET perception_component APPEND PROPERTY INSTALL_RPATH
      "${TINYNAV_GTSAM_ROOT}/build/gtsam;${TINYNAV_GTSAM_ROOT}/build/gtsam_unstable")
  endif()
  ament_target_dependencies(${comp}_component
    rclcpp rclcpp_components sensor_msgs nav_msgs geometry_msgs std_msgs
    tf2 tf2_ros tf2_geometry_msgs cv_bridge message_filters)
  list(APPEND COMPONENT_TARGETS ${comp}_component)
endforeach()

rclcpp_components_register_nodes(planning_component "tinynav::PlanningComponent")
rclcpp_components_register_nodes(mapping_component "tinynav::MappingComponent")
rclcpp_components_register_nodes(perception_component "tinynav::PerceptionComponent")
rclcpp_components_register_nodes(imu_propagator_component "tinynav::ImuPropagatorComponent")

# Single-process host: all components in one executor, intra-process comms.
# logging_setup.cpp is host-only infra (rcutils hook, per-day files) — pure
# libraries stay ROS-free per the hard rules.
add_executable(tinynav_node src/main.cpp src/components/logging_setup.cpp)
target_link_libraries(tinynav_node ${COMPONENT_TARGETS})
ament_target_dependencies(tinynav_node rclcpp)

# Split-site link relay (port of the fleet's looper_bridge, relay mode):
# subscribes the camera-box-namespace products once, republishes the
# canonical /slam/* names locally. Separate process = separate lifecycle
# from the stack (the link must survive a stack crash).
add_executable(tinynav_bridge src/bridge_node.cpp src/components/logging_setup.cpp)
target_include_directories(tinynav_bridge PRIVATE include)
target_link_libraries(tinynav_bridge yaml-cpp)
ament_target_dependencies(tinynav_bridge rclcpp sensor_msgs nav_msgs)

# ---------------------------------------------------------------- install
install(TARGETS tinynav_core tinynav_trt ${COMPONENT_TARGETS} tinynav_node tinynav_bridge
  ARCHIVE DESTINATION lib LIBRARY DESTINATION lib RUNTIME DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY include/ DESTINATION include)
install(DIRECTORY launch config DESTINATION share/${PROJECT_NAME})

# ---------------------------------------------------------------- tests
if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  find_package(rcutils REQUIRED)
  file(GLOB TEST_SOURCES CONFIGURE_DEPENDS test/*.cpp)
  if(TEST_SOURCES)
    ament_add_gtest(tinynav_core_test ${TEST_SOURCES})
    target_sources(tinynav_core_test PRIVATE src/components/logging_setup.cpp)
    ament_target_dependencies(tinynav_core_test rcutils)
    target_link_libraries(tinynav_core_test tinynav_core tinynav_trt)
    if(TARGET tinynav_gtsam)
      target_link_libraries(tinynav_core_test tinynav_gtsam)
    endif()
    target_include_directories(tinynav_core_test PRIVATE include)
  endif()
endif()

ament_package()
