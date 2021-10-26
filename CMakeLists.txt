cmake_minimum_required(VERSION 3.12)

project(svmpc)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_CXX_STANDARD 17)

# Set output directories for libraries and executables
set(BASE_DIR ${CMAKE_SOURCE_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${BASE_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${BASE_DIR}/bin)

# Installation locations
set(INSTALL_LIB_DIR ${CMAKE_INSTALL_PREFIX}/lib)
set(INSTALL_INCLUDE_DIR ${CMAKE_INSTALL_PREFIX}/include/${PROJECT_NAME})
set(INSTALL_CMAKE_DIR ${CMAKE_INSTALL_PREFIX}/lib/cmake/${PROJECT_NAME})

if (NOT CMAKE_BUILD_TYPE)
    # default to an optimised build
    set(CMAKE_BUILD_TYPE Release)
endif (NOT CMAKE_BUILD_TYPE)

message(STATUS "Build configuration: " ${CMAKE_BUILD_TYPE})

# +-----------------------------------------------------------------------------
# | Build configuration options
# +-----------------------------------------------------------------------------
#option( BUILD_CTAGS "Build ctag file?" FALSE )
#option( BUILD_APPLICATIONS "Build applications?" FALSE )

# +-----------------------------------------------------------------------------
# | Library search and setup
# +-----------------------------------------------------------------------------

# Build shared libraries by default
set(BUILD_SHARED_LIBS True)

find_package(Eigen3)


# Use local include and library directories
include_directories(
        #${ZMQ_INCLUDE_DIRS}
        #${Protobuf_INCLUDE_DIR}
        ${EIGEN3_INCLUDE_DIR}
        ${PROJECT_SOURCE_DIR}
        src/
        src/external
)

link_directories(${BASE_DIR}/lib)

# +-----------------------------------------------------------------------------
# | Custom targets and macros
# +-----------------------------------------------------------------------------

if (BUILD_CTAGS)
    # Custom target that builds ctags for the source code
    add_custom_target(
            generate_ctags ALL
            ctags -R --c++-kinds=+p --fields=+iaS --extra=+q .
            WORKING_DIRECTORY ${BASE_DIR}/src
            COMMENT "Regenerating ctags file"
    )
endif ()

# +-----------------------------------------------------------------------------
# | Compiler settings
# +-----------------------------------------------------------------------------
# Compiler specifiic settings to enable C++14 support
#if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
#    set(CMAKE_CXX_FLAGS "-std=c++14 -Wall -pthread")
#elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
#    set(CMAKE_CXX_FLAGS "-std=c++14 -Wall -Wno-int-in-bool-context -pthread")
#endif ()

option(OPTIMIZE_FOR_NATIVE "Build with -march=native" ON)


if (OPTIMIZE_FOR_NATIVE)

  include(CheckCXXCompilerFlag)
  CHECK_CXX_COMPILER_FLAG("-march=native" COMPILER_SUPPORTS_MARCH_NATIVE)
  if(COMPILER_SUPPORTS_MARCH_NATIVE)
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
  endif()

endif()


# +-----------------------------------------------------------------------------
# | Compile code
# +-----------------------------------------------------------------------------


add_subdirectory(src)

#
## python scripts
#configure_file(
#        ${CMAKE_CURRENT_SOURCE_DIR}/scripts/cpp2py_plotly.py
#        ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/cpp2py_plotly.py
#)
#
## demo
#if (CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
#
#    # only build demo if this is the root project
#    add_executable(pub "publisher_example.cpp")
#    target_link_libraries(pub cpp2py_plotly ${link_eigen})
#
#    find_package(Eigen3)
#    if (TARGET Eigen3::Eigen)
#        # Use the imported target
#        set(link_eigen Eigen3::Eigen)
#        add_definitions(-DWITH_EIGEN)
#
#        target_include_directories(pub PRIVATE ${EIGEN3_INCLUDE_DIR})
#    endif (TARGET Eigen3::Eigen)
#endif ()

# +-----------------------------------------------------------------------------
# | Installation setup
# +-----------------------------------------------------------------------------

# Export the package for use from the build tree. This
# registers the build tree with a global CMake registry.
export(PACKAGE ${PROJECT_NAME})

# Determine the relative path between CMake configuration file installation
# location and header files installation directory
file(RELATIVE_PATH REL_INCLUDE_DIR ${INSTALL_CMAKE_DIR} ${INSTALL_INCLUDE_DIR})

# Create the ${ProjectName}Config.cmake file for the build tree.
set(CONF_INCLUDE_DIRS ${PROJECT_SOURCE_DIR} ${PROJECT_BINARY_DIR})
configure_file(
        ${PROJECT_NAME}Config.cmake.in
        "${PROJECT_BINARY_DIR}/${PROJECT_NAME}Config.cmake" @ONLY
)

# Create the ${PROJECT_NAME}Config.cmake file for the install tree.
# CONF_INCLUDE_DIRS is a variable that the generated
# ${PROJECT_NAME}Config.cmake file uses by performing variable substitution on.
# This is the reason that ${*_CMAKE_DIR} is used even though it is not defined
# here.
set(CONF_INCLUDE_DIRS ${CMAKE_INSTALL_PREFIX}/include)
configure_file(
        ${PROJECT_NAME}Config.cmake.in
        "${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/${PROJECT_NAME}Config.cmake" @ONLY
)

# Install the ${PROJECT_NAME}Config.cmake file
install(
        FILES "${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/${PROJECT_NAME}Config.cmake"
        DESTINATION "${INSTALL_CMAKE_DIR}"
        COMPONENT dev
)

# Install the export set for use with the install tree
install(
        EXPORT ${PROJECT_NAME}Targets
        DESTINATION "${INSTALL_CMAKE_DIR}"
        COMPONENT dev
)

