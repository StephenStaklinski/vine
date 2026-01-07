cmake_minimum_required(VERSION 3.16)

project(vine
  VERSION 0.2.0
  LANGUAGES C)

# Use C99 
set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)

# allows use of upper case PHAST_ROOT
if(POLICY CMP0144)
  cmake_policy(SET CMP0144 NEW)
endif()

# ---- PHAST dependency --------------------------------------------------
# Users/packagers provide -DPHAST_ROOT=/path/to/prefix (the install prefix)
set(PHAST_ROOT "" CACHE PATH "PHAST install prefix (contains lib/cmake/phast)")

if (NOT PHAST_ROOT)
  message(FATAL_ERROR "PHAST_ROOT is not set. Configure with -DPHAST_ROOT=/path/to/phast-prefix")
endif()

# Tell CMake where to find phastConfig.cmake
list(PREPEND CMAKE_PREFIX_PATH "${PHAST_ROOT}")

find_package(phast CONFIG REQUIRED)

# ---- Project layout ----------------------------------------------------

# public headers 
set(VINE_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/include")

add_subdirectory(src/lib)
add_subdirectory(src/progs)

# ---- Installation (for Homebrew, etc.) ---------------------------------
# Install public headers
install(DIRECTORY "${VINE_INCLUDE_DIR}/"
        DESTINATION include
        FILES_MATCHING PATTERN "*.h")

