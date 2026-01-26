# one executable per program, linked against vine + PHAST

find_package(OpenMP QUIET)

add_executable(vine vine.c)
add_executable(evalTrees evalTrees.c)

foreach(tgt vine evalTrees)
    target_link_libraries(${tgt} PRIVATE vine_core)
    if(TARGET OpenMP::OpenMP_C) 
        target_link_libraries(${tgt} PRIVATE OpenMP::OpenMP_C)
    endif()
endforeach()

# Read version file and strip whitespace
file(READ "${CMAKE_SOURCE_DIR}/version" VINE_VERSION_RAW)
string(STRIP "${VINE_VERSION_RAW}" VINE_VERSION)
message(STATUS "VINE_VERSION='${VINE_VERSION}'")

# add to vine compile target
target_compile_definitions(vine PRIVATE VINE_VERSION="${VINE_VERSION}")


# Install all progs into bin/
install(TARGETS
  vine
  evalTrees
  RUNTIME DESTINATION bin
)
