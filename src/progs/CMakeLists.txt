# one executable per program, linked against vine + PHAST

add_executable(vine vine.c)
add_executable(evalTrees evalTrees.c)
add_executable(crisprLnl crisprLnl.c)

foreach(tgt vine evalTrees crisprLnl)
    target_link_libraries(${tgt} PRIVATE vine_core)
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
  crisprLnl
  RUNTIME DESTINATION bin
)
