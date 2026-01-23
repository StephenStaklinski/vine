# one executable per program, linked against vine + PHAST

find_package(OpenMP QUIET)

add_executable(vine vine.c)
add_executable(evalTrees evalTrees.c)

foreach(tgt vine evalTrees)
    target_link_libraries(${tgt} PRIVATE vine_core OpenMP::OpenMP_C)
endforeach()

# Install all progs into bin/
install(TARGETS
  vine
  evalTrees
  RUNTIME DESTINATION bin
)
