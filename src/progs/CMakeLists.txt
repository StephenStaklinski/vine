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

# Install all progs into bin/
install(TARGETS
  vine
  evalTrees
  RUNTIME DESTINATION bin
)
