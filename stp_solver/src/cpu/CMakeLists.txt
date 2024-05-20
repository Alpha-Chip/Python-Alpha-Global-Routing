# Add source files
set(CPU_FILES disjoin_set.hpp mst.hpp)

# Include source files
target_sources(${PROJECT_NAME} PRIVATE ${CPU_FILES}) # Main project

if(GTest_Found)
    target_sources(${PROJECT_NAME}_lib PUBLIC ${CPU_FILES}) # Library
endif()