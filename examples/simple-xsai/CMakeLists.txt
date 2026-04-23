set(TARGET llama-simple-xsai)
add_executable(${TARGET} simple-xsai.cpp)
install(TARGETS ${TARGET} RUNTIME)
target_link_libraries(${TARGET} PRIVATE llama ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)
if (GGML_RV_AME)
    target_compile_definitions(${TARGET} PRIVATE GGML_USE_RV_AME)
endif()
if (GGML_XSAI_ALLOC)
    target_compile_definitions(${TARGET} PRIVATE GGML_XSAI_ALLOC)
    target_include_directories(${TARGET} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../../ggml/include
        ${CMAKE_CURRENT_SOURCE_DIR}/../../ggml/src/ggml-cpu
    )
endif()
