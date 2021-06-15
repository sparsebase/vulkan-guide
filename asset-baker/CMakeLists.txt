include(CheckLanguage)
check_language(CUDA)
enable_language(CUDA)
find_package(CUDAToolkit REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(DEVICE_LIBS ${DEVICE_LIBS} cudadevrt CUDA::cudart CUDA::cublas)

find_package(assimp REQUIRED)
# Add source to this project's executable.
add_executable (baker "asset_main.cpp")

set_property(TARGET baker PROPERTY VS_DEBUGGER_WORKING_DIRECTORY "$<TARGET_FILE_DIR:extra>")
set_target_properties(baker PROPERTIES LINK_DIRECTORIES ${CUDAToolkit_LIBRARY_DIR}) 

target_include_directories(baker PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}" )

target_link_libraries(baker PUBLIC tinyobjloader stb_image json lz4 assetlib tinyGLTF nvtt glm assimp::assimp ${DEVICE_LIBS})

