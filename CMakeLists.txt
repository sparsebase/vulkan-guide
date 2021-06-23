# CMakeList.txt : CMake project for vulkan_guide, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.8)

project ("vulkan_guide")


find_package(Vulkan REQUIRED)

if(MSVC)
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MT")
    set(CMAKE_C_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MT")
endif(MSVC)

add_subdirectory(third_party)

set (CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/bin")
 
add_subdirectory(assetlib)
add_subdirectory(asset-baker)
add_subdirectory(extra-engine)
if (${CMAKE_HOST_SYSTEM_PROCESSOR} STREQUAL "AMD64")
  set(GLSL_VALIDATOR "$ENV{VULKAN_SDK}/Bin/glslangValidator.exe")
else()
  set(GLSL_VALIDATOR "$ENV{VULKAN_SDK}/Bin32/glslangValidator.exe")
endif()  
message(STATUS "GLSL command: ${GLSL_VALIDATOR}")

file(GLOB_RECURSE GLSL_SOURCE_FILES
    "${PROJECT_SOURCE_DIR}/shaders/*.frag"  
    "${PROJECT_SOURCE_DIR}/shaders/*.vert"
    "${PROJECT_SOURCE_DIR}/shaders/*.comp"
    )
set(shader_output_path "${PROJECT_BINARY_DIR}/shaders")
file(MAKE_DIRECTORY ${shader_output_path})

foreach(GLSL ${GLSL_SOURCE_FILES})
  message(STATUS "BUILDING SHADER")
  get_filename_component(FILE_NAME ${GLSL} NAME)
  set(SPIRV "${shader_output_path}/${FILE_NAME}.spv")  
  message(STATUS "${GLSL} : To SPIRV : ${SPIRV}")
  add_custom_command(
    OUTPUT ${SPIRV}
    COMMAND ${GLSL_VALIDATOR} -V ${GLSL} -o ${SPIRV} 
    DEPENDS ${GLSL}
    VERBATIM
   )
  list(APPEND SPIRV_BINARY_FILES ${SPIRV})
endforeach(GLSL)

add_custom_target(
    run ALL
    DEPENDS ${SPIRV_BINARY_FILES}
    )
