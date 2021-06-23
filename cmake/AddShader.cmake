function(ADD_SHADER TARGET SHADER_FILES)
    find_program(GLSLC glslc)
    set(current_output_dir ${CMAKE_BINARY_DIR}/shaders)
    file(MAKE_DIRECTORY ${current_output_dir})
    foreach(_file ${SHADER_FILES})
        get_filename_component(current_filename ${_file} NAME)
        set(current_output_path ${current_output_dir}/${current_filename}.spv)
        get_filename_component(shader_abs_path ${_file} ABSOLUTE)
        
        # Add a custom command to compile GLSL to SPIR-V.
        add_custom_command(
           OUTPUT ${current_output_path}
           COMMAND ${GLSLC} -o ${current_output_path} ${shader_abs_path}
           DEPENDS ${shader_abs_path}
           IMPLICIT_DEPENDS CXX ${shader_abs_path}
           VERBATIM
        )
        # Make sure our build depends on this output.
        set_source_files_properties(${current_output_path} PROPERTIES GENERATED TRUE)
        target_sources(${TARGET} PRIVATE ${current_output_path})  
    endforeach()
endfunction(ADD_SHADER)