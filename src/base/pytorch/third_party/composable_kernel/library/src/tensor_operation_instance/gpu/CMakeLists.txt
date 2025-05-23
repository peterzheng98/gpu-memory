function(add_instance_library INSTANCE_NAME)
    message("adding instance ${INSTANCE_NAME}")
    set(result 1)
    if(DEFINED DTYPES)
        foreach(source IN LISTS ARGN)
            set(test 0)
            foreach(type IN LISTS DTYPES)
                if(type MATCHES "fp16")
                    set(type1 "_f16")
                elseif(type MATCHES "fp32")
                    set(type1 "_f32")
                elseif(type MATCHES "fp8")
                    set(type1 "_f8")
                elseif(type MATCHES "bf16")
                    set(type1 "_b16")
                elseif(type MATCHES "fp64")
                    set(type1 "_f64")
                elseif(type MATCHES "int8")
                    set(type1 "_i8")
                endif()
                #make an exception for reduction kernels
                if("${source}" MATCHES "${type}" OR "${source}" MATCHES "${type1}" OR "${source}" MATCHES "device_reduce_instance" OR ${source} MATCHES "device_image_to_column")
                    #if filename matches any selected type, exit type loop and do no exclude the file from the list
                    set(test 0)
                    break()
                elseif((source MATCHES "fp8" OR source MATCHES "fp32" OR source MATCHES "fp64" OR source MATCHES "bf16" OR source MATCHES "int8" OR source MATCHES "fp16" OR
                    source MATCHES "_f8" OR source MATCHES "_f32" OR source MATCHES "_f64" OR source MATCHES "_i8" OR source MATCHES "_f16" OR source MATCHES "_b16") AND
                    NOT(source MATCHES type OR source MATCHES type1))
                    #if filename contains a type which doesn't match any selected type, mark it for removal
                    set(test 1)
                endif()
            endforeach()
            if(test EQUAL 1)
                message("removing instance ${source} ")
                list(REMOVE_ITEM ARGN "${source}")
            endif()
        endforeach()
    endif()

    set(INST_TARGETS ${SUPPORTED_GPU_TARGETS})

    # Do not build DPP instances if DPP_KERNELS macro is not set
    foreach(source IN LISTS ARGN)
        if(NOT DEFINED DPP_KERNELS AND source MATCHES "_dpp")
            message("removing dpp instance ${source} ")
            list(REMOVE_ITEM ARGN "${source}")
        endif()
    endforeach()
    # Do not build DL instances if DL_KERNELS macro is not set
    foreach(source IN LISTS ARGN)
        if(NOT DEFINED DL_KERNELS AND source MATCHES "_dl")
            message("removing dl instance ${source} ")
            list(REMOVE_ITEM ARGN "${source}")
        endif()
    endforeach()
    # Do not build XDL instances if gfx9 targets are not on the target list
    foreach(source IN LISTS ARGN)
        if(NOT INST_TARGETS MATCHES "gfx9" AND source MATCHES "_xdl")
            message("removing xdl instance ${source} ")
            list(REMOVE_ITEM ARGN "${source}")
        endif()
    endforeach()
    # Do not build WMMA instances if gfx11 targets are not on the target list
    foreach(source IN LISTS ARGN)
	if(NOT INST_TARGETS MATCHES "gfx11" AND NOT INST_TARGETS MATCHES "gfx12" AND source MATCHES "_wmma")
            message("removing wmma instance ${source} ")
            list(REMOVE_ITEM ARGN "${source}")
        endif()
    endforeach()
    # Do not build mha instances if gfx94 or gfx90a targets are not on the target list
    foreach(source IN LISTS ARGN)
	    if(NOT INST_TARGETS MATCHES "gfx94" AND NOT INST_TARGETS MATCHES "gfx90a" AND NOT INST_TARGETS MATCHES "gfx95" AND source MATCHES "mha")
         message("removing mha instance ${source} ")
         list(REMOVE_ITEM ARGN "${source}")
    endif()
    endforeach()
    # Do not build gemm_universal_f8 or gemm_multiply_multiply_f8 for any targets except gfx94
    if(NOT CK_USE_FP8_ON_UNSUPPORTED_ARCH)
    foreach(source IN LISTS ARGN)
    if(NOT INST_TARGETS MATCHES "gfx94" AND NOT INST_TARGETS MATCHES "gfx95" AND source MATCHES "gemm_multiply_multiply_xdl_f8")
         message("removing gemm_multiply_multiply_f8 instance ${source} ")
         list(REMOVE_ITEM ARGN "${source}")
    endif()
    endforeach()
    foreach(source IN LISTS ARGN)
    if(NOT INST_TARGETS MATCHES "gfx94" AND NOT INST_TARGETS MATCHES "gfx95" AND source MATCHES "gemm_xdl_universal" AND source MATCHES "_f8_")
         message("removing gemm_universal_f8 instance ${source} ")
         list(REMOVE_ITEM ARGN "${source}")
    endif()
    endforeach()
    foreach(source IN LISTS ARGN)
    if(NOT INST_TARGETS MATCHES "gfx94" AND NOT INST_TARGETS MATCHES "gfx95" AND source MATCHES "batched_gemm_xdl_universal" AND source MATCHES "_f8_")
         message("removing batched_gemm_universal_f8 instance ${source} ")
         list(REMOVE_ITEM ARGN "${source}")
    endif()
    endforeach()
    foreach(source IN LISTS ARGN)
    if(NOT INST_TARGETS MATCHES "gfx94" AND NOT INST_TARGETS MATCHES "gfx95" AND source MATCHES "gemm_xdl_universal_streamk" AND source MATCHES "_f8_")
         message("removing gemm_universal_streamk_f8 instance ${source} ")
         list(REMOVE_ITEM ARGN "${source}")
    endif()
    endforeach()
    endif()
    #only continue if there are some source files left on the list
    if(ARGN)
        set(INST_OBJ)
        foreach(source IN LISTS ARGN)
            set(INST_TARGETS ${SUPPORTED_GPU_TARGETS})
            if(source MATCHES "_xdl")
                list(REMOVE_ITEM INST_TARGETS gfx900 gfx906 gfx906:xnack- gfx1030 gfx1100 gfx1101 gfx1102 gfx1103 gfx1200 gfx1201 gfx10.3-generic gfx11-generic gfx12-generic)
            elseif(source MATCHES "_wmma")
                list(REMOVE_ITEM INST_TARGETS gfx900 gfx906 gfx906:xnack- gfx908:xnack+ gfx908:xnack- gfx90a:xnack+ gfx90a:xnack- gfx908 gfx90a gfx940 gfx941 gfx942 gfx1030 gfx950)
            elseif(source MATCHES "mha")
                list(REMOVE_ITEM INST_TARGETS gfx900 gfx906 gfx906:xnack- gfx908:xnack- gfx908:xnack+ gfx908 gfx1030 gfx1100 gfx1101 gfx1102 gfx1103 gfx1200 gfx1201 gfx10.3-generic gfx11-generic gfx12-generic)
            endif()
            #only build the fp8 gemm instances for gfx908/90a if the build argument is set
            if(NOT CK_USE_FP8_ON_UNSUPPORTED_ARCH)
                if(source MATCHES "gemm_xdl_universal" AND source MATCHES "f8")
                    list(REMOVE_ITEM INST_TARGETS gfx900 gfx906 gfx906:xnack- gfx908:xnack- gfx908:xnack+ gfx90a:xnack+ gfx90a:xnack- gfx908 gfx90a gfx1030 gfx1100 gfx1101 gfx1102 gfx1103 gfx1200 gfx1201 gfx10.3-generic gfx11-generic gfx12-generic)
                endif()
                if(source MATCHES "gemm_multiply_multiply_f8")
                    list(REMOVE_ITEM INST_TARGETS gfx900 gfx906 gfx906:xnack- gfx908:xnack- gfx908:xnack+ gfx90a:xnack+ gfx90a:xnack- gfx908 gfx90a gfx1030 gfx1100 gfx1101 gfx1102 gfx1103 gfx1200 gfx1201 gfx10.3-generic gfx11-generic gfx12-generic)
                endif()
                if(source MATCHES "bached_gemm_multiply_multiply_f8")
                    list(REMOVE_ITEM INST_TARGETS gfx900 gfx906 gfx906:xnack- gfx908:xnack- gfx908:xnack+ gfx90a:xnack+ gfx90a:xnack- gfx908 gfx90a gfx1030 gfx1100 gfx1101 gfx1102 gfx1103 gfx1200 gfx1201 gfx10.3-generic gfx11-generic gfx12-generic)
                endif()
            endif()
            set(offload_targets)
            foreach(target IN LISTS INST_TARGETS)
                    string(APPEND offload_targets "--offload-arch=${target} ")
            endforeach()
            set_source_files_properties(${source} PROPERTIES COMPILE_FLAGS ${offload_targets})
            list(APPEND INST_OBJ ${source})
        endforeach()
        add_library(${INSTANCE_NAME} OBJECT ${INST_OBJ})

        # Allow comparing floating points directly in order to check sentinel values
        if(${INSTANCE_NAME} STREQUAL "device_mha_instance")
            if(NOT DEFINED FMHA_FWD_FAST_EXP2)
                set(FMHA_FWD_FAST_EXP2 true)
            endif()
            if(FMHA_FWD_FAST_EXP2)
                list(APPEND FMHA_COMPILE_OPTIONS -Wno-undefined-func-template -DCK_TILE_FMHA_FWD_FAST_EXP2=1 -fgpu-flush-denormals-to-zero)
            else()
                list(APPEND FMHA_COMPILE_OPTIONS -Wno-undefined-func-template -DCK_TILE_FMHA_FWD_FAST_EXP2=0)
            endif()
            list(APPEND FMHA_COMPILE_OPTIONS -Wno-float-equal)
            list(APPEND FMHA_COMPILE_OPTIONS -DCK_TILE_FMHA_FWD_SPLITKV_API=1)
            list(APPEND FMHA_COMPILE_OPTIONS -DCK_TILE_FMHA_FWD_APPENDKV_API=1)
            target_compile_options(device_mha_instance PRIVATE ${FMHA_COMPILE_OPTIONS})
        endif()
        
        target_compile_features(${INSTANCE_NAME} PUBLIC)

        # flags to compress the library
        if(NOT WIN32 AND ${hip_VERSION_FLAT} GREATER 600241132)
            #message("Adding --offload-compress flag for ${INSTANCE_NAME}")
            target_compile_options(${INSTANCE_NAME} PRIVATE --offload-compress)
        endif()

        set_target_properties(${INSTANCE_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)
        clang_tidy_check(${INSTANCE_NAME})
        set(result 0)
        message("add_instance_library ${INSTANCE_NAME}")
    else()
        message("skip_instance_libary ${INSTANCE_NAME}")
    endif()
    set(result ${result} PARENT_SCOPE)
endfunction(add_instance_library INSTANCE_NAME)


file(GLOB dir_list LIST_DIRECTORIES true *)
set(CK_DEVICE_OTHER_INSTANCES)
set(CK_DEVICE_GEMM_INSTANCES)
set(CK_DEVICE_CONV_INSTANCES)
set(CK_DEVICE_MHA_INSTANCES)
set(CK_DEVICE_CONTRACTION_INSTANCES)
set(CK_DEVICE_REDUCTION_INSTANCES)
FOREACH(subdir_path ${dir_list})
    set(target_dir)
    IF(IS_DIRECTORY "${subdir_path}")
        set(cmake_instance)
        file(READ "${subdir_path}/CMakeLists.txt" cmake_instance)
        set(add_inst 0)
        if(("${cmake_instance}" MATCHES "_fp8" OR "${cmake_instance}" MATCHES "_f8") AND DTYPES MATCHES "fp8")
            message("fp8 instance found!")
            set(add_inst 1)
        endif()
        if(("${cmake_instance}" MATCHES "_bf8" OR "${cmake_instance}" MATCHES "_b8") AND DTYPES MATCHES "bf8")
            message("bf8 instance found!")
            set(add_inst 1)
        endif()
        if(("${cmake_instance}" MATCHES "_bf16" OR "${cmake_instance}" MATCHES "_b16") AND DTYPES MATCHES "bf16")
            message("bf16 instance found!")
            set(add_inst 1)
        endif()
        if(("${cmake_instance}" MATCHES "_fp16" OR "${cmake_instance}" MATCHES "_f16") AND DTYPES MATCHES "fp16")
            message("fp16 instance found!")
            set(add_inst 1)
        endif()
        if(("${cmake_instance}" MATCHES "_fp32" OR "${cmake_instance}" MATCHES "_f32") AND DTYPES MATCHES "fp32")
            message("fp32 instance found!")
            set(add_inst 1)
        endif()
        if(("${cmake_instance}" MATCHES "_fp64" OR "${cmake_instance}" MATCHES "_f64") AND DTYPES MATCHES "fp64")
            message("fp64 instance found!")
            set(add_inst 1)
        endif()
        if(("${cmake_instance}" MATCHES "_int8" OR "${cmake_instance}" MATCHES "_i8") AND DTYPES MATCHES "int8")
            message("int8 instance found!")
            set(add_inst 1)
        endif()
        if(NOT ("${cmake_instance}" MATCHES "_fp8" OR
                "${cmake_instance}" MATCHES "_f8" OR
                "${cmake_instance}" MATCHES "_fp16" OR
                "${cmake_instance}" MATCHES "_f16" OR
                "${cmake_instance}" MATCHES "_fp32" OR
                "${cmake_instance}" MATCHES "_f32" OR
                "${cmake_instance}" MATCHES "_fp64" OR
                "${cmake_instance}" MATCHES "_f64" OR
                "${cmake_instance}" MATCHES "_bf16" OR
                "${cmake_instance}" MATCHES "_int8" OR
                "${cmake_instance}" MATCHES "_i8" OR
                "${cmake_instance}" MATCHES "_int4"))
            message("instance should be built for all types!")
            set(add_inst 1)
        endif()
        if(NOT DEFINED DTYPES)
            set(add_inst 1)
        endif()

        set(INST_TARGETS ${SUPPORTED_GPU_TARGETS})

        if(("${cmake_instance}" MATCHES "quantization") AND (DEFINED DTYPES) AND (NOT DTYPES MATCHES "int8"))
            message("quantization instances will not be built!")
            set(add_inst 0)
        endif()
        if(("${cmake_instance}" MATCHES "ONLY DL_KERNELS") AND (NOT DEFINED DL_KERNELS))
            message("Found only dl instances, but DL_KERNELS is not set. Skipping.")
            set(add_inst 0)
        endif()
        if(("${cmake_instance}" MATCHES "ONLY XDL_KERNELS") AND (NOT INST_TARGETS MATCHES "gfx9"))
            message("Found only xdl instances, but gfx9 is not on the targets list. Skipping.")
            set(add_inst 0)
        endif()
	if(("${cmake_instance}" MATCHES "ONLY WMMA_KERNELS") AND (NOT INST_TARGETS MATCHES "gfx11") AND (NOT INST_TARGETS MATCHES "gfx12"))
            message("Found only wmma instances, but gfx11 is not on the targets list. Skipping.")
            set(add_inst 0)
        endif()
        if(("${cmake_instance}" MATCHES "ONLY XDL_AND_DL_KERNELS") AND (NOT DEFINED DL_KERNELS) AND (NOT INST_TARGETS MATCHES "gfx9"))
            message("Found only xdl and dl instances, but gfx9 is not on the targets listand DL_KERNELS is not set. Skipping.")
            set(add_inst 0)
        endif()
	if(("${cmake_instance}" MATCHES "ONLY XDL_AND_WMMA_KERNELS") AND (NOT INST_TARGETS MATCHES "gfx11") AND (NOT INST_TARGETS MATCHES "gfx12") AND (NOT INST_TARGETS MATCHES "gfx9"))
            message("Found only xdl and wmma instances, but gfx11 and gfx9 are not on the targets list. Skipping.")
            set(add_inst 0)
        endif()
	if(("${cmake_instance}" MATCHES "XDL_DL_WMMA_KERNELS") AND (NOT INST_TARGETS MATCHES "gfx11") AND (NOT INST_TARGETS MATCHES "gfx12") AND (NOT INST_TARGETS MATCHES "gfx9") AND (NOT DEFINED DL_KERNELS))
            message("Found xdl, dl, and wmma instances, but none of those meet the target list. Skipping.")
            set(add_inst 0)
        endif()
        if((add_inst EQUAL 1))
            get_filename_component(target_dir ${subdir_path} NAME)
            add_subdirectory(${target_dir})
            if("${cmake_instance}" MATCHES "gemm")
                list(APPEND CK_DEVICE_GEMM_INSTANCES $<TARGET_OBJECTS:device_${target_dir}_instance>)
            elseif("${cmake_instance}" MATCHES "conv")
                 list(APPEND CK_DEVICE_CONV_INSTANCES $<TARGET_OBJECTS:device_${target_dir}_instance>)
            elseif("${cmake_instance}" MATCHES "mha")
                 list(APPEND CK_DEVICE_MHA_INSTANCES $<TARGET_OBJECTS:device_${target_dir}_instance>)
            elseif("${cmake_instance}" MATCHES "contr")
                 list(APPEND CK_DEVICE_CONTRACTION_INSTANCES $<TARGET_OBJECTS:device_${target_dir}_instance>)
            elseif("${cmake_instance}" MATCHES "reduce")
                 list(APPEND CK_DEVICE_REDUCTION_INSTANCES $<TARGET_OBJECTS:device_${target_dir}_instance>)
            else()
                 list(APPEND CK_DEVICE_OTHER_INSTANCES $<TARGET_OBJECTS:device_${target_dir}_instance>)
            endif()
            message("add_instance_directory ${subdir_path}")
        else()
            message("skip_instance_directory ${subdir_path}")
        endif()
    ENDIF()
ENDFOREACH()



if(CK_DEVICE_OTHER_INSTANCES)
        add_library(device_other_operations ${CK_DEVICE_OTHER_INSTANCES})
        add_library(composablekernels::device_other_operations ALIAS device_other_operations)
        set_target_properties(device_other_operations PROPERTIES POSITION_INDEPENDENT_CODE ON)
        set_target_properties(device_other_operations
            PROPERTIES
            VERSION ${CMAKE_PROJECT_VERSION}
            SOVERSION ${CMAKE_PROJECT_VERSION_MAJOR}
        )
        target_include_directories(device_other_operations PUBLIC
            $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/ck>
            $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/ck/utility>
            $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/ck/tensor_description>
            $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/ck/tensor>
            $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/ck/problem_transform>
            $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/ck/tensor_operation/gpu/device>
            $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/ck/tensor_operation/gpu/device/impl>
            $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/ck/tensor_operation/gpu/grid>
            $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/ck/tensor_operation/gpu/block>
            $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/ck/tensor_operation/gpu/warp>
            $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/ck/tensor_operation/gpu/thread>
            $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/ck/tensor_operation/gpu/element>
            $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/ck/library/utility>
            $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/ck/library/tensor_operation_instance>
            $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/ck/library/tensor_operation_instance/gpu>
            $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/ck/library/tensor_operation_instance/gpu/quantization>
            $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/ck/library/tensor_operation_instance/gpu/softmax>
        )
        rocm_install(TARGETS device_other_operations
            EXPORT device_other_operationsTargets)
        rocm_install(EXPORT device_other_operationsTargets
            FILE composable_kerneldevice_other_operationsTargets.cmake
            NAMESPACE composable_kernel::
            DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/composable_kernel
        )
endif()
if(CK_DEVICE_GEMM_INSTANCES)
        add_library(device_gemm_operations ${CK_DEVICE_GEMM_INSTANCES})
        add_library(composablekernels::device_gemm_operations ALIAS device_gemm_operations)
        target_compile_features(device_gemm_operations PUBLIC)
        set_target_properties(device_gemm_operations PROPERTIES POSITION_INDEPENDENT_CODE ON)
        set_target_properties(device_gemm_operations
            PROPERTIES
            VERSION ${CMAKE_PROJECT_VERSION}
            SOVERSION ${CMAKE_PROJECT_VERSION_MAJOR}
        )
        target_include_directories(device_gemm_operations PUBLIC
            $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/ck/library/tensor_operation_instance/gpu>
        )
        rocm_install(TARGETS device_gemm_operations
            EXPORT device_gemm_operationsTargets)
        rocm_install(EXPORT device_gemm_operationsTargets
            FILE composable_kerneldevice_gemm_operationsTargets.cmake
            NAMESPACE composable_kernel::
            DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/composable_kernel
        )
endif()
if(CK_DEVICE_CONV_INSTANCES)
        add_library(device_conv_operations ${CK_DEVICE_CONV_INSTANCES})
        add_library(composablekernels::device_conv_operations ALIAS device_conv_operations)
        target_compile_features(device_conv_operations PUBLIC)
        set_target_properties(device_conv_operations PROPERTIES POSITION_INDEPENDENT_CODE ON)
        set_target_properties(device_conv_operations
            PROPERTIES
            VERSION ${CMAKE_PROJECT_VERSION}
            SOVERSION ${CMAKE_PROJECT_VERSION_MAJOR}
        )
        target_include_directories(device_conv_operations PUBLIC
            $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/ck/library/tensor_operation_instance/gpu>
            $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/ck/library/tensor_operation_instance/gpu/conv_tensor_rearrange>
            $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/ck/library/tensor_operation_instance/gpu/grouped_conv_bwd_data>
            $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/ck/library/tensor_operation_instance/gpu/grouped_conv_bwd_weight>
            $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/ck/library/tensor_operation_instance/gpu/grouped_conv_fwd>
        )
        rocm_install(TARGETS device_conv_operations
            EXPORT device_conv_operationsTargets)
        rocm_install(EXPORT device_conv_operationsTargets
            FILE composable_kerneldevice_conv_operationsTargets.cmake
            NAMESPACE composable_kernel::
            DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/composable_kernel
        )
endif()
if(CK_DEVICE_MHA_INSTANCES)
        set(gpu_list ${INST_TARGETS})
        if(gpu_list MATCHES "gfx94" OR gpu_list MATCHES "gfx90a" OR gpu_list MATCHES "gfx95")
            add_library(device_mha_operations ${CK_DEVICE_MHA_INSTANCES})
            set_target_properties(device_mha_operations
                PROPERTIES
                VERSION ${CMAKE_PROJECT_VERSION}
                SOVERSION ${CMAKE_PROJECT_VERSION_MAJOR}
            )
            add_library(composablekernels::device_mha_operations ALIAS device_mha_operations)
            target_compile_features(device_mha_operations PUBLIC)
            set_target_properties(device_mha_operations PROPERTIES POSITION_INDEPENDENT_CODE ON)
            
            rocm_install(TARGETS device_mha_operations
                EXPORT device_mha_operationsTargets)
            rocm_install(EXPORT device_mha_operationsTargets
                FILE composable_kerneldevice_mha_operationsTargets.cmake
                NAMESPACE composable_kernel::
                DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/composable_kernel
            )
        endif()
endif()
if(CK_DEVICE_CONTRACTION_INSTANCES)
        add_library(device_contraction_operations ${CK_DEVICE_CONTRACTION_INSTANCES})
        add_library(composablekernels::device_contraction_operations ALIAS device_contraction_operations)
        target_compile_features(device_contraction_operations PUBLIC)
        set_target_properties(device_contraction_operations PROPERTIES POSITION_INDEPENDENT_CODE ON)
        set_target_properties(device_contraction_operations
            PROPERTIES
            VERSION ${CMAKE_PROJECT_VERSION}
            SOVERSION ${CMAKE_PROJECT_VERSION_MAJOR}
        )
        target_include_directories(device_contraction_operations PUBLIC
            $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/ck/library/tensor_operation_instance/gpu>
            $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/ck/library/tensor_operation_instance/gpu/contraction>
        )
        rocm_install(TARGETS device_contraction_operations
            EXPORT device_contraction_operationsTargets)
        rocm_install(EXPORT device_contraction_operationsTargets
            FILE composable_kerneldevice_contraction_operationsTargets.cmake
            NAMESPACE composable_kernel::
            DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/composable_kernel
        )
endif()
if(CK_DEVICE_REDUCTION_INSTANCES)
        add_library(device_reduction_operations ${CK_DEVICE_REDUCTION_INSTANCES})
        add_library(composablekernels::device_reduction_operations ALIAS device_reduction_operations)
        target_compile_features(device_reduction_operations PUBLIC)
        set_target_properties(device_reduction_operations PROPERTIES POSITION_INDEPENDENT_CODE ON)
        set_target_properties(device_reduction_operations
            PROPERTIES
            VERSION ${CMAKE_PROJECT_VERSION}
            SOVERSION ${CMAKE_PROJECT_VERSION_MAJOR}
        )
        target_include_directories(device_reduction_operations PUBLIC
            $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/ck/library/tensor_operation_instance/gpu/reduce>
        )
        rocm_install(TARGETS device_reduction_operations
            EXPORT device_reduction_operationsTargets)
        rocm_install(EXPORT device_reduction_operationsTargets
            FILE composable_kerneldevice_reduction_operationsTargets.cmake
            NAMESPACE composable_kernel::
            DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/composable_kernel
        )
endif()

add_library(device_operations INTERFACE)
target_link_libraries(device_operations INTERFACE
    device_contraction_operations
    device_conv_operations
    device_gemm_operations
    device_other_operations
    device_reduction_operations
    utility)

set(DEV_OPS_INC_DIRS
    ${PROJECT_SOURCE_DIR}/include/ck/
    ${PROJECT_SOURCE_DIR}/library/include/ck/
)
rocm_install(DIRECTORY ${DEV_OPS_INC_DIRS} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/ck)

