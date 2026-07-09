# PerceptThirdPartyInstall.cmake
# Find*Custom 模块在链接 SHARED 库时，将 .so 族安装到 CMAKE_INSTALL_LIBDIR。

if(PERCEPT_THIRD_PARTY_INSTALL_INCLUDED)
    return()
endif()
set(PERCEPT_THIRD_PARTY_INSTALL_INCLUDED TRUE)

include(GNUInstallDirs)

# 安装 libfoo.so / libfoo.so.N / libfoo.so.N.M.K 及同目录 symlink 到 ${CMAKE_INSTALL_LIBDIR}。
# 按安装路径去重，避免多模块重复 install 同一文件。
function(percept_install_shared_lib_family lib_path)
    if(NOT lib_path OR NOT EXISTS "${lib_path}")
        return()
    endif()
    if(NOT lib_path MATCHES "\\.so")
        return()
    endif()

    get_filename_component(_lib_path "${lib_path}" ABSOLUTE)
    get_filename_component(_lib_dir "${_lib_path}" DIRECTORY)
    get_filename_component(_lib_name "${_lib_path}" NAME)

    string(REGEX REPLACE "(\\.so).*$" "\\1" _so_stem "${_lib_name}")
    if(NOT _so_stem MATCHES "\\.so$")
        return()
    endif()

    file(GLOB _candidates LIST_DIRECTORIES false "${_lib_dir}/${_so_stem}*")
    if(NOT _candidates)
        set(_candidates "${_lib_path}")
    endif()

    foreach(_candidate IN LISTS _candidates)
        if(NOT EXISTS "${_candidate}")
            continue()
        endif()
        if(NOT _candidate MATCHES "\\.so")
            continue()
        endif()

        get_filename_component(_candidate_abs "${_candidate}" ABSOLUTE)
        get_property(_already GLOBAL PROPERTY "PERCEPT_RUNTIME_INSTALL_${_candidate_abs}")
        if(_already)
            continue()
        endif()
        set_property(GLOBAL PROPERTY "PERCEPT_RUNTIME_INSTALL_${_candidate_abs}" TRUE)

        get_filename_component(_real "${_candidate_abs}" REALPATH)
        get_filename_component(_install_name "${_candidate_abs}" NAME)
        if(NOT _real OR NOT EXISTS "${_real}")
            continue()
        endif()

        # 按原名安装真实 ELF（不保留 symlink），避免 rsync/scp 到板端后 symlink 变成文本文件。
        install(FILES "${_real}" DESTINATION "${CMAKE_INSTALL_LIBDIR}" RENAME "${_install_name}")
    endforeach()
endfunction()

function(percept_target_imported_shared_location target out_var)
    set(${out_var} "" PARENT_SCOPE)
    if(NOT TARGET ${target})
        return()
    endif()

    get_target_property(_type ${target} TYPE)
    if(NOT _type STREQUAL "SHARED_LIBRARY")
        return()
    endif()

    set(_loc "")
    foreach(_prop IMPORTED_LOCATION IMPORTED_LOCATION_RELEASE IMPORTED_LOCATION_DEBUG
                 IMPORTED_LOCATION_MINSIZEREL IMPORTED_LOCATION_RELWITHDEBINFO)
        if(_loc)
            break()
        endif()
        get_target_property(_candidate ${target} ${_prop})
        if(_candidate AND NOT _candidate STREQUAL "${_prop}-NOTFOUND")
            set(_loc "${_candidate}")
        endif()
    endforeach()

    set(${out_var} "${_loc}" PARENT_SCOPE)
endfunction()

# SHARED IMPORTED target → install 其 .so 族（由 find_package 判定静/动）
function(percept_install_imported_shared_runtime target)
    percept_target_imported_shared_location(${target} _loc)
    if(_loc)
        percept_install_shared_lib_family("${_loc}")
    endif()
endfunction()

function(percept_install_imported_shared_runtime_targets)
    foreach(_target IN LISTS ARGN)
        percept_install_imported_shared_runtime("${_target}")
    endforeach()
endfunction()
