cmake_minimum_required(VERSION 3.15)

# Quick return if NAG::dco_cpp is already imported.
if(TARGET NAG::dco_cpp)
  return()
endif()

# Tries to retrieve installation path from given product code
function(_dco_get_installation_path prodcode installation_path)
  # Search default installation directories
  set(_dco_def_dirs $ENV{HOME}/NAG/${prodcode} /opt/NAG/${prodcode} /opt/${prodcode})
  find_file(
    _installation_path_loc dco.hpp
    PATHS ${_dco_def_dirs}
    PATH_SUFFIXES include
    NO_DEFAULT_PATH)
  if(_installation_path_loc)
    get_filename_component(_installation_path_loc ${_installation_path_loc}
      DIRECTORY)
    string(REPLACE "/include" "" _installation_path_loc
      ${_installation_path_loc})
  endif()
  set(${installation_path}
    ${_installation_path_loc}
    PARENT_SCOPE)
  unset(_installation_path_loc CACHE)
endfunction()

if("${CMAKE_SIZEOF_VOID_P}" STREQUAL "4")
  set(_dco_is_32 TRUE)
  unset(_dco_is_64)
else()
  unset(_dco_is_32)
  set(_dco_is_64 TRUE)
endif()

if(NOT DEFINED NAG_dco_cpp_USE_DYNAMIC_RUNTIME)
  set(NAG_dco_cpp_USE_DYNAMIC_RUNTIME FALSE)
endif()

# Local search
# ##############################################################################
if(NOT(NAG_dco_cpp_DIR))
  find_path(
    NAG_dco_cpp_INCLUDE_DIR dco.hpp
    HINTS ${CMAKE_SOURCE_DIR}/dco_cpp_dev
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/..
    $ENV{HOME}/dco_cpp_dev
    $ENV{HOME}/dco
    $ENV{HOME}/dco/dco_cpp_dev
    $ENV{HOME}/git/dco_cpp_dev
    $ENV{HOME}/mygit/dco_cpp_dev
    $ENV{HOME}/Software/dco_cpp_dev
    $ENV{DCO_LIB}
    PATH_SUFFIXES src include)

  # Set path if include dir was found otherwise try to find dco/c++ in other
  # common places (store result in NAG_dco_cpp_DIR)
  if(NAG_dco_cpp_INCLUDE_DIR)
    set(NAG_dco_cpp_DIR ${NAG_dco_cpp_INCLUDE_DIR}/..)
    get_filename_component(NAG_dco_cpp_DIR "${NAG_dco_cpp_DIR}" ABSOLUTE)
  else()
    # LINUX
    foreach(_dco_version_major 9;8;7;6;5;4;3;2)
      if(NAG_dco_cpp_DIR)
        break()
      endif()

      foreach(_dco_version_minor 9;8;7;6;5;4;3;2;1;0)
        # Try to find non-license-managed version
        _dco_get_installation_path("dclin${_dco_version_major}${_dco_version_minor}nn" NAG_dco_cpp_DIR)
        if(NAG_dco_cpp_DIR)
          break()
        endif()

        # Try to find license-managed version, 64-bit, using the new product codes
        # _dco_get_installation_path("dbl6i0${_dco_version_major}xnl" NAG_dco_cpp_DIR)
        # if(NAG_dco_cpp_DIR AND _dco_is_64)
        #   break()
        # endif()

        # Try to find license-managed version 64-bit
        _dco_get_installation_path("dcl6i${_dco_version_major}${_dco_version_minor}ngl" NAG_dco_cpp_DIR)
        if(NAG_dco_cpp_DIR AND _dco_is_64)
          break()
        endif()

        # Try to find license-managed version 32-bit
        _dco_get_installation_path("dclux${_dco_version_major}${_dco_version_minor}ngl" NAG_dco_cpp_DIR)
        if(NAG_dco_cpp_DIR AND _dco_is_64)
          if(NOT(NAG_dco_cpp_FIND_QUIETLY))
            message(
              STATUS
              "NAG config - dco/c++: 32-bit version found here: "
              "'${NAG_dco_cpp_DIR}'. Continue searching. If you really want "
              "to use this 32-bit version on your 64-bit system, please "
              "specify NAG_dco_cpp_DIR explicitly.")
          endif()
        endif()

        if(NAG_dco_cpp_DIR AND _dco_is_32)
          break()
        endif()
      endforeach()
    endforeach()
  endif()
else()
  if(NOT(IS_ABSOLUTE ${NAG_dco_cpp_DIR}))
    if(NOT(NAG_dco_cpp_FIND_QUIETLY))
      message(
        WARNING "NAG config - dco/c++: Relative path given as NAG_dco_cpp_DIR; "
        "interpreted as relative to CMAKE_CURRENT_SOURCE_DIR, i.e. "
        "'${CMAKE_CURRENT_SOURCE_DIR}/${NAG_dco_cpp_DIR}'.")
    endif()
  endif()
endif()

# From here on, NAG_dco_cpp_DIR should be defined
# ##############################################################################
get_filename_component(NAG_dco_cpp_DIR "${NAG_dco_cpp_DIR}" ABSOLUTE)

# Find path to dco.hpp
find_path(NAG_dco_cpp_INCLUDE_DIR dco.hpp
  HINTS ${NAG_dco_cpp_DIR}/include ${NAG_dco_cpp_DIR}/src ${NAG_dco_cpp_DIR})

if(NOT(NAG_dco_cpp_INCLUDE_DIR))
  if(NAG_dco_cpp_FIND_REQUIRED)
    message(FATAL_ERROR "NAG config - dco/c++: not found. Set NAG_dco_cpp_DIR respectively.")
  else()
    if(NOT(NAG_dco_cpp_FIND_QUIETLY))
      message(WARNING "NAG config - dco/c++: not found. Set NAG_dco_cpp_DIR respectively.")
    endif()
  endif()

  unset(NAG_dco_cpp_INCLUDE_DIR CACHE)
  return()
endif()

file(READ ${NAG_dco_cpp_INCLUDE_DIR}/dco.hpp FILE_CONTENTS)
string(REGEX REPLACE ".*used DCO_FLAGS: \(.*\)" "\\1" _DCO_HPP_BUILD_DEFINES ${FILE_CONTENTS})

if(_DCO_HPP_BUILD_DEFINES MATCHES "DCO_LICENSE")
  set(_DCO_LICENSE YES)
endif()

set(NAG_dco_cpp_INCLUDE_DIRS ${NAG_dco_cpp_INCLUDE_DIR})
mark_as_advanced(NAG_dco_cpp_INCLUDE_DIR)
mark_as_advanced(NAG_dco_cpp_INCLUDE_DIRS)

# Get dco/c++ version number
# ##############################################################################
set(DCO_VERSION_MAJOR "0")
set(DCO_VERSION_MINOR "0")
set(DCO_VERSION_PATCH "0")
find_file(_DCO_VERSION_HPP dco_version.hpp
  HINTS ${NAG_dco_cpp_DIR}/include ${NAG_dco_cpp_DIR}/src ${NAG_dco_cpp_DIR})

if(_DCO_VERSION_HPP)
  file(READ ${_DCO_VERSION_HPP} FILE_CONTENTS)
  string(REGEX REPLACE ".*#define DCO_VERSION [v]*\([0-9].[0-9].[0-9]\).*"
    "\\1" VERSION ${FILE_CONTENTS})
  string(REGEX REPLACE "\([0-9]\).[0-9].[0-9]" "\\1" DCO_VERSION_MAJOR
    ${VERSION})
  string(REGEX REPLACE "[0-9].\([0-9]\).[0-9]" "\\1" DCO_VERSION_MINOR
    ${VERSION})
  string(REGEX REPLACE "[0-9].[0-9].\([0-9]\)" "\\1" DCO_VERSION_PATCH
    ${VERSION})
endif()

unset(_DCO_VERSION_HPP CACHE)

# Licensed version? => search library
# ##############################################################################
find_library(
  NAG_dco_cpp_LIBRARY
  NAMES dcoc
  HINTS ${NAG_dco_cpp_DIR}/lib)

get_filename_component(_dco_library_dir ${NAG_dco_cpp_LIBRARY} DIRECTORY)

# Finalize find_package with some output
# ##############################################################################
string(CONCAT _dco_cpp_version "${DCO_VERSION_MAJOR}." "${DCO_VERSION_MINOR}."
  "${DCO_VERSION_PATCH}")
string(CONCAT _dco_cpp_find_version "${NAG_dco_cpp_FIND_VERSION_MAJOR}."
  "${NAG_dco_cpp_FIND_VERSION_MINOR}."
  "${NAG_dco_cpp_FIND_VERSION_PATCH}")

if(${_dco_cpp_version} VERSION_LESS ${_dco_cpp_find_version})
  message(
    FATAL_ERROR
    "NAG config - dco/c++: not found. Required version "
    "${_dco_cpp_find_version} larger than detected version "
    "${_dco_cpp_version}. Set NAG_dco_cpp_DIR respectively.")
endif()

if(NAG_dco_cpp_FIND_VERSION_EXACT)
  if(NOT ${_dco_cpp_version} VERSION_EQUAL ${_dco_cpp_find_version})
    message(
      FATAL_ERROR
      "NAG config - dco/c++: not found. Required version "
      "${_dco_cpp_find_version} doesn't match detected version "
      "${_dco_cpp_version}. Set NAG_dco_cpp_DIR respectively.")
  endif()
endif()

if(NOT(NAG_dco_cpp_INCLUDE_DIR) OR(_DCO_LICENSE AND NOT(NAG_dco_cpp_LIBRARY)
  ))
  set(NAG_dco_cpp_FOUND 0)

  if(NAG_dco_cpp_FIND_REQUIRED)
    if(NOT(NAG_dco_cpp_INCLUDE_DIR))
      message(FATAL_ERROR "NAG config - dco/c++: include directory not found. "
        "Set NAG_dco_cpp_DIR respectively.")
    else()
      message(FATAL_ERROR "NAG config - dco/c++: library not found. "
        "Check correct installation.")
    endif()
  else()
    if(NOT(NAG_dco_cpp_FIND_QUIETLY))
      if(NOT(NAG_dco_cpp_INCLUDE_DIR))
        message(WARNING "NAG config - dco/c++: include directory not found. "
          "Set NAG_dco_cpp_DIR respectively.")
      else()
        message(WARNING "NAG config - dco/c++: library not found. "
          "Check correct installation.")
      endif()
    endif()
  endif()
else()
  if(NAG_dco_cpp_DIR)
    if(NOT(NAG_dco_cpp_FIND_QUIETLY))
      message(STATUS "NAG config - dco/c++: Picking v${_dco_cpp_version} in "
        "'${NAG_dco_cpp_DIR}'. Imported target 'NAG::dco_cpp'.")
    endif()
  endif()

  mark_as_advanced(DCO_VERSION_MAJOR)
  mark_as_advanced(DCO_VERSION_MINOR)
  mark_as_advanced(DCO_VERSION_PATCH)
  set(NAG_dco_cpp_FOUND 1)

  if(NAG_dco_cpp_LIBRARY)
    add_library(NAG::dco_cpp STATIC IMPORTED GLOBAL)
    set_target_properties(NAG::dco_cpp PROPERTIES IMPORTED_LOCATION
      "${NAG_dco_cpp_LIBRARY}")
  else()
    add_library(NAG::dco_cpp INTERFACE IMPORTED GLOBAL)
  endif()

  set_target_properties(NAG::dco_cpp PROPERTIES INTERFACE_INCLUDE_DIRECTORIES
    "${NAG_dco_cpp_INCLUDE_DIR}")
endif()

unset(NAG_dco_cpp_INCLUDE_DIR CACHE)
unset(NAG_dco_cpp_LIBRARY CACHE)

