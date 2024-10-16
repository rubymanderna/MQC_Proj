#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "gau2grid::gg" for configuration "Release"
set_property(TARGET gau2grid::gg APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(gau2grid::gg PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "C"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libgg.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS gau2grid::gg )
list(APPEND _IMPORT_CHECK_FILES_FOR_gau2grid::gg "${_IMPORT_PREFIX}/lib/libgg.a" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
