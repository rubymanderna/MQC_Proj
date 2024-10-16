# psi4PluginCache.cmake
# ---------------------
#
# This module sets some likely variable values to initialize the CMake cache in your plugin.
# See ``psi4 --plugin-compile`` for use.
#

set(CMAKE_C_COMPILER          "/usr/bin/cc" CACHE STRING "")
set(CMAKE_C_FLAGS             " -march=native" CACHE STRING "")
set(CMAKE_CXX_COMPILER        "/usr/bin/c++" CACHE STRING "")
set(CMAKE_CXX_FLAGS           " -march=native" CACHE STRING "")
set(CMAKE_Fortran_COMPILER    "" CACHE STRING "")
set(CMAKE_Fortran_FLAGS       "" CACHE STRING "")

#set(CMAKE_INSTALL_PREFIX      "/repo/build/psi4/objdir/stage" CACHE PATH "")
set(CMAKE_INSTALL_LIBDIR      "lib" CACHE STRING "")
set(CMAKE_INSTALL_BINDIR      "bin" CACHE STRING "")
set(CMAKE_INSTALL_DATADIR     "share" CACHE STRING "")
set(CMAKE_INSTALL_INCLUDEDIR  "include" CACHE STRING "")
set(PYMOD_INSTALL_LIBDIR      "/" CACHE STRING "")

set(CMAKE_INSTALL_MESSAGE     "LAZY" CACHE STRING "")
set(pybind11_DIR              "/repo/build/psi4/objdir/stage/share/cmake/pybind11" CACHE PATH "")

set(PYTHON_VERSION_MAJORMINOR "3.10" CACHE STRING "")
set(Python_VERSION_MAJORMINOR "3.10" CACHE STRING "")
set(PYTHON_EXECUTABLE         "/opt/venv/bin/python3.10" CACHE STRING "")
set(Python_EXECUTABLE         "/opt/venv/bin/python3.10" CACHE STRING "")

