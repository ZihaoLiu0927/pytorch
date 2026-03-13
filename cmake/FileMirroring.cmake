# File mirroring previously handled by setup.py.
# Copies source files into torchgen/packaged/ and torch/_inductor/kernel/
# so they are included in the installed packages.
#
# These use SKBUILD_PLATLIB_DIR (set by scikit-build-core) to install into
# the Python package tree directly, since they need to go into packages
# other than torch/ (e.g., torchgen/) or into specific torch/ subdirectories.

# Under scikit-build-core SKBUILD_PLATLIB_DIR points to the wheel's
# site-packages root.  For setuptools builds CMAKE_INSTALL_PREFIX is
# <source>/torch, so the correct fallback is the project source directory
# (the parent of torch/, torchgen/, tools/, etc.).
if(NOT DEFINED SKBUILD_PLATLIB_DIR)
  set(SKBUILD_PLATLIB_DIR "${PROJECT_SOURCE_DIR}")
endif()

# --- mirror_files_into_torchgen ---
# Copy ATen native function definitions and templates into torchgen/packaged/
# so that torchgen can be used standalone without the full source tree.
install(FILES
  "${PROJECT_SOURCE_DIR}/aten/src/ATen/native/native_functions.yaml"
  DESTINATION "${SKBUILD_PLATLIB_DIR}/torchgen/packaged/ATen/native"
)
install(FILES
  "${PROJECT_SOURCE_DIR}/aten/src/ATen/native/tags.yaml"
  DESTINATION "${SKBUILD_PLATLIB_DIR}/torchgen/packaged/ATen/native"
)
install(DIRECTORY
  "${PROJECT_SOURCE_DIR}/aten/src/ATen/templates/"
  DESTINATION "${SKBUILD_PLATLIB_DIR}/torchgen/packaged/ATen/templates"
)
install(DIRECTORY
  "${PROJECT_SOURCE_DIR}/tools/autograd/"
  DESTINATION "${SKBUILD_PLATLIB_DIR}/torchgen/packaged/autograd"
)

# --- mirror_inductor_external_kernels ---
# Copy vendored external kernel sources into torch/_inductor.
set(_cutedsl_src "${PROJECT_SOURCE_DIR}/third_party/cutlass/examples/python/CuTeDSL/blackwell/grouped_gemm.py")
if(EXISTS "${_cutedsl_src}")
  set(_cutedsl_dest "${SKBUILD_PLATLIB_DIR}/torch/_inductor/kernel/vendored_templates/cutedsl/kernels")
  install(FILES "${_cutedsl_src}"
    DESTINATION "${_cutedsl_dest}"
    RENAME "cutedsl_grouped_gemm.py"
  )
  # Ensure __init__.py exists for each package level
  file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/_empty_init.py" "")
  install(FILES "${CMAKE_CURRENT_BINARY_DIR}/_empty_init.py"
    DESTINATION "${SKBUILD_PLATLIB_DIR}/torch/_inductor/kernel/vendored_templates"
    RENAME "__init__.py"
  )
  install(FILES "${CMAKE_CURRENT_BINARY_DIR}/_empty_init.py"
    DESTINATION "${SKBUILD_PLATLIB_DIR}/torch/_inductor/kernel/vendored_templates/cutedsl"
    RENAME "__init__.py"
  )
  install(FILES "${CMAKE_CURRENT_BINARY_DIR}/_empty_init.py"
    DESTINATION "${_cutedsl_dest}"
    RENAME "__init__.py"
  )
endif()

# --- Symlink-replacement copies ---
# Copy files that were previously handled via symlinks in setup.py.
install(FILES
  "${PROJECT_SOURCE_DIR}/torch/_utils_internal.py"
  DESTINATION "${SKBUILD_PLATLIB_DIR}/tools/shared"
  RENAME "_utils_internal.py"
)
install(FILES
  "${PROJECT_SOURCE_DIR}/third_party/valgrind-headers/callgrind.h"
  DESTINATION "${SKBUILD_PLATLIB_DIR}/torch/utils/benchmark/utils/valgrind_wrapper"
)
install(FILES
  "${PROJECT_SOURCE_DIR}/third_party/valgrind-headers/valgrind.h"
  DESTINATION "${SKBUILD_PLATLIB_DIR}/torch/utils/benchmark/utils/valgrind_wrapper"
)
