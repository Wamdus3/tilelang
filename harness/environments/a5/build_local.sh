#!/usr/bin/env bash
# Run only inside local WSL, using micromamba run -p <local build-toolchain>.
set -euo pipefail
if [[ "$(uname -r)" != *microsoft* ]]; then
  echo "This build procedure is restricted to local WSL."
  exit 2
fi
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
artifact_dir="${TLH_A5_ARTIFACT_DIR:-$repo_root/output/a5-offline}"
local_work="${TLH_A5_LOCAL_WORK:-$HOME/.cache/tilelang-950/a5-offline}"
runtime_prefix="$local_work/target-env"
toolchain_prefix="$local_work/build-toolchain"
export PATH="$runtime_prefix/bin:$PATH"
export TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export USE_CUDA=OFF USE_ROCM=OFF USE_ASCEND=ON USE_LLVM=OFF NO_TOOLCHAIN_VERSION=1
export PIP_NO_INDEX=1 PIP_FIND_LINKS="$artifact_dir/wheels"
export CMAKE_BUILD_PARALLEL_LEVEL=8
# Keep the build toolchain's absolute prefix out of installed library RPATHs.
export LDFLAGS="-Wl,-O2 -Wl,--as-needed -Wl,-rpath-link,$toolchain_prefix/lib"
export CC="$toolchain_prefix/bin/x86_64-conda-linux-gnu-cc"
export CXX="$toolchain_prefix/bin/x86_64-conda-linux-gnu-c++"
mkdir -p "$artifact_dir/repository-wheels"
cd "$repo_root"
"$runtime_prefix/bin/python" -m pip wheel . --verbose --no-deps --no-build-isolation \
  --wheel-dir "$artifact_dir/repository-wheels" \
  --config-settings="build-dir=$local_work/native-build" \
  --config-settings="cmake.define.CMAKE_C_COMPILER=$CC" \
  --config-settings="cmake.define.CMAKE_CXX_COMPILER=$CXX" \
  --config-settings="cmake.define.CMAKE_SYSROOT=$toolchain_prefix/x86_64-conda-linux-gnu/sysroot" \
  --config-settings=cmake.define.USE_CUDA=OFF \
  --config-settings=cmake.define.USE_ROCM=OFF \
  --config-settings=cmake.define.USE_ASCEND=ON \
  --config-settings=cmake.define.TILELANG_USE_ASCEND_STUBS=ON \
  --config-settings=cmake.define.USE_LLVM=OFF \
  --config-settings="cmake.define.CMAKE_PROJECT_tvm_ffi_INCLUDE=$repo_root/harness/environments/a5/ffi_portable_options.cmake" \
  --config-settings=cmake.define.TVM_FFI_USE_LIBBACKTRACE=OFF \
  --config-settings=cmake.define.TVM_FFI_BACKTRACE_ON_SEGFAULT=OFF
"$runtime_prefix/bin/python" -m pip wheel ./harness --no-deps --no-build-isolation \
  --wheel-dir "$artifact_dir/repository-wheels"
