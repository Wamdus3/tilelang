# Applied after project(tvm_ffi), following TVM's config.cmake template.
# This keeps the Windows checkout's CRLF libbacktrace shell scripts out of
# the portable build without modifying vendored files or the primary build.
set(TVM_FFI_USE_LIBBACKTRACE OFF)
set(TVM_FFI_BACKTRACE_ON_SEGFAULT OFF)
