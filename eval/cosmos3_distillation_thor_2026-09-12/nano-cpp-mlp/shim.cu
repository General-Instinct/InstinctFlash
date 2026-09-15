// Thin diagnostic ABI over the existing inference GemmRunner. No new GEMM kernel.
#include "gemm_runner.h"
#include <string>
static thread_local std::string last_error;
extern "C" const char* ifl_error() { return last_error.c_str(); }
extern "C" void* ifl_create() {
    try { return new GemmRunner(); }
    catch (const std::exception& e) { last_error=e.what(); return nullptr; }
}
extern "C" int ifl_destroy(void* handle) {
    try { delete static_cast<GemmRunner*>(handle); return 0; }
    catch (const std::exception& e) { last_error=e.what(); return -1; }
}
extern "C" int ifl_run(void* handle, void* a, void* b, void* d,
                       int m, int n, int k, void* stream) {
    try { static_cast<GemmRunner*>(handle)->bf16_nn(a,b,d,m,n,k,
              static_cast<cudaStream_t>(stream)); return 0; }
    catch (const std::exception& e) { last_error=e.what(); return -1; }
}
extern "C" int ifl_tune(void* handle, void* a, void* b, void* d,
                        int m, int n, int k) {
    try { static_cast<GemmRunner*>(handle)->autotune_bf16_nn(a,b,d,m,n,k,32); return 0; }
    catch (const std::exception& e) { last_error=e.what(); return -1; }
}
