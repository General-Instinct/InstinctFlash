#include "gemm_runner.h"
#include <string>
static thread_local std::string error;
extern "C" const char* gemm_error(){return error.c_str();}
extern "C" void* gemm_create(){try{return new GemmRunner();}catch(std::exception const& e){error=e.what();return nullptr;}}
extern "C" void gemm_destroy(void* p){delete static_cast<GemmRunner*>(p);}
extern "C" int gemm_run(void* p,void* a,void* b,void* d,int m,int n,int k,void* stream){
 try{static_cast<GemmRunner*>(p)->bf16_nn(a,b,d,m,n,k,static_cast<cudaStream_t>(stream));return 0;}
 catch(std::exception const& e){error=e.what();return -1;}}
extern "C" int gemm_tune(void* p,void* a,void* b,void* d,int m,int n,int k){
 try{static_cast<GemmRunner*>(p)->autotune_bf16_nn(a,b,d,m,n,k,32);return 0;}
 catch(std::exception const& e){error=e.what();return -1;}}
