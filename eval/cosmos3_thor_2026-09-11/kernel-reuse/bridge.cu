// Thin experiment-only ABI: calls existing engine kernels unchanged.
#include "norm.cuh"
#include "activation.cuh"
#include "elementwise.cuh"
extern "C" {
void probe_norm(const void* x,const void* w,void* y,int s,int d,float eps,void* stream) {
 rms_norm((const __nv_bfloat16*)x,(const __nv_bfloat16*)w,(__nv_bfloat16*)y,s,d,eps,(cudaStream_t)stream);
}
void probe_activation(const void* x,const void* u,void* y,int n,void* stream) {
 gate_silu_mul((const __nv_bfloat16*)x,(const __nv_bfloat16*)u,(__nv_bfloat16*)y,n,(cudaStream_t)stream);
}
void probe_residual(void* r,const void* x,int n,void* stream) {
 residual_add((__nv_bfloat16*)r,(const __nv_bfloat16*)x,n,(cudaStream_t)stream);
}
void probe_gate(void* r,const void* x,const void* g,int n,void* stream) {
 gate_mul_residual((__nv_bfloat16*)r,(const __nv_bfloat16*)x,(const __nv_bfloat16*)g,n,(cudaStream_t)stream);
}
}
#include "silu_mul_qwen36.cuh"
extern "C" void probe_swiglu(const void* x,const void* u,void* y,int n,void* stream) {
 flash_rt::kernels::silu_mul_qwen36_bf16((const __nv_bfloat16*)x,(const __nv_bfloat16*)u,(__nv_bfloat16*)y,n,(cudaStream_t)stream);
}
