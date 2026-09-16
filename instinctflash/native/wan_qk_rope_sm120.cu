#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

// P009-A4: bitexact fused BF16 RMSNorm materialization + FP64-complex RoPE.
extern "C" int instinctflash_sm120_wan_qk_rope_abi_version() { return 1; }
namespace {
constexpr int kDim=3072, kVec=4, kPairs=64;
struct RMSData { float mean, sigma2, count; };
__device__ __forceinline__ RMSData combine(RMSData dataB, RMSData dataA) {
  return {0.f, dataB.sigma2 + dataA.sigma2, 0.f};
}
__device__ __forceinline__ RMSData reduce_stats(RMSData wd, float* buf) {
  for (int off=16; off>0; off>>=1) {
    RMSData other{__shfl_down_sync(0xffffffffu,wd.mean,off),
                  __shfl_down_sync(0xffffffffu,wd.sigma2,off),
                  __shfl_down_sync(0xffffffffu,wd.count,off)};
    wd=combine(wd,other);
  }
  float* ms=buf; float* counts=buf+blockDim.y;
  for (int off=blockDim.y/2; off>0; off/=2) {
    if (threadIdx.x==0 && threadIdx.y>=off && threadIdx.y<2*off) {
      int y=threadIdx.y-off; ms[2*y]=wd.mean; ms[2*y+1]=wd.sigma2; counts[y]=wd.count;
    }
    __syncthreads();
    if (threadIdx.x==0 && threadIdx.y<off) {
      RMSData other{ms[2*threadIdx.y],ms[2*threadIdx.y+1],counts[threadIdx.y]};
      wd=combine(wd,other);
    }
    __syncthreads();
  }
  if (threadIdx.x==0 && threadIdx.y==0) { ms[0]=0.f; ms[1]=wd.sigma2/float(kDim); }
  __syncthreads(); return {0.f,ms[1],0.f};
}
__global__ void kernel(const __nv_bfloat16* x, const __nv_bfloat16* weight,
                       const float2* freqs, __nv_bfloat16* out, float* rstds,
                       int rows, float eps) {
  extern __shared__ float buf[];
  int row=blockIdx.x, base=row*kDim;
  int tid=threadIdx.x+threadIdx.y*blockDim.x, threads=blockDim.x*blockDim.y;
  RMSData wd{0.f,0.f,0.f};
  for(int i=tid;i<kDim/kVec;i+=threads) {
    #pragma unroll
    for(int j=0;j<kVec;j++) { float v=__bfloat162float(x[base+i*kVec+j]); wd.sigma2=wd.sigma2+v*v; }
  }
  wd=reduce_stats(wd,buf); float rstd=rsqrtf(wd.sigma2+eps);
  // One logical iteration handles a complex pair so BF16 materialization precedes FP64 RoPE.
  for(int pair=tid;pair<kDim/2;pair+=threads) {
    int c0=2*pair, c1=c0+1;
    float n0=__bfloat162float(weight[c0])*(rstd*__bfloat162float(x[base+c0]));
    float n1=__bfloat162float(weight[c1])*(rstd*__bfloat162float(x[base+c1]));
    __nv_bfloat16 b0=__float2bfloat16_rn(n0), b1=__float2bfloat16_rn(n1);
    double a=(double)__bfloat162float(b0), b=(double)__bfloat162float(b1);
    float2 f=freqs[row*kPairs+(pair%kPairs)];
    double real=a*(double)f.x-b*(double)f.y;
    double imag=a*(double)f.y+b*(double)f.x;
    out[base+c0]=__float2bfloat16_rn((float)real);
    out[base+c1]=__float2bfloat16_rn((float)imag);
  }
  if(tid==0) rstds[row]=rstd;
}
}
extern "C" int wan_qk_rms_rope_bf16(uintptr_t x, uintptr_t weight, uintptr_t freqs,
 uintptr_t out, uintptr_t rstds, int rows, int dim, float eps, uintptr_t stream) {
 if(rows<=0||dim!=kDim) return (int)cudaErrorInvalidValue;
 kernel<<<rows,dim3(32,4,1),6*sizeof(float),(cudaStream_t)stream>>>(
  (const __nv_bfloat16*)x,(const __nv_bfloat16*)weight,(const float2*)freqs,
  (__nv_bfloat16*)out,(float*)rstds,rows,eps);
 return (int)cudaGetLastError();
}
