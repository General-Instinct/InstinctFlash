// Experimental BF16 instantiation of the engine's Blackwell strided FMHA.
// Reuses layout/GQA mapping from serving/csrc/attention/fmha_fp16_strided.cu.
// Single batch, HD=128, non-causal dense attention with tail bounds. Workspace belongs to a caller-owned handle.
#include <cuda_runtime.h>
#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cute/tensor.hpp"
#include "cutlass/arch/config.h"
#if defined(CUTLASS_ARCH_MMA_SM110A_ENABLED) && !defined(CUTLASS_ARCH_MMA_SM100A_ENABLED)
#define CUTLASS_ARCH_MMA_SM100A_ENABLED 1
#endif
#include "device/fmha.hpp"
#include "kernel/sm100_fmha_fwd_kernel_tma_warpspecialized.hpp"
#include "collective/sm100_fmha_fwd_mainloop_tma_warpspecialized.hpp"
#include "collective/sm100_fmha_fwd_epilogue_tma_warpspecialized.hpp"
#include "collective/sm100_fmha_load_tma_warpspecialized.hpp"
#include "collective/fmha_fusion.hpp"
using namespace cute;
using Bf16Element = cutlass::bfloat16_t;
using SQStride = cute::tuple<int,_1,cute::tuple<cute::tuple<int,int>,int>>;
using SKStride = cute::tuple<int,_1,cute::tuple<cute::tuple<_0,int>,int>>;
using SLStride = cute::tuple<_1,cute::tuple<cute::tuple<int,int>,int>>;
using ShapeP = cute::tuple<int,int,int,cute::tuple<cute::tuple<int,int>,int>>;
using ML = cutlass::fmha::collective::Sm100FmhaFwdMainloopTmaWarpspecialized<
 Bf16Element,float,float,Shape<_256,_128,_128>,SQStride,SKStride,SKStride,cutlass::fmha::collective::ResidualMask>;
using EP = cutlass::fmha::collective::Sm100FmhaFwdEpilogueTmaWarpspecialized<Bf16Element,float,typename ML::TileShapePV,SQStride,SLStride>;
using Bf16Kernel = cutlass::fmha::kernel::Sm100FmhaFwdKernelTmaWarpspecialized<ShapeP,ML,EP,cutlass::fmha::kernel::IndividualTileScheduler>;
using Op = cutlass::fmha::device::FMHA<Bf16Kernel>;
struct Context {void* ws=nullptr;float* lse=nullptr;int sm=0,device=0;size_t lse_bytes=0;};
constexpr size_t workspace_bytes=8*1024*1024;
extern "C" void bf16_fmha_destroy(void* ptr) {
 auto* c=(Context*)ptr;if(!c)return;
 int old=0;cudaGetDevice(&old);cudaSetDevice(c->device);
 if(c->ws)cudaFree(c->ws);if(c->lse)cudaFree(c->lse);
 cudaSetDevice(old);delete c;
}
extern "C" void* bf16_fmha_create(int max_q,int heads) {
 if(max_q<=0||heads<=0)return nullptr;
 auto* c=new Context();cudaGetDevice(&c->device);
 if(cudaDeviceGetAttribute(&c->sm,cudaDevAttrMultiProcessorCount,c->device)!=cudaSuccess){delete c;return nullptr;}
 c->lse_bytes=(size_t)((max_q+127)/128)*128*heads*sizeof(float);
 if(cudaMalloc(&c->ws,workspace_bytes)!=cudaSuccess||cudaMalloc(&c->lse,c->lse_bytes)!=cudaSuccess){bf16_fmha_destroy(c);return nullptr;}
 return c;
}
extern "C" int bf16_fmha_run(void* ctx,const void* q,const void* k,const void* v,void* o,
 int nq,int nk,int hq,int hk,int qs,int ks,float scale,void* stream) {
 auto* c=(Context*)ctx;
 if(!c||!q||!k||!v||!o||nq<=0||nk<=0||hq<=0||hk<=0||hq%hk||qs<hq*128||ks<hk*128)return -1;
 int current=0;cudaGetDevice(&current);if(current!=c->device)return -2;
 int group=hq/hk;int qr=((nq+127)/128)*128;
 if((size_t)qr*hq*sizeof(float)>c->lse_bytes)return -3;
 auto ps=make_tuple(nq,nk,128,make_tuple(make_tuple(group,hk),1));
 SQStride sQ=make_stride(qs,_1{},make_stride(make_stride(128,group*128),qs*nq));
 SQStride sO=make_stride(hq*128,_1{},make_stride(make_stride(128,group*128),hq*128*nq));
 SKStride sK=make_stride(ks,_1{},make_stride(make_stride(_0{},128),ks*nk));
 SLStride sL=make_stride(_1{},make_stride(make_stride(qr,qr*group),qr*hq));
 Op::Arguments args{ps,{{(Bf16Element const*)q,sQ,(Bf16Element const*)k,sK,(Bf16Element const*)v,sK},scale,1.f,1.f,1.f,1.f},
 {(Bf16Element*)o,sO,c->lse,sL},{0,c->sm}};
 Op op;
 if(op.can_implement(args)!=cutlass::Status::kSuccess)return -4;
 if(Op::get_workspace_size(args)>workspace_bytes)return -5;
 auto st=(cudaStream_t)stream;
 if(op.initialize(args,c->ws,st)!=cutlass::Status::kSuccess)return -6;
 return op.run(st)==cutlass::Status::kSuccess?0:-7;
}
