// Independent module: no change to flash_rt_kernels or its qualified GEMM cache.
#include <pybind11/pybind11.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace py = pybind11;
void pi05_bias_qkv(const void*, const void*, void*, void*, void*, cudaStream_t);
void pi05_bias_gelu_fp8(const void*, const void*, void*, const float*, int, cudaStream_t);

struct Buffer { uintptr_t address; size_t bytes; size_t alignment; };

static void validate(int rows, int dim, int expected_dim, const std::vector<Buffer>& buffers) {
    if (rows != 512 || dim != expected_dim) throw std::invalid_argument("unsupported pi05 SM120 fusion shape");
    int device;
    cudaDeviceProp properties{};
    if (cudaGetDevice(&device) != cudaSuccess || cudaGetDeviceProperties(&properties, device) != cudaSuccess)
        throw std::runtime_error("cannot query CUDA device");
    if (properties.major != 12 || properties.minor != 0)
        throw std::invalid_argument("pi05 fusion requires SM120");
    for (const auto& buffer : buffers) {
        if (!buffer.address || buffer.address % buffer.alignment)
            throw std::invalid_argument("null or misaligned fusion pointer");
        cudaPointerAttributes attributes{};
        if (cudaPointerGetAttributes(&attributes, reinterpret_cast<void*>(buffer.address)) != cudaSuccess) {
            cudaGetLastError();
            throw std::invalid_argument("invalid CUDA fusion pointer");
        }
        if (attributes.type != cudaMemoryTypeDevice || attributes.device != device)
            throw std::invalid_argument("fusion pointers must belong to the current CUDA device");
    }
    for (size_t i = 0; i < buffers.size(); ++i)
        for (size_t j = i + 1; j < buffers.size(); ++j)
            if (buffers[i].address < buffers[j].address + buffers[j].bytes &&
                buffers[j].address < buffers[i].address + buffers[i].bytes)
                throw std::invalid_argument("overlapping fusion buffers");
}

PYBIND11_MODULE(flash_rt_pi05_sm120_fusion, module) {
    module.attr("abi_version") = 1;
    module.def("bias_qkv", [](uintptr_t x, uintptr_t b, uintptr_t q, uintptr_t k, uintptr_t v,
                              int rows, int dim, uintptr_t stream) {
        validate(rows, dim, 1152, {{x, 512*3456*2, 16}, {b, 3456*2, 16}, {q, 512*1152*2, 16},
                                  {k, 512*1152*2, 16}, {v, 512*1152*2, 16}});
        pi05_bias_qkv((void*)x, (void*)b, (void*)q, (void*)k, (void*)v, (cudaStream_t)stream);
    }, py::arg("x"), py::arg("bias"), py::arg("q"), py::arg("k"), py::arg("v"),
       py::arg("rows"), py::arg("dim"), py::arg("stream") = 0);
    module.def("bias_gelu_fp8", [](uintptr_t x, uintptr_t b, uintptr_t out, uintptr_t scale,
                                   int rows, int dim, int vector, uintptr_t stream) {
        if (vector != 4 && vector != 8) throw std::invalid_argument("unsupported vector width");
        validate(rows, dim, 4304, {{x, 512*4304*2, 16}, {b, 4304*2, 16},
                                  {out, 512*4304, 8}, {scale, 4, 4}});
        pi05_bias_gelu_fp8((void*)x, (void*)b, (void*)out, (float*)scale, vector, (cudaStream_t)stream);
    }, py::arg("x"), py::arg("bias"), py::arg("out"), py::arg("scale"),
       py::arg("rows"), py::arg("dim"), py::arg("vector") = 8, py::arg("stream") = 0);
}
