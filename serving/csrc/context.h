#pragma once

#include <cublas_v2.h>
#include <cuda_runtime.h>

// ================================================================
// FvkContext: per-instance runtime resources.
//
// Owns a single cuBLAS handle shared by ALL kernel calls.
// Created by Python (via pybind11), passed to every kernel.
// Eliminates static handles — kernels are fully stateless.
// ================================================================

struct FvkContext {
    cublasHandle_t cublas_handle;

    FvkContext() : cublas_handle(nullptr) {
        cublasCreate(&cublas_handle);
    }

    ~FvkContext() {
        if (cublas_handle) {
            cublasDestroy(cublas_handle);
            cublas_handle = nullptr;
        }
    }

    // Non-copyable (handle is unique resource)
    FvkContext(const FvkContext&) = delete;
    FvkContext& operator=(const FvkContext&) = delete;

    // Movable
    FvkContext(FvkContext&& other) noexcept : cublas_handle(other.cublas_handle) {
        other.cublas_handle = nullptr;
    }
};
