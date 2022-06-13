#pragma once

#include "optixHelper.h"
#include <vector>

class cudaBuffer
{
public:
    size_t size { 0 };
    void* d_ptr { nullptr };

    inline void alloc(size_t _s)
    {
        if(d_ptr != nullptr)
            CUDA_CHECK(cudaFree(d_ptr));
        size = _s;
        CUDA_CHECK(cudaMalloc(&d_ptr, size));
    }

    template<typename T>
    inline void upload(const std::vector<T>& vt)
    {
        if(d_ptr != nullptr)
            CUDA_CHECK(cudaFree(d_ptr));
        size = vt.size() * sizeof(T);
        CUDA_CHECK(cudaMalloc(&d_ptr, size));
        CUDA_CHECK(cudaMemcpy(d_ptr, vt.data(), size, cudaMemcpyHostToDevice));
    }

    template<typename T>
    inline void download(T* t)
    {
        if(d_ptr == nullptr) return;
        CUDA_CHECK(cudaMemcpy(t, d_ptr, size, cudaMemcpyDeviceToHost));
    }

    ~cudaBuffer()
    {
        if(d_ptr != nullptr)
            CUDA_CHECK(cudaFree(d_ptr));
    }
};