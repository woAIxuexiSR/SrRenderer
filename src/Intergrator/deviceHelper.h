#pragma once

#include <vector>

#include <cuda_runtime.h>
#include <optix.h>
#include <optix_stubs.h>

#include "helper_math.h"
#include "helper_cuda.h"

/* ray type */
enum
{
    RADIANCE_RAY_TYPE,
    SHADOW_RAY_TYPE,
    RAY_TYPE_COUNT
};

/* sbt record datastructure */
template <class T>
struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) SBTRecord
{
    __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};

struct RaygenData
{
    float3 background;
};
struct MissData
{
};
struct HitgroupData
{
    float3 color;
    float3 *vertex;
    uint3 *index;
    float3 *normal;
    float2 *texcoord;

    bool hasTexture;
    cudaTextureObject_t texture;

    bool isLight;
    float3 emittance;
};

typedef SBTRecord<RaygenData> RaygenSBTRecord;
typedef SBTRecord<MissData> MissSBTRecord;
typedef SBTRecord<HitgroupData> HitgroupSBTRecord;

/* cuda buffer */
class cudaBuffer
{
public:
    size_t size{0};
    void *d_ptr{nullptr};

    inline void alloc(size_t _s)
    {
        if (d_ptr != nullptr)
            checkCudaErrors(cudaFree(d_ptr));
        size = _s;
        checkCudaErrors(cudaMalloc(&d_ptr, size));
    }

    template <typename T>
    inline void upload(const std::vector<T> &vt)
    {
        if (d_ptr != nullptr)
            checkCudaErrors(cudaFree(d_ptr));
        size = vt.size() * sizeof(T);
        checkCudaErrors(cudaMalloc(&d_ptr, size));
        checkCudaErrors(cudaMemcpy(d_ptr, vt.data(), size, cudaMemcpyHostToDevice));
    }

    template <typename T>
    inline void upload(const T *t, int count)
    {
        if (d_ptr != nullptr)
            checkCudaErrors(cudaFree(d_ptr));
        size = count * sizeof(T);
        checkCudaErrors(cudaMalloc(&d_ptr, size));
        checkCudaErrors(cudaMemcpy(d_ptr, t, size, cudaMemcpyHostToDevice));
    }

    template <typename T>
    inline void download(T *t)
    {
        if (d_ptr == nullptr)
            return;
        checkCudaErrors(cudaMemcpy(t, d_ptr, size, cudaMemcpyDeviceToHost));
    }

    ~cudaBuffer()
    {
        if (d_ptr != nullptr)
            checkCudaErrors(cudaFree(d_ptr));
    }
};