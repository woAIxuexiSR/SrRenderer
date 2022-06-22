#include <optix_device.h>
#include <device_launch_parameters.h>
#include "LaunchParams.h"
#include "deviceHelper.h"

extern "C" __constant__ LaunchParams launchParams;

static __forceinline__ __device__
void* unpackPointer(uint32_t i0, uint32_t i1)
{
    const uint64_t uptr = static_cast<uint64_t>(i0) << 32 | i1;
    void* ptr = reinterpret_cast<void*>(uptr);
    return ptr;
}

static __forceinline__ __device__
void  packPointer(void* ptr, uint32_t& i0, uint32_t& i1)
{
    const uint64_t uptr = reinterpret_cast<uint64_t>(ptr);
    i0 = uptr >> 32;
    i1 = uptr & 0x00000000ffffffff;
}

template<typename T>
static __forceinline__ __device__ T* getPRD()
{
    const uint32_t u0 = optixGetPayload_0();
    const uint32_t u1 = optixGetPayload_1();
    return reinterpret_cast<T*>(unpackPointer(u0, u1));
}


extern "C" __global__ void __closesthit__radiance() {
    int* p = getPRD<int>();
    p[0] = 1;
}

extern "C" __global__ void __closesthit__shadow() {}

extern "C" __global__ void __anyhit__radiance() {}

extern "C" __global__ void __anyhit__shadow() {}

extern "C" __global__ void __miss__radiance() {}

extern "C" __global__ void __miss__shadow() {}



extern "C" __global__ void __raygen__()
{
    const int ix = optixGetLaunchIndex().x;
    const int iy = optixGetLaunchIndex().y;

    float xx = ((ix + 0.5f) / launchParams.width) * 2.0f - 1.0f;
    float yy = ((iy + 0.5f) / launchParams.height) * 2.0f - 1.0f;

    float3 ori = make_float3(0.0f, 0.0f, -1.0f);
    float3 dir = make_float3(xx, yy, 1.0f);

    int p = 0;
    unsigned u0, u1;
    packPointer(&p, u0, u1);
    optixTrace(
        launchParams.traversable,
        ori,
        dir,
        1.0e-3f,
        1.0e20f,
        0.0f,
        (unsigned) 255,
        OPTIX_RAY_FLAG_DISABLE_ANYHIT,
        RADIANCE_RAY_TYPE,
        RAY_TYPE_COUNT,
        RADIANCE_RAY_TYPE,
        u0, u1
    );

    float4 black = make_float4(0.0f, 0.0f, 0.0f, 1.0f);
    float4 white = make_float4(1.0f, 1.0f, 1.0f, 1.0f);
    
    int idx = ix + iy * launchParams.width;
    if(p == 0)
        launchParams.colorBuffer[idx] = white;
    else
        launchParams.colorBuffer[idx] = black;
}