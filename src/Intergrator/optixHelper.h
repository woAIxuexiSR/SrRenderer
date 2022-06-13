#pragma once

#include <cuda_runtime.h>
#include <optix.h>
#include <optix_stubs.h>


#define OPTIX_CHECK(call)								\
	{													\
		OptixResult res = call;							\
		if (res != OPTIX_SUCCESS)						\
		{												\
			fprintf(stderr, "Optix call (%s) failed with code %d (line %d)\n", #call, res, __LINE__);	\
			exit(-1);									\
		}												\
	}

#define LOG_SIZE 2048
#define OPTIX_LOG(call)									\
	{													\
		char log[LOG_SIZE];								\
		size_t logSize = sizeof(log);					\
		OPTIX_CHECK(call);								\
		if(logSize > 1)									\
			std::cout << "Optix log : " << log << std::endl;	\
	}

#define CUDA_CHECK(call)								\
	{													\
		cudaError_t rc = call;							\
		if (rc != cudaSuccess)							\
		{												\
			fprintf(stderr, "CUDA Error %s (%s)\n", cudaGetErrorName(rc), cudaGetErrorString(rc));	\
			exit(-1);									\
		}												\
	}

#define CUDA_SYNC_CHECK()								\
	{													\
		cudaDeviceSynchronize();						\
		cudaError_t error = cudaGetLastError();			\
		if (error != cudaSuccess)						\
		{												\
			fprintf(stderr, "CUDA Error %s (%s)\n", cudaGetErrorName(error), cudaGetErrorString(error));	\
			exit(-1);									\
		}												\
	}