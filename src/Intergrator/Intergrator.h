#pragma once

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "Model.h"
#include "Model.h"
#include "deviceHelper.h"

#include "LaunchParams.h"

/* optix helper definitions */
#define OPTIX_CHECK(call)                                                                             \
    {                                                                                                 \
        OptixResult res = call;                                                                       \
        if (res != OPTIX_SUCCESS)                                                                     \
        {                                                                                             \
            fprintf(stderr, "Optix call (%s) failed with code %d (line %d)\n", #call, res, __LINE__); \
            exit(-1);                                                                                 \
        }                                                                                             \
    }

#define LOG_SIZE 2048
#define OPTIX_LOG(call)                                      \
    {                                                        \
        char log[LOG_SIZE];                                  \
        size_t logSize = sizeof(log);                        \
        OPTIX_CHECK(call);                                   \
        if (logSize > 1)                                     \
            std::cout << "Optix log : " << log << std::endl; \
    }

class ModuleProgramGroup
{
public:
    OptixProgramGroup raygenPG;
    OptixProgramGroup missPGs[2];
    OptixProgramGroup hitgroupPGs[2];
};

/* Intergrator virtual class */
class Intergrator
{
private:
    std::vector<std::string> modulePTXs;

    CUstream stream;
    OptixDeviceContext optixContext;
    std::vector<ModuleProgramGroup> modulePGs;
    std::vector<OptixPipeline> pipelines;

    const Model *model;
    std::vector<cudaBuffer> vertexBuffer;
    std::vector<cudaBuffer> indexBuffer;
    std::vector<cudaBuffer> texcoordBuffer;
    std::vector<cudaBuffer> normalBuffer;

    std::vector<cudaArray_t> textureArrays;
    std::vector<cudaTextureObject_t> textureObjects;

    std::vector<OptixShaderBindingTable> sbts;
    std::vector<cudaBuffer> raygenRecordsBuffer;
    std::vector<cudaBuffer> missRecordsBuffer;
    std::vector<cudaBuffer> hitgroupRecordsBuffer;

    OptixTraversableHandle traversable;

    int width, height;
    cudaBuffer colorBuffer;

    void initOptix();

    void createContext();
    void createModule(const std::string &ptx, const OptixPipelineCompileOptions &pipelineCompileOptions, std::vector<OptixProgramGroup> &programGroups);
    void createPipelines();

    void buildAccel();
    void createTextures();
    void buildSBT();

public:
    Intergrator(const Model *_model, int _w, int _h);

    void render();
    void download(float4 *pixels);
};