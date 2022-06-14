#pragma once

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "helper_math.h"
#include "optixHelper.h"
#include "cudaBuffer.h"
#include "Model.h"

#include "LaunchParams.h"


class ModuleDesc
{
public:
    std::string filename;
    std::string raygenName;
    std::vector<std::string> missNames;
    std::vector<std::pair<std::string, std::string> > hitgroupNames;
};


class Intergrator
{
private:
    std::vector<ModuleDesc> moduleDescs;

    CUstream stream;
    OptixDeviceContext optixContext;
    std::vector<OptixProgramGroup> raygenPGs;
    std::vector<OptixProgramGroup> missPGs;
    std::vector<OptixProgramGroup> hitgroupPGs;
    std::vector<OptixPipeline> pipelines;

    const Model* model;
    std::vector<cudaBuffer> vertexBuffer;
    std::vector<cudaBuffer> indexBuffer;
    std::vector<cudaBuffer> texcoordBuffer;
    std::vector<cudaBuffer> normalBuffer;
    
    std::vector<cudaArray_t> textureArrays;
    std::vector<cudaTextureObject_t> textureObjects;

    OptixShaderBindingTable sbt = {};
    cudaBuffer raygenRecordsBuffer;
    cudaBuffer missRecordsBuffer;
    cudaBuffer hitgroupRecordsBuffer;

    OptixTraversableHandle traversable = {};

    int width, height;
    cudaBuffer colorBuffer;

	
    void initOptix();
	
    void createContext();
    void createModule(const ModuleDesc& moduleDesc, const OptixPipelineCompileOptions& pipelineCompileOptions, std::vector<OptixProgramGroup>& programGroups);
	void createPipelines();

    void buildAccel();
	void createTextures();
	void buildSBT();

public:
    Intergrator(const Model* _model, int _w, int _h);

    void render();
    void download(float4* pixels);
};