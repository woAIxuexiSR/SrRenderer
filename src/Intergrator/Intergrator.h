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
    std::vector<OptixPipeline> pipelines;

    const Model* model;
    std::vector<cudaBuffer> vertexBuffer;
    std::vector<cudaBuffer> indexBuffer;
    std::vector<cudaBuffer> texcoordBuffer;
    std::vector<cudaBuffer> normalBuffer;
    
    std::vector<cudaArray_t> textureArrays;
    std::vector<cudaTextureObject_t> textureObjects;
	
    void initOptix();
	
    void createContext();
    void createModule(const ModuleDesc& moduleDesc, const OptixPipelineCompileOptions& pipelineCompileOptions, std::vector<OptixProgramGroup>& programGroups);
	void createPipelines();

    OptixTraversableHandle buildAccel();
	void createTextures();
	void buildSBT();

public:
    Intergrator(const Model* _model);
};


// class Renderer
// {
// private:
// 	CUcontext cudaContext;
// 	CUstream stream;
// 	cudaDeviceProp deviceProps;

// 	OptixDeviceContext optixContext;

// 	OptixPipeline pipeline;
// 	OptixPipelineCompileOptions pipelineCompileOptions = {};
// 	OptixPipelineLinkOptions pipelineLinkOptions = {};

// 	OptixModule module;
// 	OptixModuleCompileOptions moduleCompileOptions = {};

// 	std::vector<OptixProgramGroup> raygenPGs;
// 	CUDABuffer raygenRecordsBuffer;
// 	std::vector<OptixProgramGroup> missPGs;
// 	CUDABuffer missRecordsBuffer;
// 	std::vector<OptixProgramGroup> hitgroupPGs;
// 	CUDABuffer hitgroupRecordsBuffer;

// 	OptixShaderBindingTable sbt = {};

// 	LaunchParams launchParams;
// 	CUDABuffer launchParamsBuffer;

// 	CUDABuffer colorBuffer;

// 	CUDABuffer denoisedBuffer;
// 	OptixDenoiser denoiser = nullptr;
// 	CUDABuffer denoiserScratch;
// 	CUDABuffer denoiserState;

// 	Camera lastSetCamera;

// 	const Model* model;
// 	std::vector<CUDABuffer> vertexBuffer;
// 	std::vector<CUDABuffer> indexBuffer;
// 	std::vector<CUDABuffer> texcoordBuffer;
// 	std::vector<CUDABuffer> normalBuffer;
// 	CUDABuffer asBuffer;

// 	std::vector<cudaArray_t> textureArrays;
// 	std::vector<cudaTextureObject_t> textureObjects;



// public:
// 	bool denoiserOn = true;

// 	Renderer(const Model* _model, const QuadLight* _light);
// 	void render();
// 	void resize(int _width, int _height);
// 	void downloadPixels(vec4 h_pixels[]);
// 	void setCamera(const Camera& _camera);
// };