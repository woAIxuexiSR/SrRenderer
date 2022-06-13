#include "Intergrator.h"
#include <optix_function_table_definition.h>


std::string readPtxFromFile(const std::string& filename)
{
    std::fstream fs(filename, std::ios::in);
    if (!fs.is_open())
    {
        std::cout << "Failed to open file " << filename << std::endl;
        exit(-1);
    }
    std::stringstream ss;
    ss << fs.rdbuf();
    fs.close();
    return ss.str();
}


void Intergrator::initOptix()
{
    CUDA_CHECK(cudaFree(0));

    int numDevices;
    CUDA_CHECK(cudaGetDeviceCount(&numDevices));
    if(numDevices == 0)
    {
        std::cout << "no CUDA capable devices found!" << std::endl;
        exit(-1);
    }
    std::cout << "found " << numDevices << " CUDA devices" << std::endl;

    OPTIX_CHECK(optixInit());
    std::cout << "successfully initialized optix ..." << std::endl;
}


void Intergrator::createContext()
{
    const int deviceID = 0;
    CUDA_CHECK(cudaSetDevice(deviceID));
    CUDA_CHECK(cudaStreamCreate(&stream));

    cudaDeviceProp deviceProps;
    CUDA_CHECK(cudaGetDeviceProperties(&deviceProps, deviceID));
    std::cout << "running on device " << deviceProps.name << std::endl;

    CUcontext cudaContext;
    cuCtxGetCurrent(&cudaContext);

    OPTIX_CHECK(optixDeviceContextCreate(cudaContext, 0, &optixContext));
}


void Intergrator::createModule(const ModuleDesc& moduleDesc, const OptixPipelineCompileOptions& pipelineCompileOptions, std::vector<OptixProgramGroup>& programGroups) 
{
    // create module
    OptixModule module;

    OptixModuleCompileOptions moduleCompileOptions = {};
    moduleCompileOptions.maxRegisterCount = 50;
    moduleCompileOptions.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    moduleCompileOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;

    std::string ptx = readPtxFromFile(moduleDesc.filename);


    OPTIX_LOG(optixModuleCreateFromPTX(optixContext, &moduleCompileOptions, &pipelineCompileOptions, ptx.c_str(), ptx.size(), log, &logSize, &module));

    // create raygen programs
    {
        OptixProgramGroup raygenPG;
        OptixProgramGroupOptions pgOptions = {};
        OptixProgramGroupDesc pgDesc = {};

        pgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
        pgDesc.raygen.module = module;
        pgDesc.raygen.entryFunctionName = moduleDesc.raygenName.c_str();

        OPTIX_LOG(optixProgramGroupCreate(optixContext, &pgDesc, 1, &pgOptions, log, &logSize, &raygenPG));

        programGroups.push_back(raygenPG);
    }

    // create miss programs
    {
        for(int i = 0; i < moduleDesc.missNames.size(); i++)
        {
            OptixProgramGroup missPG;
            OptixProgramGroupOptions pgOptions = {};
            OptixProgramGroupDesc pgDesc = {};

            pgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
            pgDesc.miss.module = module;
            pgDesc.miss.entryFunctionName = moduleDesc.missNames[i].c_str();

            OPTIX_LOG(optixProgramGroupCreate(optixContext, &pgDesc, 1, &pgOptions, log, &logSize, &missPG));
        
            programGroups.push_back(missPG);
        }
    }

    // create hitgroup programs
    {
        for(int i = 0; i < moduleDesc.hitgroupNames.size(); i++)
        {
            OptixProgramGroup hitgroupPG;
            OptixProgramGroupOptions pgOptions = {};
            OptixProgramGroupDesc pgDesc = {};

            pgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
            pgDesc.hitgroup.moduleCH = module;
            pgDesc.hitgroup.entryFunctionNameCH = moduleDesc.hitgroupNames[i].first.c_str();
            pgDesc.hitgroup.moduleAH = module;
            pgDesc.hitgroup.entryFunctionNameAH = moduleDesc.hitgroupNames[i].second.c_str();

            OPTIX_LOG(optixProgramGroupCreate(optixContext, &pgDesc, 1, &pgOptions, log, &logSize, &hitgroupPG));
        
            programGroups.push_back(hitgroupPG);
        }
    }
}


void Intergrator::createPipelines()
{
    OptixPipelineCompileOptions pipelineCompileOptions = {};
    pipelineCompileOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    pipelineCompileOptions.usesMotionBlur = false;
    pipelineCompileOptions.numPayloadValues = 2;
    pipelineCompileOptions.numAttributeValues = 2;
    pipelineCompileOptions.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
    pipelineCompileOptions.pipelineLaunchParamsVariableName = "optixLaunchParams";

    OptixPipelineLinkOptions pipelineLinkOptions = {};
    pipelineLinkOptions.maxTraceDepth = 2;

    pipelines.resize(moduleDescs.size());
    for(int i = 0; i < moduleDescs.size(); i++)
    {
        std::vector<OptixProgramGroup> programGroups;
        createModule(moduleDescs[i], pipelineCompileOptions, programGroups);

        OPTIX_LOG(optixPipelineCreate(optixContext, &pipelineCompileOptions, &pipelineLinkOptions, programGroups.data(), (unsigned)programGroups.size(), log, &logSize, &pipelines[i]));

        OPTIX_CHECK(optixPipelineSetStackSize(pipelines[i], 2048, 2048, 2048, 1));
    }
}


OptixTraversableHandle Intergrator::buildAccel()
{
    int meshNum = (int)model->meshes.size();

    vertexBuffer.resize(meshNum);
    indexBuffer.resize(meshNum);
    texcoordBuffer.resize(meshNum);
    normalBuffer.resize(meshNum);

    OptixTraversableHandle asHandle { 0 };

    std::vector<OptixBuildInput> triangleInput(meshNum);
    std::vector<CUdeviceptr> d_vertices(meshNum);
    std::vector<CUdeviceptr> d_indices(meshNum);
    std::vector<uint32_t> triangleInputFlags(meshNum);

    for(int i = 0; i < meshNum; i++)
    {
        TriangleMesh& mesh = *(model->meshes[i]);
        vertexBuffer[i].upload(mesh.vertex);
        indexBuffer[i].upload(mesh.index);
        if(!mesh.texcoord.empty()) texcoordBuffer[i].upload(mesh.texcoord);
        if(!mesh.normal.empty()) normalBuffer[i].upload(mesh.normal);

        triangleInput[i] = {};
        triangleInput[i].type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;

        d_vertices[i] = (CUdeviceptr)vertexBuffer[i].d_ptr;
        d_indices[i] = (CUdeviceptr)indexBuffer[i].d_ptr;

        triangleInput[i].triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
        triangleInput[i].triangleArray.vertexStrideInBytes = sizeof(float3);
        triangleInput[i].triangleArray.numVertices = (unsigned)mesh.vertex.size();
        triangleInput[i].triangleArray.vertexBuffers = &d_vertices[i];

        triangleInput[i].triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
        triangleInput[i].triangleArray.indexStrideInBytes = sizeof(uint3);
        triangleInput[i].triangleArray.numIndexTriplets = (unsigned)mesh.index.size();
        triangleInput[i].triangleArray.indexBuffer = d_indices[i];

        triangleInputFlags[i] = 0;
        triangleInput[i].triangleArray.flags = &triangleInputFlags[i];
        triangleInput[i].triangleArray.numSbtRecords = 1;
        triangleInput[i].triangleArray.sbtIndexOffsetBuffer = 0;
        triangleInput[i].triangleArray.sbtIndexOffsetSizeInBytes = 0;
        triangleInput[i].triangleArray.sbtIndexOffsetStrideInBytes = 0;
    }

    OptixAccelBuildOptions accelOptions = {};
    accelOptions.buildFlags = OPTIX_BUILD_FLAG_NONE | OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
    accelOptions.motionOptions.numKeys = 1;
    accelOptions.operation = OPTIX_BUILD_OPERATION_BUILD;

    OptixAccelBufferSizes blasBufferSizes;
    OPTIX_CHECK(optixAccelComputeMemoryUsage(optixContext, &accelOptions, triangleInput.data(), meshNum, &blasBufferSizes));

    cudaBuffer compactedSizeBuffer;
    compactedSizeBuffer.alloc(sizeof(uint64_t));

    OptixAccelEmitDesc emitDesc;
    emitDesc.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
    emitDesc.result = (CUdeviceptr)compactedSizeBuffer.d_ptr;

    cudaBuffer tempBuffer;
    tempBuffer.alloc(blasBufferSizes.tempSizeInBytes);
    cudaBuffer outputBuffer;
    outputBuffer.alloc(blasBufferSizes.outputSizeInBytes);

    OPTIX_CHECK(optixAccelBuild(
        optixContext, 
        stream, 
        &accelOptions, 
        triangleInput.data(), 
        meshNum, 
        (CUdeviceptr)tempBuffer.d_ptr, 
        tempBuffer.size, 
        (CUdeviceptr)outputBuffer.d_ptr, 
        outputBuffer.size, 
        &asHandle, 
        &emitDesc, 
        1
    ));
    CUDA_SYNC_CHECK();

    uint64_t compactedSize;
    compactedSizeBuffer.download(&compactedSize);

    cudaBuffer asBuffer;
    asBuffer.alloc(compactedSize);
    OPTIX_CHECK(optixAccelCompact(optixContext, stream, asHandle, (CUdeviceptr)asBuffer.d_ptr, asBuffer.size, &asHandle));
    CUDA_SYNC_CHECK();
    
    return asHandle;
}


void Intergrator::createTextures() 
{
    int numTextures = model->textures.size();

    textureArrays.resize(numTextures);
    textureObjects.resize(numTextures);

    for(int i = 0; i < numTextures; i++)
    {
        Texture* texture = model->textures[i];

        cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc<uchar4>();
        
        int width = texture->resolution.x;
        int height = texture->resolution.y;
        int nComponents = 4;
        int pitch = width * nComponents * sizeof(unsigned char);

        cudaArray_t &pixelArray = textureArrays[i];
        CUDA_CHECK(cudaMallocArray(&pixelArray, &channel_desc, width, height));
        CUDA_CHECK(cudaMemcpy2DToArray(pixelArray, 0, 0, texture->pixels, pitch, pitch, height, cudaMemcpyHostToDevice));
    
        cudaResourceDesc res_desc = {};
        res_desc.resType = cudaResourceTypeArray;
        res_desc.res.array.array = pixelArray;

        cudaTextureDesc tex_desc = {};
        tex_desc.addressMode[0] = cudaAddressModeWrap;
        tex_desc.addressMode[1] = cudaAddressModeWrap;
        tex_desc.filterMode = cudaFilterModeLinear;
        tex_desc.readMode = cudaReadModeNormalizedFloat;
        tex_desc.normalizedCoords = 1;
        tex_desc.maxAnisotropy = 1;
        tex_desc.maxMipmapLevelClamp = 99;
        tex_desc.minMipmapLevelClamp = 0;
        tex_desc.mipmapFilterMode = cudaFilterModePoint;
        tex_desc.borderColor[0] = 1.0f;
        tex_desc.sRGB = 0;

        cudaTextureObject_t cuda_tex = 0;
        CUDA_CHECK(cudaCreateTextureObject(&cuda_tex, &res_desc, &tex_desc, nullptr));
        textureObjects[i] = cuda_tex;
    }
}



struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) RaygenRecord
{
    __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    void* data;
};

struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) MissRecord
{
    __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    void* data;
};

struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) HitgroupRecord
{
    __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    // TriangleMeshSBTData data;
};

void Intergrator::buildSBT()
{
    std::vector<RaygenRecord> raygenRecords;
}



Intergrator::Intergrator(const Model* _model) : model(_model)
{
    
    moduleDescs.push_back(
        {
            "../../../src/Intergrator/shader/bdpt.ptx",
            "__raygen__renderFrame",
            {"__miss__radiance", "__miss__shadow"},
            {
                std::make_pair("__closesthit__radiance", "__anyhit__radiance"),
                std::make_pair("__closesthit__shadow", "__anyhit__shadow")
            }
        }
        
    );

    std::cout << "initializing optix ..." << std::endl;
    initOptix();

    std::cout << "creating optix context ..." << std::endl;
    createContext();

    std::cout << "setting up optix pipline ..." << std::endl;
    createPipelines();

    std::cout << "building acceleration structure ..." << std::endl;
    buildAccel();

    std::cout << "creating textures ..." << std::endl;
    createTextures();

    std::cout << "building SBT ..." << std::endl;
    buildSBT();

    std::cout << "Optix 7 Renderer fully set up!" << std::endl;
}