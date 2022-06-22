#include "Intergrator.h"
#include <optix_function_table_definition.h>

extern "C" const char simple[];

/* the only allowed shader function names */
const std::string raygenName = "__raygen__";
const std::string missName[2] = {"__miss__radiance", "__miss__shadow"};
const std::pair<std::string, std::string> hitgroupName[2] = {
    {"__closesthit__radiance", "__anyhit__radiance"},
    {"__closesthit__shadow", "__anyhit__shadow"}};

void Intergrator::initOptix()
{
    checkCudaErrors(cudaFree(0));

    int numDevices;
    checkCudaErrors(cudaGetDeviceCount(&numDevices));
    if (numDevices == 0)
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
    checkCudaErrors(cudaSetDevice(deviceID));
    checkCudaErrors(cudaStreamCreate(&stream));

    cudaDeviceProp deviceProps;
    checkCudaErrors(cudaGetDeviceProperties(&deviceProps, deviceID));
    std::cout << "running on device " << deviceProps.name << std::endl;

    CUcontext cudaContext;
    cuCtxGetCurrent(&cudaContext);

    OPTIX_CHECK(optixDeviceContextCreate(cudaContext, 0, &optixContext));
}

void Intergrator::createModule(const std::string &ptx, const OptixPipelineCompileOptions &pipelineCompileOptions, std::vector<OptixProgramGroup> &programGroups)
{
    // create module
    OptixModule module;

    OptixModuleCompileOptions moduleCompileOptions = {};
    moduleCompileOptions.maxRegisterCount = 50;
    moduleCompileOptions.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    moduleCompileOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;

    OPTIX_LOG(optixModuleCreateFromPTX(optixContext, &moduleCompileOptions, &pipelineCompileOptions, ptx.c_str(), ptx.size(), log, &logSize, &module));

    ModuleProgramGroup modulePG;

    // create raygen programs
    {
        OptixProgramGroupOptions pgOptions = {};
        OptixProgramGroupDesc pgDesc = {};

        pgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
        pgDesc.raygen.module = module;
        pgDesc.raygen.entryFunctionName = raygenName.c_str();

        OPTIX_LOG(optixProgramGroupCreate(optixContext, &pgDesc, 1, &pgOptions, log, &logSize, &modulePG.raygenPG));

        programGroups.push_back(modulePG.raygenPG);
    }

    // create miss programs
    {
        for (int i = 0; i < 2; i++)
        {
            OptixProgramGroupOptions pgOptions = {};
            OptixProgramGroupDesc pgDesc = {};

            pgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
            pgDesc.miss.module = module;
            pgDesc.miss.entryFunctionName = missName[i].c_str();

            OPTIX_LOG(optixProgramGroupCreate(optixContext, &pgDesc, 1, &pgOptions, log, &logSize, &modulePG.missPGs[i]));

            programGroups.push_back(modulePG.missPGs[i]);
        }
    }

    // create hitgroup programs
    {
        for (int i = 0; i < 2; i++)
        {
            OptixProgramGroupOptions pgOptions = {};
            OptixProgramGroupDesc pgDesc = {};

            pgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
            pgDesc.hitgroup.moduleCH = module;
            pgDesc.hitgroup.entryFunctionNameCH = hitgroupName[i].first.c_str();
            pgDesc.hitgroup.moduleAH = module;
            pgDesc.hitgroup.entryFunctionNameAH = hitgroupName[i].second.c_str();

            OPTIX_LOG(optixProgramGroupCreate(optixContext, &pgDesc, 1, &pgOptions, log, &logSize, &modulePG.hitgroupPGs[i]));

            programGroups.push_back(modulePG.hitgroupPGs[i]);
        }
    }

    modulePGs.push_back(modulePG);
}

void Intergrator::createPipelines()
{
    OptixPipelineCompileOptions pipelineCompileOptions = {};
    pipelineCompileOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    pipelineCompileOptions.usesMotionBlur = false;
    pipelineCompileOptions.numPayloadValues = 2;
    pipelineCompileOptions.numAttributeValues = 2;
    pipelineCompileOptions.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
    pipelineCompileOptions.pipelineLaunchParamsVariableName = "launchParams";

    OptixPipelineLinkOptions pipelineLinkOptions = {};
    pipelineLinkOptions.maxTraceDepth = 2;

    pipelines.resize(modulePTXs.size());
    for (size_t i = 0; i < modulePTXs.size(); i++)
    {
        std::vector<OptixProgramGroup> programGroups;
        createModule(modulePTXs[i], pipelineCompileOptions, programGroups);

        OPTIX_LOG(optixPipelineCreate(optixContext, &pipelineCompileOptions, &pipelineLinkOptions, programGroups.data(), (unsigned)programGroups.size(), log, &logSize, &pipelines[i]));

        OPTIX_CHECK(optixPipelineSetStackSize(pipelines[i], 2048, 2048, 2048, 1));
    }
}

void Intergrator::buildAccel()
{
    int meshNum = (int)model->meshes.size();

    vertexBuffer.resize(meshNum);
    indexBuffer.resize(meshNum);
    texcoordBuffer.resize(meshNum);
    normalBuffer.resize(meshNum);

    std::vector<OptixBuildInput> triangleInput(meshNum);
    std::vector<CUdeviceptr> d_vertices(meshNum);
    std::vector<CUdeviceptr> d_indices(meshNum);
    std::vector<uint32_t> triangleInputFlags(meshNum);

    for (int i = 0; i < meshNum; i++)
    {
        TriangleMesh &mesh = *(model->meshes[i]);
        vertexBuffer[i].upload(mesh.vertex);
        indexBuffer[i].upload(mesh.index);
        if (!mesh.texcoord.empty())
            texcoordBuffer[i].upload(mesh.texcoord);
        if (!mesh.normal.empty())
            normalBuffer[i].upload(mesh.normal);

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
        &traversable,
        &emitDesc,
        1));
    checkCudaErrors(cudaDeviceSynchronize());

    uint64_t compactedSize;
    compactedSizeBuffer.download(&compactedSize);

    cudaBuffer asBuffer;
    asBuffer.alloc(compactedSize);
    OPTIX_CHECK(optixAccelCompact(optixContext, stream, traversable, (CUdeviceptr)asBuffer.d_ptr, asBuffer.size, &traversable));
    checkCudaErrors(cudaDeviceSynchronize());
}

void Intergrator::createTextures()
{
    int numTextures = (int)model->textures.size();

    textureArrays.resize(numTextures);
    textureObjects.resize(numTextures);

    for (int i = 0; i < numTextures; i++)
    {
        Texture *texture = model->textures[i];

        cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc<uchar4>();

        int width = texture->resolution.x;
        int height = texture->resolution.y;
        int nComponents = 4;
        int pitch = width * nComponents * sizeof(unsigned char);

        cudaArray_t &pixelArray = textureArrays[i];
        checkCudaErrors(cudaMallocArray(&pixelArray, &channel_desc, width, height));
        checkCudaErrors(cudaMemcpy2DToArray(pixelArray, 0, 0, texture->pixels, pitch, pitch, height, cudaMemcpyHostToDevice));

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
        checkCudaErrors(cudaCreateTextureObject(&cuda_tex, &res_desc, &tex_desc, nullptr));
        textureObjects[i] = cuda_tex;
    }
}

void Intergrator::buildSBT()
{
    sbts.resize(modulePGs.size());
    raygenRecordsBuffer.resize(modulePGs.size());
    missRecordsBuffer.resize(modulePGs.size());
    hitgroupRecordsBuffer.resize(modulePGs.size());

    for (int i = 0; i < modulePGs.size(); i++)
    {
        RaygenSBTRecord raygenRec;
        OPTIX_CHECK(optixSbtRecordPackHeader(modulePGs[i].raygenPG, &raygenRec));
        raygenRec.data.background = make_float3(0.0f, 0.0f, 0.0f);
        raygenRecordsBuffer[i].upload(&raygenRec, 1);
        sbts[i].raygenRecord = (CUdeviceptr)raygenRecordsBuffer[i].d_ptr;

        std::vector<MissSBTRecord> missRecs;
        for (int j = 0; j < 2; j++)
        {
            MissSBTRecord rec;
            OPTIX_CHECK(optixSbtRecordPackHeader(modulePGs[i].missPGs[j], &rec));
            missRecs.push_back(rec);
        }
        missRecordsBuffer[i].upload(missRecs);
        sbts[i].missRecordBase = (CUdeviceptr)missRecordsBuffer[i].d_ptr;
        sbts[i].missRecordStrideInBytes = sizeof(MissSBTRecord);
        sbts[i].missRecordCount = (unsigned)missRecs.size();

        int meshNum = (int)model->meshes.size();
        std::vector<HitgroupSBTRecord> hitgroupRecs;
        for (int k = 0; k < meshNum; k++)
        {
            for (int j = 0; j < 2; j++)
            {
                HitgroupSBTRecord rec;
                OPTIX_CHECK(optixSbtRecordPackHeader(modulePGs[i].hitgroupPGs[j], &rec));

                rec.data.vertex = (float3 *)vertexBuffer[k].d_ptr;
                rec.data.index = (uint3 *)indexBuffer[k].d_ptr;
                rec.data.normal = (float3 *)normalBuffer[k].d_ptr;
                rec.data.texcoord = (float2 *)texcoordBuffer[k].d_ptr;
                rec.data.color = model->meshes[k]->diffuse;
                if (length(model->meshes[k]->emittance) > 1e-5)
                {
                    rec.data.isLight = true;
                    rec.data.emittance = model->meshes[k]->emittance;
                }
                else
                    rec.data.isLight = false;
                if (model->meshes[k]->textureId >= 0)
                {
                    rec.data.hasTexture = true;
                    rec.data.texture = textureObjects[model->meshes[k]->textureId];
                }
                else
                    rec.data.hasTexture = false;

                hitgroupRecs.push_back(rec);
            }
        }
        hitgroupRecordsBuffer[i].upload(hitgroupRecs);
        sbts[i].hitgroupRecordBase = (CUdeviceptr)hitgroupRecordsBuffer[i].d_ptr;
        sbts[i].hitgroupRecordStrideInBytes = sizeof(HitgroupSBTRecord);
        sbts[i].hitgroupRecordCount = (unsigned)hitgroupRecs.size();
    }
}

Intergrator::Intergrator(const Model *_model, int _w, int _h) : model(_model), width(_w), height(_h)
{

    modulePTXs.push_back(simple);

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

void Intergrator::render()
{
    colorBuffer.alloc(width * height * sizeof(float4));

    LaunchParams launchParams(width, height, (float4 *)colorBuffer.d_ptr, traversable);
    cudaBuffer launchParamsBuffer;
    launchParamsBuffer.upload(&launchParams, 1);

    OPTIX_CHECK(optixLaunch(
        pipelines[0],
        stream,
        (CUdeviceptr)launchParamsBuffer.d_ptr,
        launchParamsBuffer.size,
        &sbts[0],
        width,
        height,
        1));

    checkCudaErrors(cudaDeviceSynchronize());
}

void Intergrator::download(float4 *pixels)
{
    colorBuffer.download(pixels);
}