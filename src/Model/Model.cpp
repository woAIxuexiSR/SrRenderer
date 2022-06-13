#include "Model.h"

#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

void TriangleMesh::addVertices(const std::vector<float3>& _v, const std::vector<uint3>& _i)
{
    unsigned firstVertexID = (unsigned)vertex.size();
    vertex.insert(vertex.end(), _v.begin(), _v.end());

    uint3 offset = make_uint3(firstVertexID);
    for(const auto& i : _i)
        index.push_back(i + offset);
}

void TriangleMesh::addCube(const float3& center, const float3& size)
{
    float3 leftBackDown = center - size / 2.0f;
    float3 rightFrontUp = center + size / 2.0f;

    unsigned firstVertexID = (unsigned)vertex.size();

    vertex.push_back(leftBackDown);
    vertex.push_back(make_float3(rightFrontUp.x, leftBackDown.y, leftBackDown.z));
    vertex.push_back(make_float3(leftBackDown.x, rightFrontUp.y, leftBackDown.z));
    vertex.push_back(make_float3(rightFrontUp.x, rightFrontUp.y, leftBackDown.z));
    vertex.push_back(make_float3(leftBackDown.x, leftBackDown.y, rightFrontUp.z));
    vertex.push_back(make_float3(rightFrontUp.x, leftBackDown.y, rightFrontUp.z));
    vertex.push_back(make_float3(leftBackDown.x, rightFrontUp.y, rightFrontUp.z));
    vertex.push_back(rightFrontUp);

    unsigned indices[] = 
    {
        0, 1, 3, 2, 3, 0,
        5, 7, 6, 5, 6, 4,
        0, 4, 5, 0, 5, 1,
        2, 3, 7, 2, 7, 6,
        1, 5, 7, 1, 7, 3,
        4, 0, 2, 4, 2, 6
    };

    uint3 offset = make_uint3(firstVertexID);
    for(int i = 0; i < 12; i++)
        index.push_back(offset + make_uint3(indices[3 * i + 0], indices[3 * i + 1], indices[3 * i + 2]));
}


namespace std
{
    inline bool operator<(const tinyobj::index_t& _a, const tinyobj::index_t& _b)
    {
        if(_a.vertex_index == _b.vertex_index && _a.normal_index == _b.normal_index)
            return _a.texcoord_index < _b.texcoord_index;
        if(_a.vertex_index == _b.vertex_index)
            return _a.normal_index < _b.normal_index;
        return _a.vertex_index < _b.vertex_index;
    }
}


int addVertex(TriangleMesh* mesh, tinyobj::attrib_t& attributes, tinyobj::index_t& idx, std::map<tinyobj::index_t, int>& knownVertices)
{
    if(knownVertices.find(idx) != knownVertices.end())
        return knownVertices[idx];

    int newIdx = (int)mesh->vertex.size();
    knownVertices[idx] = newIdx;

    const float3* vertex_array = (const float3*) attributes.vertices.data();
    const float3* normal_array = (const float3*) attributes.normals.data();
    const float2* texcoord_array = (const float2*) attributes.texcoords.data();

    mesh->vertex.push_back(vertex_array[idx.vertex_index]);
    if(idx.normal_index >= 0) mesh->normal.push_back(normal_array[idx.normal_index]);
    if(idx.texcoord_index >= 0) mesh->texcoord.push_back(texcoord_array[idx.texcoord_index]);

    return newIdx;
}


void Model::loadObj(const std::string& objPath)
{
    stbi_set_flip_vertically_on_load(true);

    std::filesystem::path p(objPath);
    std::filesystem::path modelDir = p.parent_path();

    tinyobj::attrib_t attributes;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;
    std::string warn, err;

    bool result = tinyobj::LoadObj(&attributes, &shapes, &materials, &warn, &err, objPath.c_str(), modelDir.string().c_str());
    if(!result)
    {
        std::cout << "Failed to load Obj File" << objPath << " " << err << std::endl;
        exit(-1);
    }
    if(materials.empty())
    {
        std::cout << "No materials found" << std::endl;
        exit(-1);
    }

    std::cout << "Done loading obj file " << objPath << " - found " << shapes.size() << " shapes with " << materials.size() << " materials." << std::endl;

    std::map<std::string, int> knownTextures;
    for(size_t shapeID = 0; shapeID < shapes.size(); shapeID++)
    {
        tinyobj::shape_t& shape = shapes[shapeID];

        std::set<int> materialIDs;
        for(auto id : shape.mesh.material_ids)
            materialIDs.insert(id);
        
        std::map<tinyobj::index_t, int> knownVertices;
        for(int materialID : materialIDs)
        {
            TriangleMesh* mesh = new TriangleMesh;
            for(int faceID = 0; faceID < shape.mesh.material_ids.size(); faceID++)
            {
                if(shape.mesh.material_ids[faceID] != materialID) continue;

                tinyobj::index_t idx0 = shape.mesh.indices[3 * faceID + 0];
                tinyobj::index_t idx1 = shape.mesh.indices[3 * faceID + 1];
                tinyobj::index_t idx2 = shape.mesh.indices[3 * faceID + 2];

                uint3 idx = make_uint3(addVertex(mesh, attributes, idx0, knownVertices),
                                       addVertex(mesh, attributes, idx1, knownVertices),
                                       addVertex(mesh, attributes, idx2, knownVertices));
                mesh->index.push_back(idx);
            }
            mesh->diffuse = (const float3&)materials[materialID].diffuse;
            mesh->textureId = loadTexture(knownTextures, materials[materialID].diffuse_texname, modelDir);
        
            if(mesh->vertex.empty()) delete mesh;
            else meshes.push_back(mesh);
        }
    }

    std::cout << "create " << meshes.size() << " meshes." << std::endl;
}


int Model::loadTexture(std::map<std::string, int>& knownTextures, const std::string& textureName, const std::filesystem::path& modelDir)
{
    if(textureName == "") return -1;
    if(knownTextures.find(textureName) != knownTextures.end()) return knownTextures[textureName];

    std::string fileName = (modelDir / textureName).string();

    int2 resolution;
    int nComponents;
    unsigned char* image = stbi_load(fileName.c_str(), &resolution.x, &resolution.y, &nComponents, STBI_rgb_alpha);

    if(!image)
    {
        std::cout << "Failed to load texture " << fileName << std::endl;
        exit(-1);
    }

    int newIdx = (int)textures.size();
    knownTextures[textureName] = newIdx;

    Texture* texture = new Texture;
    texture->pixels = (uint32_t*)image;
    texture->resolution = resolution;

    textures.push_back(texture);

    return newIdx;
}


Model::~Model()
{
    for(auto mesh : meshes) delete mesh;
    for(auto texture : textures) delete texture;
}