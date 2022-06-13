#pragma once

#include <iostream>
#include <vector>
#include <string>
#include <set>
#include <map>
#include <filesystem>

#include "helper_math.h"


class TriangleMesh
{
public:
	std::vector<float3> vertex;
	std::vector<uint3> index;
	std::vector<float3> normal;
	std::vector<float2> texcoord;

	float3 diffuse;
	float3 emittance;
	int textureId{ -1 };

public:
	void addVertices(const std::vector<float3>& _v, const std::vector<uint3>& _i);
	void addCube(const float3& center, const float3& size);
	inline void setColor(const float3& _c) { diffuse = _c; }
	inline void setEmittance(const float3& _e) { emittance = _e; }
};


class Texture
{
public:
	uint32_t* pixels { nullptr };
	int2 resolution { -1, -1 };

	~Texture() { if (pixels) delete[] pixels; }
};


class Model
{
public:
	std::vector<TriangleMesh*> meshes;
	std::vector<Texture*> textures;

public:
	Model() {}
	Model(const std::string& objPath) { loadObj(objPath); }
	~Model();

	void loadObj(const std::string& objPath);
	int loadTexture(std::map<std::string, int>& knownTextures, const std::string& textureName, const std::filesystem::path& modelDir);
};