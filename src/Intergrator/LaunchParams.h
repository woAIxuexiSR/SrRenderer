#pragma once

struct LaunchParams
{
    int frameID{ 0 };
    float4* colorBuffer;
    int width, height;
	OptixTraversableHandle traversable;
    
    LaunchParams(int _w, int _h, float4* _c, OptixTraversableHandle _t)
        : width(_w), height(_h), colorBuffer(_c), traversable(_t) {}
};