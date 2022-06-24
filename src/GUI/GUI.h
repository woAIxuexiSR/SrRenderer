#pragma once

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include "imgui.h"
#include "imgui_impl_opengl3.h"
#include "imgui_impl_glfw.h"
#include "implot.h"
#include "implot_internal.h"

#include <cuda_runtime.h>
#include "helper_cuda.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <filesystem>

class GUI
{
private:
    unsigned int programId;
    unsigned int vao;

    unsigned int loadShader(GLenum type, std::string filepath);
    unsigned int loadTexture(const std::vector<float4>& pixels, int pw, int ph);

    void createProgram(std::string vertexPath, std::string fragmentPath);
    void createVAO();

public:
    GLFWwindow* window;
    int width, height;

    GUI(int w, int h, GLFWcursorposfun cursor_pos_callback = nullptr, GLFWscrollfun scroll_callback = nullptr);
    ~GUI();
    bool shouldClose();
    void run(const std::vector<float4>& pixels, int pw, int ph);

};