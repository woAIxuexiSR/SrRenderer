#include <iostream>
#include <vector>
#include "Window.h"

void processInput(GLFWwindow *window)
{
    if(glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);
}

int main()
{
    const int width = 800, height = 600;
    Window w(width, height);

    std::vector<float> v(width * height * 4, 0);
    for(int i = 0; i < width; i++)
        for(int j = 0; j < height; j++)
        {
            int idx = (i + j * width) * 4;
            v[idx] = i / (float)width;
            v[idx + 1] = j / (float)height;
            v[idx + 2] = 0;
            v[idx + 3] = 1;
        }

    while(!w.shouldClose())
    {
        processInput(w.window);
        w.run(v, width, height);
    }

    return 0;
}