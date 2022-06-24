#include <iostream>
#include <vector>
#include "GUI.h"
#include "Model.h"

#include "Intergrator.h"

#include "helper_math.h"

void processInput(GLFWwindow *window)
{
    if(glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);
}

int main()

{
        float3 a = make_float3(1, 2, 3);
    std::cout << a.x << " " << a.y << " " << a.z << std::endl;

    TriangleMesh* mesh = new TriangleMesh;
    mesh->addCube(make_float3(0, 0, 5), make_float3(1, 1, 1));
    Model* model = new Model;
    model->meshes.push_back(mesh);

    const int width = 800, height = 600;
    GUI w(width, height);

    Intergrator i(model, width, height);

    std::vector<float4> v(width * height);
    for(int i = 0; i < width; i++)
        for(int j = 0; j < height; j++)
        {
            int idx = (i + j * width);
            v[idx] = make_float4(i / (float)width, j / (float)height, 0, 1);
        }

    while(!w.shouldClose())
    {
        processInput(w.window);
        i.render();
        i.download(v.data());
        w.run(v, width, height);
    }

    return 0;
}