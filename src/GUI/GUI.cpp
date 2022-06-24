#include "GUI.h"


void framebuffer_size_callback(GLFWwindow* window, int w, int h)
{
    glViewport(0, 0, w, h);
}


unsigned int GUI::loadShader(GLenum type, std::string filepath)
{
    std::fstream fs(filepath, std::ios::in);
    if(!fs.is_open())
    {
        std::cout << "Failed to open shader file " << filepath << std::endl;
        exit(-1);
    }

    std::stringstream ss;
    ss << fs.rdbuf();
    std::string str = ss.str();
    fs.close();

    const char* shaderSource = str.c_str();

    unsigned int shaderId;
    shaderId = glCreateShader(type);
    glShaderSource(shaderId, 1, &shaderSource, nullptr);
    glCompileShader(shaderId);

    int success;
    char infoLog[512];
    glGetShaderiv(shaderId, GL_COMPILE_STATUS, &success);
    if(!success)
    {
        glGetShaderInfoLog(shaderId, 512, nullptr, infoLog);
        std::cout << "Compile shader file " << filepath << " error!" << std::endl;
        std::cout << infoLog << std::endl;
        exit(-1);
    }

    return shaderId;
}


void GUI::createProgram(std::string vertexPath, std::string fragmentPath)
{
    unsigned int vertexShader, fragmentShader;
    vertexShader = loadShader(GL_VERTEX_SHADER, vertexPath);
    fragmentShader = loadShader(GL_FRAGMENT_SHADER, fragmentPath);

    programId = glCreateProgram();
    glAttachShader(programId, vertexShader);
    glAttachShader(programId, fragmentShader);
    glLinkProgram(programId);

    int success;
    char infoLog[512];
    glGetProgramiv(programId, GL_LINK_STATUS, &success);
    if(!success)
    {
        glGetProgramInfoLog(programId, 512, nullptr, infoLog);
        std::cout << "Failed to link program!" << std:: endl;
        std::cout << infoLog << std::endl;
        exit(-1);
    }

    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);
}


void GUI::createVAO()
{
    float quadVertices[] = 
    {
        -1.0f,  1.0f,  0.0f, 1.0f,
        -1.0f, -1.0f,  0.0f, 0.0f,
         1.0f, -1.0f,  1.0f, 0.0f,

        -1.0f,  1.0f,  0.0f, 1.0f,
         1.0f, -1.0f,  1.0f, 0.0f,
         1.0f,  1.0f,  1.0f, 1.0f
    };
    int size = sizeof(quadVertices);

	glGenVertexArrays(1, &vao);
	glBindVertexArray(vao);

	unsigned int vbo;
	glGenBuffers(1, &vbo);
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glBufferData(GL_ARRAY_BUFFER, size, quadVertices, GL_STATIC_DRAW);

	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float)));
	glEnableVertexAttribArray(1);

	glBindVertexArray(0);
	glDeleteBuffers(1, &vbo);
}


unsigned int GUI::loadTexture(const std::vector<float4>& pixels, int pw, int ph)
{
	unsigned int texture;
	glGenTextures(1, &texture);
	glBindTexture(GL_TEXTURE_2D, texture);

	glTextureParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTextureParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	glTextureParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
	glTextureParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

	GLenum format = GL_RGBA;

	glTexImage2D(GL_TEXTURE_2D, 0, format, pw, ph, 0, GL_RGBA, GL_FLOAT, pixels.data());
	glGenerateMipmap(GL_TEXTURE_2D);

	return texture;
}


GUI::GUI(int w, int h, GLFWcursorposfun cursor_pos_callback, GLFWscrollfun scroll_callback) : width(w), height(h)
{
    glfwInit();
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

	window = glfwCreateWindow(width, height, "SR Renderer", nullptr, nullptr);
	if (!window)
	{
		std::cout << "Failed to create window!" << std::endl;
		glfwTerminate();
		exit(-1);
	}
	glfwMakeContextCurrent(window);

	if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
	{
		std::cout << "Failed to initialize glad!" << std::endl;
		exit(-1);
	}

	glViewport(0, 0, width, height);
	glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    if(cursor_pos_callback != nullptr)
        glfwSetCursorPosCallback(window, cursor_pos_callback);
    if(scroll_callback != nullptr)
	    glfwSetScrollCallback(window, scroll_callback);
    glfwSwapInterval(0);

	IMGUI_CHECKVERSION();
	ImGui::CreateContext();
    ImPlot::CreateContext();
	ImGuiIO& io = ImGui::GetIO(); (void)io;
	ImGui::StyleColorsDark();
	ImGui_ImplGlfw_InitForOpenGL(window, true);
	const char* glsl_version = "#version 460";
	ImGui_ImplOpenGL3_Init(glsl_version);

    std::filesystem::path p(__FILE__);
	auto shaderPath = p.parent_path();
	auto vertexShaderPath = shaderPath / "hello.vert";
	auto fragmentShaderPath = shaderPath / "hello.frag";
    createProgram(vertexShaderPath.string(), fragmentShaderPath.string());

	createVAO();
}


GUI::~GUI()
{
    glDeleteProgram(programId);
    glDeleteVertexArrays(1, &vao);

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImPlot::DestroyContext();
    ImGui::DestroyContext();

    glfwDestroyWindow(window);
    glfwTerminate();
}


bool GUI::shouldClose()
{
    return glfwWindowShouldClose(window);
}


void GUI::run(const std::vector<float4>& pixels, int pw, int ph)
{
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    ImGui::Begin("SR Renderer");
    ImGui::Text("Application Time %.1f s", glfwGetTime());
    ImGui::Text("Application average %.1f FPS", ImGui::GetIO().Framerate);


    unsigned texture = loadTexture(pixels, pw, ph);
    // ImTextureID image_id = (GLuint*)texture;
    // ImGui::Image(image_id, ImVec2((float)pw, (float)ph));

    // cudaArray *texture_ptr;
    // cudaGraphicsResource* cuda_tex_result_resource;
    // checkCudaErrors(cudaGraphicsMapResources(1, &cuda_tex_result_resource, 0));
    // checkCudaErrors(cudaGraphicsSubResourceGetMappedArray(
    //     &texture_ptr, cuda_tex_result_resource, 0, 0));

    // int num_texels = pw * ph;
    // int num_values = num_texels * 4;
    // int size_tex_data = sizeof(GLubyte) * num_values;
    // unsigned int* cuda_dest_resource;
    // checkCudaErrors(cudaMemcpyToArray(texture_ptr, 0, 0, cuda_dest_resource,
    //                                     size_tex_data, cudaMemcpyDeviceToDevice));

    // checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_tex_result_resource, 0));


    ImGui::End();

    // ImPlot::ShowDemoWindow();

    ImGui::Render();

    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    

    glUseProgram(programId);
    glBindVertexArray(vao);
    glBindTexture(GL_TEXTURE_2D, texture);
    glDrawArrays(GL_TRIANGLES, 0, 6);

    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

    glDeleteTextures(1, &texture);

    glfwSwapBuffers(window);
    glfwPollEvents();
}