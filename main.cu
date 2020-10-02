#include <glad/glad.h> 
#include <GLFW/glfw3.h>

#include <iostream>
#include <chrono>
#include <string>
#include <iomanip>

#include "shader.h"
#include "ShaderProgram.h"
#include "fpsCounter.h"
#include "Camera.h"

#define GLM_FORCE_CUDA
#define CUDA_VERSION 101000
#include <glm/glm.hpp>
//#include <glm/vec2.hpp>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "cuda_gl_interop.h"

#include "kernel.cuh"


// Initial resolution of window
unsigned int winWidth = 512*2;
unsigned int winHeight = 512*2;

namespace settings {
    bool g_useSuperSampling = true;
    Camera g_camera(glm::dvec3(-5., 0., 0.), 0., 0. );
    int g_iterCap = 2;
    bool g_cursorLocked = false;
}

namespace input {
    glm::dvec2 g_cursorPos; // Center of screen is (0, 0), borders are +- 1;
}

namespace screenState {
    // Resolution of render
    unsigned int hRes = 1920 * 2;
    unsigned int vRes = 1027 * 2; // Supports non-powers of two, but powers of two will be somewhat faster
    unsigned int renderTexture;
    unsigned int displayTexture;
    cudaGraphicsResource_t screenCudaResource;
}


void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
    winWidth = width;
    winHeight = height;

    cudaGraphicsUnregisterResource(screenState::screenCudaResource);

    int superSamplingFactor = settings::g_useSuperSampling ? 2 : 1;

    screenState::hRes = width * superSamplingFactor;
    screenState::vRes = height * superSamplingFactor;
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, screenState::renderTexture);

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, screenState::hRes, screenState::vRes, 0, GL_RGBA, GL_FLOAT, NULL);

    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, screenState::displayTexture);

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, screenState::hRes, screenState::vRes, 0, GL_RGBA, GL_FLOAT, NULL);

    glViewport(0, 0, width, height);

    cudaGraphicsGLRegisterImage(&screenState::screenCudaResource, screenState::renderTexture, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard);
}

void printVec(glm::dvec3 vec) {
    std::cout << '(' << vec.x << ", " << vec.y << ", " << vec.z << ')';
}

// XMousepos and yMousepos given in normalized coords, [-1, 1)
// Positive X axis is to the right, positive Y axis is upwards on the screen
void processInput(GLFWwindow* window)
{
    glm::dvec2 newMousePos;
    glfwGetCursorPos(window, &(newMousePos.x), &(newMousePos.y));
    
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) {
        settings::g_camera.walk(glm::dvec3(0., 0., -0.15));
    }
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) {
        settings::g_camera.walk(glm::dvec3(0., 0., 0.15));
    }
    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) {
        settings::g_camera.walk(glm::dvec3(0.15, 0., 0.));
    }
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) {
        settings::g_camera.walk(glm::dvec3(-0.15, 0., 0.));
    }
    if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS) {
        settings::g_camera.walk(glm::dvec3(0., 0.15, 0.));
    }
    if (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS) {
        settings::g_camera.walk(glm::dvec3(0.0, -0.15, 0.));
    }
    newMousePos.x *= 2. / winWidth;
    newMousePos.x -= 1.;
    newMousePos.y *= -2. / winHeight;
    newMousePos.y += 1.;
    
    glm::dvec2 deltaMouse = newMousePos - input::g_cursorPos;
    if (!settings::g_cursorLocked) {
        settings::g_camera.rotate(-1 * deltaMouse.x, deltaMouse.y);
    }
    input::g_cursorPos = newMousePos;
}

void keyPressCallback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    if (action == GLFW_PRESS) {
        switch (key) {
        case GLFW_KEY_ESCAPE: glfwSetWindowShouldClose(window, true); break;
        //case GLFW_KEY_S:      settings::g_useSuperSampling = !settings::g_useSuperSampling;
        //                      framebuffer_size_callback(window, winWidth, winHeight); break; // Force resizing of window for supersampling settings;
        case GLFW_KEY_LEFT: settings::g_iterCap--; break;
        case GLFW_KEY_RIGHT: settings::g_iterCap++; break;
        case GLFW_KEY_P: settings::g_cursorLocked = !settings::g_cursorLocked; break;
        }
    }

}

void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods)
{
    return;
}

void scrollCallback(GLFWwindow* window, double xoffset, double yoffset)
{
    ;//settings::g_screenBounds.zoom(settings::g_screenBounds.screenPointToWorld(input::g_cursorPos), pow(0.8, -1. * yoffset));
}

int main(void) {

    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);



    GLFWwindow* window = glfwCreateWindow(winWidth, winHeight, "RenderTest", NULL, NULL);

    if (window == NULL)
    {
        std::cout << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }

    glfwMakeContextCurrent(window);

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    {
        std::cout << "Failed to initialize GLAD" << std::endl;
        return -1;
    }

    glViewport(0, 0, winWidth, winHeight);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    glfwSetMouseButtonCallback(window, mouseButtonCallback);
    glfwSetKeyCallback(window, keyPressCallback);
    glfwSetScrollCallback(window, scrollCallback);

    //
    float vertices[] = {
         // positions     // Tex coords
         1.00f,  1.00f, 0.0f,  1.f,  1.f, // top right
         1.00f, -1.f, 0.0f,  1.f,  0.f,  // bottom right
        -1.f, -1.f, 0.0f,  0.f,  0.f,  // bottom left
        -1.f,  1.00f, 0.0f,  0.f,  1.f,   // top left 
    };
    unsigned int indices[] = {  
        0, 1, 3,   // first triangle
        1, 2, 3    // second triangle
    };
    
    unsigned int VAO;
    glGenVertexArrays(1, &VAO);
    glBindVertexArray(VAO);
    
    unsigned int VBO;
    glGenBuffers(1, &VBO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(0));
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);
    

    unsigned int EBO;
    glGenBuffers(1, &EBO);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);
    

    glGenTextures(1, &screenState::renderTexture);

   
    glBindTexture(GL_TEXTURE_2D, screenState::renderTexture);
    {
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        //float borderColor[] = { 1.0f, 1.0f, 0.0f, 1.0f };
        //glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, borderColor);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, screenState::renderTexture);


        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, screenState::hRes, screenState::vRes, 0, GL_RGBA, GL_FLOAT, NULL);

        glBindImageTexture(0, screenState::renderTexture, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32F);
    }
    glGenTextures(1, &screenState::displayTexture);
    glBindTexture(GL_TEXTURE_2D, screenState::displayTexture);
    {
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, screenState::displayTexture);


        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, screenState::hRes, screenState::vRes, 0, GL_RGBA, GL_FLOAT, NULL);

        glBindImageTexture(1, screenState::displayTexture, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32F);
    }

    // Vertex and fragment shaders are pretty much empty, and just pass through vertex/texture coord data
    Shader vertex("doNothing.vert", GL_VERTEX_SHADER);
    Shader fragment("doNothing.frag", GL_FRAGMENT_SHADER);
    ShaderProgram renderProg;
    renderProg.attach(vertex);
    renderProg.attach(fragment);
    renderProg.link();

    Shader frameBlendShader("frameBlend.comp", GL_COMPUTE_SHADER);
    ShaderProgram frameBlendProg;
    frameBlendProg.attach(frameBlendShader);
    frameBlendProg.link();

    
    

    input::g_cursorPos;

    FpsCounter fpsCounter;
    framebuffer_size_callback(window, winWidth, winHeight); // Force resizing of window for supersampling settings;

    curandState* cuRandPtr = setupRaytracer(screenState::hRes, screenState::vRes);

    while (!glfwWindowShouldClose(window))
    {
        fpsCounter.update(glfwGetTime());
        std::stringstream title;
        title << "Raytracer - FPS: " <<  std::setprecision(0) << std::setiosflags(std::ios::fixed) << fpsCounter.getFPS();
        glfwSetWindowTitle(window, title.str().c_str());

        // Input (updates global variable for mouse position)
        processInput(window);

        glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        cudaGraphicsMapResources(1, &screenState::screenCudaResource);
        {
            cudaArray_t viewCudaArray;
            cudaGraphicsSubResourceGetMappedArray(&viewCudaArray, screenState::screenCudaResource, 0, 0);
            cudaResourceDesc viewCudaArrayResourceDesc;
            {
                viewCudaArrayResourceDesc.resType = cudaResourceTypeArray;
                viewCudaArrayResourceDesc.res.array.array = viewCudaArray;
            }
            cudaSurfaceObject_t viewCudaSurfaceObject;
            cudaCreateSurfaceObject(&viewCudaSurfaceObject, &viewCudaArrayResourceDesc);
            {
                traceImage<<<screenState::hRes*screenState::vRes / 128 + 1, 128>>>(viewCudaSurfaceObject, screenState::hRes, screenState::vRes, settings::g_iterCap, settings::g_camera.getPos(), settings::g_camera.getLookDir(), settings::g_camera.getLookU(), settings::g_camera.getLookR(), cuRandPtr);
            }
            cudaDestroySurfaceObject(viewCudaSurfaceObject);
        }
        cudaGraphicsUnmapResources(1, &screenState::screenCudaResource);

        cudaStreamSynchronize(0);


        frameBlendProg.use();
        frameBlendProg.setUvec2("resolution", screenState::hRes, screenState::vRes);

        glDispatchCompute((screenState::hRes + 15) / 16, (screenState::vRes + 15) / 16, 1); // For local work group size 16. Ensures entire texture is written to

        glMemoryBarrier(GL_TEXTURE_FETCH_BARRIER_BIT);

        // Render screen-sized quad

        renderProg.use();
        //renderProg.setInt("textureVar", 1);
        glBindTexture(GL_TEXTURE_2D, screenState::displayTexture);
        glBindVertexArray(VAO);
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
        glBindTexture(GL_TEXTURE_2D, 0);

        // End drawing current frame
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);
    glDeleteBuffers(1, &EBO);
    glfwTerminate();
    return 0;
}