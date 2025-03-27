#include <iostream>
#include <vector>
#include <cmath>

// GLFW
#include <GLFW/glfw3.h>

// ImGui
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"

// Function prototypes
void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void processInput(GLFWwindow* window);

// Window settings
const unsigned int SCR_WIDTH = 1200;
const unsigned int SCR_HEIGHT = 800;

// Model parameters with default values
float kappa = 1.5f;
float eta = 0.04f;
float sigma = 0.3f;
float rho = -0.9f;
float v0 = 0.04f;

// View rotation
float rotationX = 30.0f;
float rotationY = 45.0f;

// Simple function to generate a placeholder volatility surface
std::vector<std::vector<float>> generatePlaceholderSurface(int width, int height, float kappa, float sigma, float rho) {
    std::vector<std::vector<float>> surface(width, std::vector<float>(height));
    
    for (int i = 0; i < width; i++) {
        float x = (float)i / width * 2.0f - 1.0f; // map to [-1, 1]
        for (int j = 0; j < height; j++) {
            float y = (float)j / height * 2.0f - 1.0f; // map to [-1, 1]
            
            // More extreme surface with smile/skew effects and peaks
            float base = 0.15f; // Base volatility
            float skew = 0.08f * rho * x; // Skew effect
            float smile = 0.12f * sigma * (x*x); // Smile/convexity
            float term_structure = 0.05f * (1.0f - exp(-kappa * y*y)); // Term structure
            float peak = 0.2f * exp(-8.0f * ((x-0.3f)*(x-0.3f) + (y+0.2f)*(y+0.2f))); // Sharp peak
            
            // Combine effects
            float value = std::pow(base + skew + smile + term_structure + peak + eta,2);
            
            // Add some randomness for realism
            value += 0.01f * ((float)rand() / RAND_MAX - 0.5f);
            
            surface[i][j] = value;
        }
    }
    
    return surface;
}

// Render the surface
void renderSurface(const std::vector<std::vector<float>>& surface, float scale) {
    int width = surface.size();
    int height = surface[0].size();
    
    // Set up the viewport for our 3D surface
    glViewport(400, 100, 700, 500);
    glEnable(GL_DEPTH_TEST);
    
    // Set up projection
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(-1.5, 1.5, -1.5, 1.5, -2.0, 2.0);
    
    // Set up modelview
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    
    // Apply rotation
    glRotatef(rotationX, 1.0f, 0.0f, 0.0f);
    glRotatef(rotationY, 0.0f, 1.0f, 0.0f);
    
    // Simple rendering of the surface as lines
    glBegin(GL_LINES);
    
    // Draw lines along width
    for (int i = 0; i < width; i++) {
        for (int j = 0; j < height - 1; j++) {
            float x1 = (float)i / width * 2.0f - 1.0f;
            float z1 = (float)j / height * 2.0f - 1.0f;
            float y1 = surface[i][j] * scale;
            
            float x2 = (float)i / width * 2.0f - 1.0f;
            float z2 = (float)(j+1) / height * 2.0f - 1.0f;
            float y2 = surface[i][j+1] * scale;
            
            // Color based on height
            glColor3f(0.2f + y1, 0.4f, 0.6f + y1);
            glVertex3f(x1, y1, z1);
            glColor3f(0.2f + y2, 0.4f, 0.6f + y2);
            glVertex3f(x2, y2, z2);
        }
    }
    
    // Draw lines along height
    for (int j = 0; j < height; j++) {
        for (int i = 0; i < width - 1; i++) {
            float x1 = (float)i / width * 2.0f - 1.0f;
            float z1 = (float)j / height * 2.0f - 1.0f;
            float y1 = surface[i][j] * scale;
            
            float x2 = (float)(i+1) / width * 2.0f - 1.0f;
            float z2 = (float)j / height * 2.0f - 1.0f;
            float y2 = surface[i+1][j] * scale;
            
            // Color based on height
            glColor3f(0.2f + y1, 0.4f, 0.6f + y1);
            glVertex3f(x1, y1, z1);
            glColor3f(0.2f + y2, 0.4f, 0.6f + y2);
            glVertex3f(x2, y2, z2);
        }
    }
    
    glEnd();
    
    // Add coordinate axes
    glBegin(GL_LINES);
    // X-axis (red)
    glColor3f(1.0f, 0.0f, 0.0f);
    glVertex3f(-1.0f, 0.0f, 0.0f);
    glVertex3f(1.0f, 0.0f, 0.0f);
    
    // Y-axis (green)
    glColor3f(0.0f, 1.0f, 0.0f);
    glVertex3f(0.0f, -1.0f, 0.0f);
    glVertex3f(0.0f, 1.0f, 0.0f);
    
    // Z-axis (blue)
    glColor3f(0.0f, 0.0f, 1.0f);
    glVertex3f(0.0f, 0.0f, -1.0f);
    glVertex3f(0.0f, 0.0f, 1.0f);
    glEnd();
    
    // Reset viewport
    glViewport(0, 0, SCR_WIDTH, SCR_HEIGHT);
}

int main() {
    putenv((char*)"LIBGL_ALWAYS_SOFTWARE=1");
    // Initialize GLFW
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        return -1;
    }
    
    // Important: Use compatibility profile instead of core profile
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_COMPAT_PROFILE);

    // Create a window
    GLFWwindow* window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "Implied Volatility Surface Explorer", NULL, NULL);
    if (window == NULL) {
        std::cout << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);
    
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

    // Setup ImGui
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    ImGui::StyleColorsDark();
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 330");

    // Main loop
    while (!glfwWindowShouldClose(window)) {
        // Process input
        processInput(window);

        // Clear the screen
        glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // Start the ImGui frame
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        // Create ImGui window for parameters
        ImGui::Begin("Heston Model Parameters");
        
        // Add sliders for each parameter
        bool paramsChanged = false;
        
        if (ImGui::SliderFloat("Kappa", &kappa, 0.1f, 10.0f)) paramsChanged = true;
        if (ImGui::SliderFloat("Eta", &eta, 0.01f, 0.5f)) paramsChanged = true;
        if (ImGui::SliderFloat("Sigma", &sigma, 0.01f, 1.0f)) paramsChanged = true;
        if (ImGui::SliderFloat("Rho", &rho, -1.0f, 1.0f)) paramsChanged = true;
        if (ImGui::SliderFloat("V0", &v0, 0.01f, 0.5f)) paramsChanged = true;
        
        ImGui::Separator();
        
        // Add rotation controls
        ImGui::Text("View Controls");
        ImGui::SliderFloat("Rotation X", &rotationX, 0.0f, 360.0f);
        ImGui::SliderFloat("Rotation Y", &rotationY, 0.0f, 360.0f);
        
        // Display current values
        ImGui::Separator();
        ImGui::Text("Current values: kappa=%.2f, eta=%.4f, sigma=%.2f, rho=%.2f, v0=%.4f", 
                    kappa, eta, sigma, rho, v0);
        
        ImGui::End();

        // Generate and render the surface before ImGui rendering
        auto surface = generatePlaceholderSurface(40, 40, kappa, sigma, rho);
        renderSurface(surface, 0.5f);

        // Render ImGui
        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        // Swap buffers and poll events
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    // Cleanup
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glfwTerminate();
    return 0;
}

// Process keyboard input
void processInput(GLFWwindow* window) {
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);
}

// Handle window resize
void framebuffer_size_callback(GLFWwindow* window, int width, int height) {
    glViewport(0, 0, width, height);
}