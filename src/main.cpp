#include <iostream>
#include <vector>
#include <cmath>

#include <Kokkos_Core.hpp>

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
    
    // Create a Kokkos View for the surface calculation
    Kokkos::View<float**> d_surface("surface", width, height);
    
    // Simple parallel calculation using Kokkos
    Kokkos::parallel_for("surface_calc", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {width, height}),
        KOKKOS_LAMBDA(const int i, const int j) {
            float x = (float)i / width * 2.0f - 1.0f; // map to [-1, 1]
            float y = (float)j / height * 2.0f - 1.0f; // map to [-1, 1]
            
            // More extreme surface with smile/skew effects and peaks
            float base = 0.15f; // Base volatility
            float skew = 0.08f * rho * x; // Skew effect
            float smile = 0.12f * sigma * (x*x); // Smile/convexity
            float term_structure = 0.05f * (1.0f - exp(-kappa * y*y)); // Term structure
            float peak = 0.2f * exp(-8.0f * ((x-0.3f)*(x-0.3f) + (y+0.2f)*(y+0.2f))); // Sharp peak
            
            // Combine effects
            float value = std::pow(base + skew + smile + term_structure + peak, 2);
            
            // Add some randomness (Note: We use a simplified approach for parallel code)
            value += 0.01f * (((i * 263 + j * 71) % 100) / 100.0f - 0.5f);
            
            d_surface(i, j) = value;
        });
    
    // Copy the results back to the host
    Kokkos::fence();
    auto h_surface = Kokkos::create_mirror_view(d_surface);
    Kokkos::deep_copy(h_surface, d_surface);
    
    // Copy from Kokkos view to std::vector
    for (int i = 0; i < width; i++) {
        for (int j = 0; j < height; j++) {
            surface[i][j] = h_surface(i, j);
        }
    }
    
    return surface;
}

// Enhanced volatility surface function using all parameters
// Enhanced volatility surface function using all parameters
std::vector<std::vector<float>> generateEnhancedSurface(int width, int height, float kappa, float eta, float sigma, float rho, float v0) {
    std::vector<std::vector<float>> surface(width, std::vector<float>(height));
    
    // Create a Kokkos View for the surface calculation
    Kokkos::View<float**> d_surface("surface", width, height);
    
    // Use explicitly CUDA execution space
    using execution_space = Kokkos::Cuda;
    
    // Number of iterations for computational intensity
    const int iterations = 50;
    
    // Computational intensive parallel calculation
    Kokkos::parallel_for("enhanced_surface", 
        Kokkos::MDRangePolicy<execution_space, Kokkos::Rank<2>>({0, 0}, {width, height}),
        KOKKOS_LAMBDA(const int i, const int j) {
            float x = (float)i / width * 4.0f - 2.0f;  // map to [-2, 2]
            float y = (float)j / height * 4.0f - 2.0f; // map to [-2, 2]
            
            // Base volatility components
            float base_vol = v0 * (1.0f + 0.1f * sin(3.0f * x * y));
            
            // Skew component with rho dependency
            float skew = 0.15f * rho * (x + 0.5f * sin(2.0f * x));
            
            // Smile component with sigma dependency
            float smile = sigma * (0.2f * x*x + 0.05f * cos(3.0f * x));
            
            // Term structure with kappa dependency
            float term = 0.08f * (1.0f - exp(-kappa * (y*y + 0.3f * sin(y))));
            
            // Volatility clustering effect with eta dependency
            float cluster = eta * 0.5f * exp(-2.0f * (pow(x-0.4f, 2) + pow(y+0.3f, 2)));
            
            // Dynamic spikes based on parameter combinations
            float spikes = 0.1f * exp(-5.0f * (pow(sin(x*sigma+0.2f), 2) + pow(cos(y*rho-0.1f), 2)));
            
            // Add computational intensity through iterations
            float iterative_component = 0.0f;
            float temp_x = x, temp_y = y;
            
            // Iterative calculation for computational intensity
            for (int iter = 0; iter < iterations; iter++) {
                // Simple orbit-trap like calculation
                float tx = temp_x * temp_x - temp_y * temp_y + 0.1f * rho;
                float ty = 2.0f * temp_x * temp_y + 0.1f * sigma;
                
                temp_x = tx;
                temp_y = ty;
                
                // Bound the values to prevent divergence - using direct comparisons
                temp_x = (temp_x > 2.0f) ? 2.0f : temp_x;
                temp_x = (temp_x < -2.0f) ? -2.0f : temp_x;
                temp_y = (temp_y > 2.0f) ? 2.0f : temp_y;
                temp_y = (temp_y < -2.0f) ? -2.0f : temp_y;
                
                // Accumulate into the iterative component
                iterative_component += 0.01f * exp(-(temp_x*temp_x + temp_y*temp_y) * kappa);
            }
            
            // Time-dependent wave patterns
            float waves = 0.05f * sin(x * 4.0f) * cos(y * 4.0f + 0.5f * sin(x * 2.0f));
            
            // Combine all effects
            float vol = base_vol + skew + smile + term + cluster + spikes + iterative_component + waves;
            
            // Ensure the result is positive and realistic
            // Using direct comparison instead of std::min/max to avoid CUDA warnings
            vol = (vol < 0.05f) ? 0.05f : vol;  // max(0.05f, vol)
            vol = (vol > 1.0f) ? 1.0f : vol;    // min(1.0f, vol)
            
            // Square the result to emphasize patterns
            float result = vol * vol;
            
            d_surface(i, j) = result;
        });
    
    // Copy the results back to the host
    Kokkos::fence();
    auto h_surface = Kokkos::create_mirror_view(d_surface);
    Kokkos::deep_copy(h_surface, d_surface);
    
    // Copy from Kokkos view to std::vector
    for (int i = 0; i < width; i++) {
        for (int j = 0; j < height; j++) {
            surface[i][j] = h_surface(i, j);
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



//Chekcs that Kokkos is working properly
#include <chrono>

void runPerformanceTest() {
    const int N = 1000; // Size of test matrix - increase for more significant results
    
    // CPU implementation
    auto start_cpu = std::chrono::high_resolution_clock::now();
    
    std::vector<std::vector<float>> cpu_result(N, std::vector<float>(N));
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            float x = (float)i / N * 2.0f - 1.0f;
            float y = (float)j / N * 2.0f - 1.0f;
            float value = std::pow(x*x + y*y, 1.5f);
            cpu_result[i][j] = value;
        }
    }
    
    auto end_cpu = std::chrono::high_resolution_clock::now();
    auto cpu_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_cpu - start_cpu).count();
    
    // GPU implementation with Kokkos
    auto start_gpu = std::chrono::high_resolution_clock::now();
    
    Kokkos::View<float**> d_result("gpu_result", N, N);
    Kokkos::parallel_for("gpu_calc", 
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {N, N}),
        KOKKOS_LAMBDA(const int i, const int j) {
            float x = (float)i / N * 2.0f - 1.0f;
            float y = (float)j / N * 2.0f - 1.0f;
            float value = std::pow(x*x + y*y, 1.5f);
            d_result(i, j) = value;
        });
    
    Kokkos::fence(); // Ensure all GPU work is complete
    
    auto end_gpu = std::chrono::high_resolution_clock::now();
    auto gpu_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_gpu - start_gpu).count();
    
    // Print results
    std::cout << "===== Performance Test =====" << std::endl;
    std::cout << "CPU time: " << cpu_time << " ms" << std::endl;
    std::cout << "GPU time: " << gpu_time << " ms" << std::endl;
    std::cout << "Speedup: " << (float)cpu_time / gpu_time << "x" << std::endl;
    
    // If GPU is significantly faster, it's using CUDA
    if ((float)cpu_time / gpu_time > 1.5f) {
        std::cout << "✓ GPU acceleration is working!" << std::endl;
    } else {
        std::cout << "⚠ GPU acceleration may not be working efficiently" << std::endl;
        std::cout << "  For small workloads, CPU might still be faster due to overhead" << std::endl;
    }
}

void checkKokkosConfig() {
    // Print execution space info
    std::cout << "Default execution space: " << 
        typeid(Kokkos::DefaultExecutionSpace).name() << std::endl;
    
    if (std::is_same<Kokkos::DefaultExecutionSpace, Kokkos::Cuda>::value) {
        std::cout << "Using CUDA as default execution space" << std::endl;
    } else {
        std::cout << "Not using CUDA as default execution space" << std::endl;
    }
    
    // Check if CUDA is enabled
    #ifdef KOKKOS_ENABLE_CUDA
        std::cout << "CUDA is enabled in Kokkos" << std::endl;
    #else
        std::cout << "CUDA is NOT enabled in Kokkos" << std::endl;
    #endif
}



int main() {
    Kokkos::initialize();

    //runPerformanceTest();
    //checkKokkosConfig();

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
        //auto surface = generatePlaceholderSurface(40, 40, kappa, sigma, rho);
        auto surface = generateEnhancedSurface(500, 500, kappa, eta, sigma, rho, v0);
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

    Kokkos::finalize();

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