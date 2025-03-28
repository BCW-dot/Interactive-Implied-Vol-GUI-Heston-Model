/*


Advanced PDE vizulization
works perfecty


*/

#include <iostream>
#include <vector>
#include <cmath>

#include <Kokkos_Core.hpp>

#include "base_prices.hpp"

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



// Add coordinate axes with better visual indicators
void renderAxes() {
    // Draw the main axes with thicker lines
    glLineWidth(2.0f);
    glBegin(GL_LINES);
    // X-axis (red) - Strike
    glColor3f(1.0f, 0.0f, 0.0f);
    glVertex3f(-1.2f, 0.0f, 0.0f);
    glVertex3f(1.2f, 0.0f, 0.0f);
    
    // Y-axis (green) - Price
    glColor3f(0.0f, 1.0f, 0.0f);
    glVertex3f(0.0f, -0.2f, 0.0f);
    glVertex3f(0.0f, 1.2f, 0.0f);
    
    // Z-axis (blue) - Maturity
    glColor3f(0.0f, 0.0f, 1.0f);
    glVertex3f(0.0f, 0.0f, -1.2f);
    glVertex3f(0.0f, 0.0f, 1.2f);
    glEnd();
    
    // Reset line width
    glLineWidth(1.0f);
    
    // Draw tick marks on axes
    glBegin(GL_LINES);
    // X-axis ticks
    glColor3f(1.0f, 0.0f, 0.0f);
    for (float x = -1.0f; x <= 1.0f; x += 0.2f) {
        glVertex3f(x, 0.0f, 0.0f);
        glVertex3f(x, -0.05f, 0.0f);
    }
    
    // Z-axis ticks
    glColor3f(0.0f, 0.0f, 1.0f);
    for (float z = -1.0f; z <= 1.0f; z += 0.2f) {
        glVertex3f(0.0f, 0.0f, z);
        glVertex3f(-0.05f, 0.0f, z);
    }
    
    // Y-axis ticks
    glColor3f(0.0f, 1.0f, 0.0f);
    for (float y = 0.0f; y <= 1.0f; y += 0.2f) {
        glVertex3f(0.0f, y, 0.0f);
        glVertex3f(-0.05f, y, 0.0f);
    }
    glEnd();
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
    
    // Add labeled coordinate axes
    renderAxes();
    
    
    
    // Reset viewport
    glViewport(0, 0, SCR_WIDTH, SCR_HEIGHT);
}


// Modify rendering function to handle both surfaces
// Helper function that just renders the surface data (no axes)
void renderSurfaceData(const std::vector<std::vector<float>>& surface, float scale) {
    int width = surface.size();
    int height = surface[0].size();
    
    // Draw lines along width (same as in your original renderSurface)
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
}

void renderBothSurfaces(
    const std::vector<std::vector<float>>& price_surface, 
    const std::vector<std::vector<float>>& iv_surface, 
    float rotationX, float rotationY) {
    
    // Left viewport for price surface
    glViewport(100, 100, 500, 500);
    glEnable(GL_DEPTH_TEST);
    
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(-1.5, 1.5, -1.5, 1.5, -2.0, 2.0);
    
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glRotatef(rotationX, 1.0f, 0.0f, 0.0f);
    glRotatef(rotationY, 0.0f, 1.0f, 0.0f);
    
    // Render price surface with its scale
    renderSurfaceData(price_surface, 0.05f);
    renderAxes();
    
    // Right viewport for implied vol surface
    glViewport(600, 100, 500, 500);
    glEnable(GL_DEPTH_TEST);
    
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(-1.5, 1.5, -1.5, 1.5, -2.0, 2.0);
    
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glRotatef(rotationX, 1.0f, 0.0f, 0.0f);
    glRotatef(rotationY, 0.0f, 1.0f, 0.0f);
    
    // Render IV surface with its scale
    renderSurfaceData(iv_surface, 1.0f);
    renderAxes();
    
    // Reset viewport to full window
    glViewport(0, 0, SCR_WIDTH, SCR_HEIGHT);
}


int main() {
    Kokkos::initialize();
    {
        // Market parameters
        const double S_0 = 100.0;
        const double r_d = 0.025;
        const double r_f = 0.0;
        const double theta = 0.8;
        
        // Grid dimensions - keep small for interactive performance
        const int m1 = 20;
        const int m2 = 15;
        const int total_size = (m1+1) * (m2+1);
        
        // Surface dimensions
        const int width = 30;  // Number of strikes
        const int height = 10; // Number of maturities
        
        // Create calibration points for surface
        std::vector<CalibrationPoint> calibration_points;
        calibration_points.reserve(width * height);
        
        std::vector<double> strikes(width);
        std::vector<double> maturities(height);
        
        // Define strikes and maturities for the surface
        for(int i = 0; i < width; i++) {
            strikes[i] = S_0 * (0.8 + 0.04 * i);  // 80% to 140% of spot
        }
        
        for(int j = 0; j < height; j++) {
            //maturities[j] = 0.25 + j * 0.25;  // 3 months to 2.5 years
            maturities[j] = 0.25 + j * 0.65;  // 3 months to 2.5 years
        }
        
        // Create calibration points
        int idx = 0;
        for(int j = 0; j < height; j++) {
            double T_m = maturities[j];
            int N_m = std::max(20, static_cast<int>(T_m * 20));
            double dt_m = T_m / N_m;
            
            for(int i = 0; i < width; i++) {
                calibration_points.push_back({
                    strikes[i],
                    T_m,
                    N_m,
                    dt_m,
                    idx++
                });
            }
        }

        // Create device view for calibration points
        Kokkos::View<CalibrationPoint*> d_calibration_points("d_calibration_points", calibration_points.size());
        auto h_calibration_points = Kokkos::create_mirror_view(d_calibration_points);
        for(size_t i = 0; i < calibration_points.size(); i++) {
            h_calibration_points(i) = calibration_points[i];
        }
        Kokkos::deep_copy(d_calibration_points, h_calibration_points);
        
        // Create solver arrays
        using Device = Kokkos::DefaultExecutionSpace;
        Kokkos::View<Device_A0_heston<Device>*> A0_solvers("A0_solvers", width * height);
        Kokkos::View<Device_A1_heston<Device>*> A1_solvers("A1_solvers", width * height);
        Kokkos::View<Device_A2_shuffled_heston<Device>*> A2_solvers("A2_solvers", width * height);
        
        // Initialize solvers
        auto h_A0 = Kokkos::create_mirror_view(A0_solvers);
        auto h_A1 = Kokkos::create_mirror_view(A1_solvers);
        auto h_A2 = Kokkos::create_mirror_view(A2_solvers);
        
        for(int i = 0; i < width * height; i++) {
            h_A0(i) = Device_A0_heston<Device>(m1, m2);
            h_A1(i) = Device_A1_heston<Device>(m1, m2);
            h_A2(i) = Device_A2_shuffled_heston<Device>(m1, m2);
        }
        
        Kokkos::deep_copy(A0_solvers, h_A0);
        Kokkos::deep_copy(A1_solvers, h_A1);
        Kokkos::deep_copy(A2_solvers, h_A2);
        
        // Create boundary conditions
        Kokkos::View<Device_BoundaryConditions<Device>*> bounds_d("bounds_d", width * height);
        auto h_bounds = Kokkos::create_mirror_view(bounds_d);
        
        for(const auto& point : calibration_points) {
            int idx = point.global_index;
            h_bounds(idx) = Device_BoundaryConditions<Device>(
                m1, m2, r_d, r_f, point.time_steps, point.delta_t);
        }
        Kokkos::deep_copy(bounds_d, h_bounds);
        
        // Initialize grid views
        std::vector<GridViews> hostGrids;
        buildMultipleGridViews(hostGrids, width * height, m1, m2);
        
        for(const auto& point : calibration_points) {
            int idx = point.global_index;
            double K = point.strike;
            
            auto h_Vec_s = Kokkos::create_mirror_view(hostGrids[idx].device_Vec_s);
            auto h_Vec_v = Kokkos::create_mirror_view(hostGrids[idx].device_Vec_v);
            auto h_Delta_s = Kokkos::create_mirror_view(hostGrids[idx].device_Delta_s);
            auto h_Delta_v = Kokkos::create_mirror_view(hostGrids[idx].device_Delta_v);
            
            Grid tempGrid(m1, 8*K, S_0, K, K/5, m2, 5.0, v0, 5.0/500);
            
            for(int j = 0; j <= m1; j++) h_Vec_s(j) = tempGrid.Vec_s[j];
            for(int j = 0; j <= m2; j++) h_Vec_v(j) = tempGrid.Vec_v[j];
            for(int j = 0; j < m1; j++) h_Delta_s(j) = tempGrid.Delta_s[j];
            for(int j = 0; j < m2; j++) h_Delta_v(j) = tempGrid.Delta_v[j];
            
            Kokkos::deep_copy(hostGrids[idx].device_Vec_s, h_Vec_s);
            Kokkos::deep_copy(hostGrids[idx].device_Vec_v, h_Vec_v);
            Kokkos::deep_copy(hostGrids[idx].device_Delta_s, h_Delta_s);
            Kokkos::deep_copy(hostGrids[idx].device_Delta_v, h_Delta_v);
        }
        
        Kokkos::View<GridViews*> deviceGrids("deviceGrids", width * height);
        auto h_deviceGrids = Kokkos::create_mirror_view(deviceGrids);
        for(int i = 0; i < width * height; i++) {
            h_deviceGrids(i) = hostGrids[i];
        }
        Kokkos::deep_copy(deviceGrids, h_deviceGrids);
        
        // Initialize payoffs
        Kokkos::View<double**> U_0("U_0", width * height, total_size);
        auto h_U_0 = Kokkos::create_mirror_view(U_0);
        
        for(const auto& point : calibration_points) {
            int idx = point.global_index;
            double K = point.strike;
            
            auto h_Vec_s = Kokkos::create_mirror_view(hostGrids[idx].device_Vec_s);
            Kokkos::deep_copy(h_Vec_s, hostGrids[idx].device_Vec_s);
            
            for(int j = 0; j <= m2; j++) {
                for(int i = 0; i <= m1; i++) {
                    h_U_0(idx, i + j*(m1+1)) = std::max(h_Vec_s(i) - K, 0.0);
                }
            }
        }
        Kokkos::deep_copy(U_0, h_U_0);

        // Create workspace for computation
        DO_Workspace<Device> workspace(width * height, total_size);
        Kokkos::deep_copy(workspace.U, U_0);
        
        // Create view for prices
        Kokkos::View<double*> base_prices("base_prices", width * height);

        Kokkos::View<double*> implied_vols("implied_vols", width * height);
        
        // Create team policy for parallel execution
        using team_policy = Kokkos::TeamPolicy<Device>;
        team_policy policy(width * height, Kokkos::AUTO);

        // Initialize GLFW and GUI
        putenv((char*)"LIBGL_ALWAYS_SOFTWARE=1");
        if (!glfwInit()) {
            std::cerr << "Failed to initialize GLFW" << std::endl;
            return -1;
        }

        // Important: Use compatibility profile
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_COMPAT_PROFILE);

        // Create window
        GLFWwindow* window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "Heston Surface Explorer", NULL, NULL);
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

        // Surface data for visualization
        std::vector<std::vector<float>> surface(width, std::vector<float>(height));
        // Create a second surface for implied volatilities
        std::vector<std::vector<float>> iv_surface(width, std::vector<float>(height));

        // Main loop
        bool paramsChanged = true;  // Force computation on first frame
        
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
            paramsChanged = false;
            
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

            // Create ImGui window for parameters
            ImGui::Begin("Heston Model Parameters");
            // [existing parameter sliders and controls]
            ImGui::End();

            // Compute new surface if parameters changed
            if (paramsChanged) {
                //Reset initial condition
                Kokkos::deep_copy(workspace.U, U_0); 
                
                // Compute option prices using the Heston PDE solver
                compute_base_prices_multi_maturity(
                    S_0, v0, r_d, r_f, 
                    rho, sigma, kappa, eta, 
                    m1, m2, total_size, theta, 
                    d_calibration_points, 
                    width * height, 
                    A0_solvers, A1_solvers, A2_solvers, 
                    bounds_d, deviceGrids, 
                    workspace, base_prices, policy
                );

                compute_implied_vol_surface(
                    S_0, r_d,
                    width, height,
                    d_calibration_points,
                    base_prices, implied_vols,
                    policy
                );
                
                // Copy prices to surface
                auto h_prices = Kokkos::create_mirror_view(base_prices);
                auto h_implied_vols = Kokkos::create_mirror_view(implied_vols);

                Kokkos::deep_copy(h_prices, base_prices);
                Kokkos::deep_copy(h_implied_vols, implied_vols);

            
                
                // Convert to surface format
                for(int i = 0; i < width; i++) {
                    for(int j = 0; j < height; j++) {
                        int idx = j * width + i;
                        surface[i][j] = static_cast<float>(h_prices(idx));
                        iv_surface[i][j] = static_cast<float>(h_implied_vols(idx));
                    }
                }
            }

            //rendering both surfaces at the same time
            renderBothSurfaces(surface, iv_surface, rotationX, rotationY);

            
            // Update your axis information windows to show two sets of labels
            ImGui::Begin("Price Surface Axes");
            ImGui::Text("Red axis (X): Strike (%.0f-%.0f)", S_0 * 0.8, S_0 * 1.4);
            ImGui::Text("Blue axis (Y): Maturity (0.25-2.5 years)");
            ImGui::Text("Green axis (Z): Option Price (0-%.1f)", 20.0f); 
            ImGui::End();

            ImGui::Begin("IV Surface Axes");
            ImGui::Text("Red axis (X): Strike (%.0f-%.0f)", S_0 * 0.8, S_0 * 1.4);
            ImGui::Text("Blue axis (Y): Maturity (0.25-2.5 years)");
            ImGui::Text("Green axis (Z): Implied Volatility (0-1)");
            ImGui::End();
            

            // Render the surface individually
            //renderSurface(surface, 0.05f);  // Scale to make heights visible
            //renderSurface(iv_surface, 1.0f);

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
    }
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




























/*

*
*
* WORKS SIMPLE AND WORKS
*
*


*/





/*
#include <iostream>
#include <vector>
#include <cmath>

#include <Kokkos_Core.hpp>

#include "base_prices.hpp"

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
        auto surface = generateEnhancedSurface(40, 40, kappa, eta, sigma, rho, v0);
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
*/




