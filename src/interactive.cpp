#include "interactive.hpp"

#include "renderer.hpp"  // your OpenGL drawing
#include "base_prices.hpp"
#include <Kokkos_Core.hpp>

// GLFW
#include <GLFW/glfw3.h>


// ImGui
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"

#include <vector>
#include <iostream>
#include <algorithm>

// View rotation
float rotationX = 30.0f;
float rotationY = 45.0f;


// Process keyboard input
void processInput(GLFWwindow* window) {
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);
}

// Handle window resize
void framebuffer_size_callback(GLFWwindow* window, int width, int height) {
    glViewport(0, 0, width, height);
}


void interactive_european(
    double theta,
    int m1, int m2, int total_size,
    int width, int height,
    float min_strike, float max_strike,
    float min_maturity, float max_maturity,
    float S_0, float v0,
    float& kappa, float& eta, float& sigma, float& rho, float& r_d, float& q
) {
    // Create calibration points for surface
    std::vector<CalibrationPoint> calibration_points;
    calibration_points.reserve(width * height);
    
    std::vector<double> strikes(width);
    std::vector<double> maturities(height);
    
    // Then use these variables to define strikes and maturities for the surface
    for(int i = 0; i < width; i++) {
        // Evenly distribute strikes between min_strike and max_strike
        strikes[i] = min_strike + (float)i/(width-1) * (max_strike - min_strike);
    }

    for(int j = 0; j < height; j++) {
        // Evenly distribute maturities between min_maturity and max_maturity
        maturities[j] = min_maturity + (float)j/(height-1) * (max_maturity - min_maturity);
    }
    
    // Create calibration points
    int idx = 0;
    for(int j = 0; j < height; j++) {
        double T_m = maturities[j];
        int N_m = 20;//std::max(20, static_cast<int>(T_m * 20));
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
    
    /*
    
    Create the solver arrays
    
    */
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
            m1, m2, r_d, q, point.time_steps, point.delta_t);
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
        //Grid tempGrid(m1, 5*K, S_0, K, K/5, m2, 3.0, v0, 3.0/500);
        
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
        return;
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
        return;
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
    static int totalPdesSolved = 0; //For PDE counter tracking
    float max_surface = 0; //Display highest option price
    float max_iv_surface = 0; //Display highest implied vol value
    
    std::cout << "starting render" << std::endl;
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
        if (ImGui::SliderFloat("V0", &v0, 0.01f, 1.5f)) paramsChanged = true;

        
        ImGui::Separator();

        //added a r_d slider, r_d is still constant for each PDE solve
        ImGui::Text("Risk Modelling controlls");
        if (ImGui::SliderFloat("r_d", &r_d, 0.0f, 0.2f)) paramsChanged = true;
        if (ImGui::SliderFloat("q", &q, 0.0f, 0.2f)) paramsChanged = true;
        if (ImGui::SliderFloat("S0", &S_0, 90.0f, 110.0f)) paramsChanged = true;

        ImGui::Separator();
        
        // Add rotation controls
        ImGui::Text("View Controls");
        ImGui::SliderFloat("Rotation X", &rotationX, 0.0f, 360.0f);
        ImGui::SliderFloat("Rotation Y", &rotationY, 0.0f, 360.0f);
        
        // Display current values
        ImGui::Separator();
        ImGui::Text("Current values: kappa=%.2f, eta=%.4f, sigma=%.2f, rho=%.2f, v0=%.4f, r_d=%.4f, q=%.4f, S0=%.1f", 
                    kappa, eta, sigma, rho, v0, r_d, q, S_0);

        // Add the PDE counter display
        ImGui::Text("PDEs solved: %d (current: %d)", totalPdesSolved, width * height);
        
        ImGui::End();

        // Create ImGui window for parameters
        ImGui::Begin("Heston Model Parameters");
        // [existing parameter sliders and controls]
        ImGui::End();

        // Compute new surface if parameters changed
        if (paramsChanged) {
            max_surface = 0; 
            max_iv_surface = 0; 
            
            // Compute European Call option prices 
            compute_base_prices_multi_maturity(
                S_0, v0, r_d, q, 
                rho, sigma, kappa, eta, 
                m1, m2, total_size, theta, 
                d_calibration_points, 
                width * height, 
                A0_solvers, A1_solvers, A2_solvers, 
                bounds_d, deviceGrids, 
                workspace, base_prices, policy
            );
            
            

            //Compute the Implied Vol-surface to the computed prices
            compute_implied_vol_surface(
                S_0, r_d,
                width, height,
                d_calibration_points,
                base_prices, implied_vols,
                policy
            );

            totalPdesSolved += (width * height);
            
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
                    max_surface = std::max(max_surface, surface[i][j]);

                    iv_surface[i][j] = static_cast<float>(h_implied_vols(idx));
                    max_iv_surface = std::max(max_iv_surface, iv_surface[i][j]);
                }
            }
        }

        //Renders both surfaces next to each other European Call
        renderBothSurfaces(surface, iv_surface, S_0, min_strike, max_strike, rotationX, rotationY);
        
        // Update your axis information windows to show two sets of labels
        ImGui::Begin("Price Surface Axes");
        ImGui::Text("Red axis (X): Strike (%.0f-%.0f)", min_strike, max_strike);
        ImGui::Text("Blue axis (Y): Maturity (%.2f-%.2f years)", min_maturity, max_maturity);
        ImGui::Text("Green axis (Z): Option Price (0-%.1f)", max_surface); 
        ImGui::End();

        ImGui::Begin("IV Surface Axes");
        ImGui::Text("Red axis (X): Strike (%.0f-%.0f)", min_strike, max_strike);
        ImGui::Text("Blue axis (Y): Maturity (%.2f-%.2f years)", min_maturity, max_maturity);
        ImGui::Text("Green axis (Z): Implied Volatility (0-%.2f)", max_iv_surface);
        ImGui::End();

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

void interactive_american(
    double theta,
    int m1, int m2, int total_size,
    int width, int height,
    float min_strike, float max_strike,
    float min_maturity, float max_maturity,
    float S_0, float v0,
    float& kappa, float& eta, float& sigma, float& rho, float& r_d, float& q,
    const std::vector<double>& dividend_dates,
    const std::vector<double>& dividend_amounts,
    const std::vector<double>& dividend_percentages
) {
    
    // Create calibration points for surface
    std::vector<CalibrationPoint> calibration_points;
    calibration_points.reserve(width * height);
    
    std::vector<double> strikes(width);
    std::vector<double> maturities(height);
    
    // Then use these variables to define strikes and maturities for the surface
    for(int i = 0; i < width; i++) {
        // Evenly distribute strikes between min_strike and max_strike
        strikes[i] = min_strike + (float)i/(width-1) * (max_strike - min_strike);
    }

    for(int j = 0; j < height; j++) {
        // Evenly distribute maturities between min_maturity and max_maturity
        maturities[j] = min_maturity + (float)j/(height-1) * (max_maturity - min_maturity);
    }
    
    // Create calibration points
    int idx = 0;
    for(int j = 0; j < height; j++) {
        double T_m = maturities[j];
        int N_m = 20;//std::max(20, static_cast<int>(T_m * 20));
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


    // On host side, create views for dividend data
    Kokkos::View<double*> d_dividend_dates("dividend_dates", dividend_dates.size());
    Kokkos::View<double*> d_dividend_amounts("dividend_amounts", dividend_amounts.size());
    Kokkos::View<double*> d_dividend_percentages("dividend_percentages", dividend_percentages.size());

    // Copy dividend data to device
    auto h_dividend_dates = Kokkos::create_mirror_view(d_dividend_dates);
    auto h_dividend_amounts = Kokkos::create_mirror_view(d_dividend_amounts);
    auto h_dividend_percentages = Kokkos::create_mirror_view(d_dividend_percentages);

    for(size_t i = 0; i < dividend_dates.size(); i++) {
        h_dividend_dates(i) = dividend_dates[i];
        h_dividend_amounts(i) = dividend_amounts[i];
        h_dividend_percentages(i) = dividend_percentages[i];
    }

    Kokkos::deep_copy(d_dividend_dates, h_dividend_dates);
    Kokkos::deep_copy(d_dividend_amounts, h_dividend_amounts);
    Kokkos::deep_copy(d_dividend_percentages, h_dividend_percentages);

    const int num_dividends = dividend_dates.size();
    
    /*
    
    Create the solver arrays
    
    */
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
            m1, m2, r_d, q, point.time_steps, point.delta_t);
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
        //Grid tempGrid(m1, 5*K, S_0, K, K/5, m2, 3.0, v0, 3.0/500);
        
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
        return;
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
        return;
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
    static int totalPdesSolved = 0; //For PDE counter tracking
    float max_surface = 0; //Display highest option price
    float max_iv_surface = 0; //Display highest implied vol value
    
    std::cout << "starting render" << std::endl;
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
        if (ImGui::SliderFloat("V0", &v0, 0.01f, 1.5f)) paramsChanged = true;

        
        ImGui::Separator();

        //added a r_d slider, r_d is still constant for each PDE solve
        ImGui::Text("Risk Modelling controlls");
        if (ImGui::SliderFloat("r_d", &r_d, 0.0f, 0.2f)) paramsChanged = true;
        if (ImGui::SliderFloat("q", &q, 0.0f, 0.2f)) paramsChanged = true;
        if (ImGui::SliderFloat("S0", &S_0, 90.0f, 110.0f)) paramsChanged = true;

        ImGui::Separator();
        
        // Add rotation controls
        ImGui::Text("View Controls");
        ImGui::SliderFloat("Rotation X", &rotationX, 0.0f, 360.0f);
        ImGui::SliderFloat("Rotation Y", &rotationY, 0.0f, 360.0f);
        
        // Display current values
        ImGui::Separator();
        ImGui::Text("Current values: kappa=%.2f, eta=%.4f, sigma=%.2f, rho=%.2f, v0=%.4f, r_d=%.4f, q=%.4f, S0=%.1f", 
                    kappa, eta, sigma, rho, v0, r_d, q, S_0);

        // Add the PDE counter display
        ImGui::Text("PDEs solved: %d (current: %d)", totalPdesSolved, width * height);
        
        ImGui::End();

        // Create ImGui window for parameters
        ImGui::Begin("Heston Model Parameters");
        // [existing parameter sliders and controls]
        ImGui::End();

        // Compute new surface if parameters changed
        if (paramsChanged) {
            max_surface = 0; 
            max_iv_surface = 0; 
        
            // Compute American Call option prices on a dividend paying stock
            compute_base_prices_multi_maturity_american_dividends(
                S_0, v0,
                r_d, q,
                rho, sigma, kappa, eta,
                m1, m2, total_size, theta,
                d_calibration_points,
                total_size,
                A0_solvers, A1_solvers, A2_solvers,
                bounds_d, deviceGrids,
                U_0,
                workspace,
                num_dividends,
                d_dividend_dates,
                d_dividend_amounts,
                d_dividend_percentages,
                base_prices,
                policy
            );
            

            //Compute the Implied Vol-surface to the computed prices
            compute_implied_vol_surface(
                S_0, r_d,
                width, height,
                d_calibration_points,
                base_prices, implied_vols,
                policy
            );

            totalPdesSolved += (width * height);
            
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
                    max_surface = std::max(max_surface, surface[i][j]);

                    iv_surface[i][j] = static_cast<float>(h_implied_vols(idx));
                    max_iv_surface = std::max(max_iv_surface, iv_surface[i][j]);
                }
            }
        }

        //Renders both surfaces with S_0 and dividend dates marked for american call with dividends
        renderBothSurfaces(surface, iv_surface, rotationX, rotationY, S_0, dividend_dates, min_strike, max_strike, min_maturity, max_maturity);

        
        // Update your axis information windows to show two sets of labels
        ImGui::Begin("Price Surface Axes");
        ImGui::Text("Red axis (X): Strike (%.0f-%.0f)", min_strike, max_strike);
        ImGui::Text("Blue axis (Y): Maturity (%.2f-%.2f years)", min_maturity, max_maturity);
        ImGui::Text("Green axis (Z): Option Price (0-%.1f)", max_surface); 
        ImGui::End();

        ImGui::Begin("IV Surface Axes");
        ImGui::Text("Red axis (X): Strike (%.0f-%.0f)", min_strike, max_strike);
        ImGui::Text("Blue axis (Y): Maturity (%.2f-%.2f years)", min_maturity, max_maturity);
        ImGui::Text("Green axis (Z): Implied Volatility (0-%.2f)", max_iv_surface);
        ImGui::End();

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

double call_price(int CP, double S, double K, double r, double v, double T) {
    const double sqrt_T = std::sqrt(T);
    const double log_SK = std::log(S/K);
    const double vol_sqrt_T = v * sqrt_T;
    
    const double d1 = (log_SK + (r + 0.5 * v * v) * T) / vol_sqrt_T;
    const double d2 = d1 - vol_sqrt_T;
    
    return S * std::erfc(-d1/std::sqrt(2.0))/2.0 
           - K * std::exp(-r * T) * std::erfc(-d2/std::sqrt(2.0))/2.0;
}

void interactive_calibration_european(
    double theta,
    int m1, int m2, int total_size,
    int width, int height,
    float min_strike, float max_strike,
    float min_maturity, float max_maturity,
    float S_0, float v0,
    float& kappa, float& eta, float& sigma, float& rho, float& r_d, float& q
){  
    const double market_vol = 0.2;
    const double eps = 1e-3;
    //convergence check parameters
    const int max_iter = 15;
    const double tol = 1.0;//1e-1 * sqrt(width * height);//1.0;//0.001 * num_strikes * (S_0/100.0)*(S_0/100.0); //0.01;
    const double delta_tol = 0.0001; //1e-1 * (1.0 + log(width * height));//0.3 * tol;
    const int total_calibration_size = width*height;


    // Create calibration points for surface
    std::vector<CalibrationPoint> calibration_points;
    calibration_points.reserve(width * height);
    
    std::vector<double> strikes(width);
    std::vector<double> maturities(height);
    
    // Then use these variables to define strikes and maturities for the surface
    for(int i = 0; i < width; i++) {
        // Evenly distribute strikes between min_strike and max_strike
        strikes[i] = min_strike + (float)i/(width-1) * (max_strike - min_strike);
    }

    for(int j = 0; j < height; j++) {
        // Evenly distribute maturities between min_maturity and max_maturity
        maturities[j] = min_maturity + (float)j/(height-1) * (max_maturity - min_maturity);
    }
    
    // Create calibration points
    int idx = 0;
    for(int j = 0; j < height; j++) {
        double T_m = maturities[j];
        int N_m = 20;//std::max(20, static_cast<int>(T_m * 20));
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

    // Market data - store as a flat array, indexed by global_index
    Kokkos::View<double*> market_prices("market_prices", total_calibration_size);
    auto h_market_prices = Kokkos::create_mirror_view(market_prices);

    // Fill market data - generate synthetic data for each maturity/strike pair
    for(const auto& point : calibration_points) {
        // Generate synthetic price for this maturity/strike
        double syn_price = call_price(1, S_0, point.strike, r_d, market_vol, point.maturity);
        h_market_prices(point.global_index) = syn_price;
    }
    Kokkos::deep_copy(market_prices, h_market_prices);
    
    /*
    
    Create the solver arrays
    
    */
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
            m1, m2, r_d, q, point.time_steps, point.delta_t);
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
        //Grid tempGrid(m1, 5*K, S_0, K, K/5, m2, 3.0, v0, 3.0/500);
        
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

    //Jacobian views
    Kokkos::View<double**> J("Jacobian", total_calibration_size, 5);
    Kokkos::View<double**> pert_prices("pert_prices", total_calibration_size, 5);

    // Create views for tracking errors
    Kokkos::View<double*> current_residuals("current_residuals", total_calibration_size);
    Kokkos::View<double*> new_residuals("new_residuals", total_calibration_size);
    Kokkos::View<double*> delta("delta", 5);

    // Current parameters that will be updated
    // will be implicitely copied ifrom host to device. This is faster than keeping it on device
    double current_kappa = kappa + 0.1;
    double current_eta = eta + 0.2;
    double current_sigma = sigma + 0.1;
    double current_rho = rho - 0.05;
    double current_v0 = v0 - 0.03;

    double lambda = 0.01; // Initial LM parameter
    bool converged = false;

    double final_error = 0.0; // for plot information
    int iteration_count = 0;

    // Define bounds for updating
    static constexpr double rho_min = -1.0, rho_max = 1.0;
    
    // Create team policy for parallel execution
    using team_policy = Kokkos::TeamPolicy<Device>;
    team_policy policy(width * height, Kokkos::AUTO);

    // Initialize GLFW and GUI
    putenv((char*)"LIBGL_ALWAYS_SOFTWARE=1");
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        return;
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
        return;
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
    static int totalPdesSolved = 0; //For PDE counter tracking
    float max_surface = 0; //Display highest option price
    float max_iv_surface = 0; //Display highest implied vol value
    
    std::cout << "starting render" << std::endl;
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
        
        if (ImGui::SliderFloat("S0", &S_0, 90.0f, 110.0f)) paramsChanged = true;

        ImGui::Separator();
        
        // Add rotation controls
        ImGui::Text("View Controls");
        ImGui::SliderFloat("Rotation X", &rotationX, 0.0f, 360.0f);
        ImGui::SliderFloat("Rotation Y", &rotationY, 0.0f, 360.0f);
        
        // Display current values
        ImGui::Separator();
        ImGui::Text("Current values: kappa=%.2f, eta=%.4f, sigma=%.2f, rho=%.2f, v0=%.4f, r_d=%.4f, q=%.4f, S0=%.1f", 
                    current_kappa, current_eta, current_sigma, current_rho, current_v0, r_d, q, S_0);

        // This should be outside the calibration loop
        ImGui::Separator();
        ImGui::Text("Final Error=%.2f, Iterations needed=%d", final_error, iteration_count);

        // Add the PDE counter display
        ImGui::Text("PDEs solved: %d", totalPdesSolved);
        
        ImGui::End();

        // Create ImGui window for parameters
        ImGui::Begin("Heston Model Parameters");
        // [existing parameter sliders and controls]
        ImGui::End();

        // Compute new calibration if parameters changed
        if (paramsChanged) {
            max_surface = 0; 
            max_iv_surface = 0; 

            //reset parameters, this can be ommited but i saw that kappa exploded
            //current_kappa = kappa;
            //current_eta = eta;
            //current_sigma = sigma;
            //current_rho = rho;
            //current_v0 = v0;

            //recomputing market prices for this new S_0
            /*
            for(const auto& point : calibration_points) {
                // Generate synthetic price for this maturity/strike
                double syn_price = call_price(1, S_0, point.strike, r_d, market_vol, point.maturity);
                h_market_prices(point.global_index) = syn_price;
            }
            */
            compute_base_prices_multi_maturity(
                S_0, v0, r_d, q, 
                rho, sigma, kappa, eta, 
                m1, m2, total_size, theta, 
                d_calibration_points, 
                width * height, 
                A0_solvers, A1_solvers, A2_solvers, 
                bounds_d, deviceGrids, 
                workspace, market_prices, policy
            );
            Kokkos::deep_copy(h_market_prices, market_prices);

            converged = false;
    
            // Reset lambda to starting value for new calibration
            lambda = 0.01;
            
            for(int iter = 0; iter < max_iter && !converged; iter++) {

                // Compute European Call option prices 
                compute_jacobian_multi_maturity(
                    S_0, current_v0, 
                    r_d, q,
                    rho, current_sigma, current_kappa, current_eta,
                    m1, m2, total_size, theta,
                    d_calibration_points,  // Make sure this parameter is passed
                    total_calibration_size,
                    A0_solvers, A1_solvers, A2_solvers,
                    bounds_d, deviceGrids,
                    U_0, workspace,
                    J, base_prices,
                    policy,
                    eps
                );

                //base prices are already computed in compute_jacobian and stored in base_price
                // Compute current residuals (for all calibration points)
                Kokkos::parallel_for("compute_residuals", total_calibration_size, 
                    KOKKOS_LAMBDA(const int i) {
                        current_residuals(i) = market_prices(i) - base_prices(i);
                });
                Kokkos::fence();

                // Compute parameter update
                compute_parameter_update_on_device(J, current_residuals, lambda, delta);
                
                // Get delta on host
                auto h_delta = Kokkos::create_mirror_view(delta);
                Kokkos::deep_copy(h_delta, delta);

                // Try new parameters
                double new_kappa = std::max(1e-3, current_kappa + h_delta(0));
                double new_eta = std::max(1e-2, current_eta + h_delta(1));
                double new_sigma = std::max(1e-2, current_sigma + h_delta(2));
                double new_rho = std::min(rho_max, std::max(rho_min, current_rho + h_delta(3)));
                double new_v0 = std::max(1e-2, current_v0 + h_delta(4));

                // Compute delta norm
                double delta_norm = 0.0;
                for(int i = 0; i < 5; ++i) {
                    delta_norm += h_delta(i) * h_delta(i);
                }
                delta_norm = std::sqrt(delta_norm);

                std::cout << "iteration" << iter << ", delta" << delta_norm << std::endl;

                // Computing current error across all calibration points
                double current_error = 0;
                auto h_current_residuals = Kokkos::create_mirror_view(current_residuals);
                Kokkos::deep_copy(h_current_residuals, current_residuals);

                for(int i = 0; i < total_calibration_size; i++) {
                    current_error += h_current_residuals(i) * h_current_residuals(i);
                }

                std::cout << "iteration" << iter << ", curretn error" << current_error << std::endl;

                // Check convergence
                //maybe do two errir checks one for delta normn and the other for the toleranze
                //if(delta_norm < delta_tol || (current_error < tol)) {
                if((current_error < tol)) {
                    converged = true;

                    // Update parameters to final values
                    current_kappa = new_kappa;
                    current_eta = new_eta;
                    current_sigma = new_sigma;
                    current_rho = new_rho;
                    current_v0 = new_v0;

                    final_error = current_error;
                    iteration_count = iter + 1;
                    std::cout << "Final error" << final_error;
                    //std::cout << ", Final delta" << delta_norm << std::endl;
                    break; // Exit the loop
                }

                // Compute new prices with updated parameters
                Kokkos::deep_copy(workspace.U, U_0); // reset initial condition

                // Call the multi-maturity base price computation
                compute_base_prices_multi_maturity(
                    S_0, new_v0,
                    r_d, q,
                    new_rho, new_sigma, new_kappa, new_eta,
                    m1, m2, total_size, theta,
                    d_calibration_points,
                    total_calibration_size,
                    A0_solvers, A1_solvers, A2_solvers,
                    bounds_d, deviceGrids,
                    workspace,
                    base_prices,
                    policy
                );

                // Compute new residuals
                Kokkos::parallel_for("compute_new_residuals", total_calibration_size, 
                    KOKKOS_LAMBDA(const int i) {
                        new_residuals(i) = market_prices(i) - base_prices(i);
                });
                Kokkos::fence();

                // Compute new error norms across all calibration points
                double new_error = 0.0;
                auto h_new_residuals = Kokkos::create_mirror_view(new_residuals);
                Kokkos::deep_copy(h_new_residuals, new_residuals);

                for(int i = 0; i < total_calibration_size; i++) {
                    new_error += h_new_residuals(i) * h_new_residuals(i);
                }

                std::cout << "iteration" << iter << ", new error" << new_error << std::endl;


                // Update parameters based on error improvement
                if(new_error < current_error) {
                    current_kappa = new_kappa;
                    current_eta = new_eta;
                    current_sigma = new_sigma;
                    current_rho = new_rho;
                    current_v0 = new_v0;
                    lambda = std::max(lambda / 10.0, 1e-10);  // Decrease lambda but not too small
                } 
                else {
                    lambda = std::min(lambda * 10.0, 1e10);  // Increase lambda but not too large
                }

                final_error = std::min(new_error, current_error);
                iteration_count = iter + 1;
            }

            //Compute the Implied Vol-surface to the computed prices
            compute_implied_vol_surface(
                S_0, r_d,
                width, height,
                d_calibration_points,
                base_prices, implied_vols,
                policy
            );

            totalPdesSolved += total_calibration_size * (1 + 5 + 1) * iteration_count - total_calibration_size;
            
            // Copy prices to surface
            auto h_prices = Kokkos::create_mirror_view(base_prices);
            auto h_implied_vols = Kokkos::create_mirror_view(implied_vols);

            Kokkos::deep_copy(h_prices, base_prices);
            Kokkos::deep_copy(h_implied_vols, implied_vols);

        
            // Convert to surface format
            for(int i = 0; i < width; i++) {
                for(int j = 0; j < height; j++) {
                    int idx = j * width + i;
                    //surface[i][j] = static_cast<float>(h_prices(idx));
                    surface[i][j] = std::abs(h_prices(idx)-h_market_prices(idx)); //market price differences
                    max_surface = std::max(max_surface, surface[i][j]);


                    iv_surface[i][j] = static_cast<float>(h_implied_vols(idx));
                    //iv_surface[i][j] = std::abs((h_implied_vols(idx)-market_vol)); //implied vol differences

                    max_iv_surface = std::max(max_iv_surface, iv_surface[i][j]);
                }
            }
        }

        //Renders both surfaces next to each other European Call
        renderBothSurfaces(surface, iv_surface, S_0, min_strike, max_strike, rotationX, rotationY);
        
        // Update your axis information windows to show two sets of labels
        ImGui::Begin("Price Surface Axes");
        ImGui::Text("Red axis (X): Strike (%.0f-%.0f)", min_strike, max_strike);
        ImGui::Text("Blue axis (Y): Maturity (%.2f-%.2f years)", min_maturity, max_maturity);
        ImGui::Text("Green axis (Z): Option Price (0-%.1f)", max_surface); 
        ImGui::End();

        ImGui::Begin("IV Surface Axes");
        ImGui::Text("Red axis (X): Strike (%.0f-%.0f)", min_strike, max_strike);
        ImGui::Text("Blue axis (Y): Maturity (%.2f-%.2f years)", min_maturity, max_maturity);
        ImGui::Text("Green axis (Z): Implied Volatility (0-%.4f)", max_iv_surface);
        ImGui::End();

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

void interactive_calibration_american(
    double theta,
    int m1, int m2, int total_size,
    int width, int height,
    float min_strike, float max_strike,
    float min_maturity, float max_maturity,
    float S_0, float v0,
    float& kappa, float& eta, float& sigma, float& rho, float& r_d, float& q,
    const std::vector<double>& dividend_dates,
    const std::vector<double>& dividend_amounts,
    const std::vector<double>& dividend_percentages
){
    std::cout << "Hello";
}