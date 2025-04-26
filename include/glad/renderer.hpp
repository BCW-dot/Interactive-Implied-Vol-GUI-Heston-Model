#pragma once

#include <vector>

// Function declarations
void renderAxes(float S_0, float min_strike, float max_strike);

void renderAxes(float S_0, const std::vector<double>& maturity_points,
                float min_strike, float max_strike,
                float min_maturity, float max_maturity);

void renderSurfaceData(const std::vector<std::vector<float>>& surface, float scale);

void renderBothSurfaces(
    const std::vector<std::vector<float>>& price_surface,
    const std::vector<std::vector<float>>& iv_surface,
    float rotationX, float rotationY,
    float S_0, const std::vector<double>& maturity_points,
    float min_strike, float max_strike,
    float min_maturity, float max_maturity);

void renderBothSurfaces(
    const std::vector<std::vector<float>>& price_surface,
    const std::vector<std::vector<float>>& iv_surface,
    float S_0, float min_strike, float max_strike,
    float rotationX, float rotationY);

// Screen dimensions (needed for viewport reset)
extern const unsigned int SCR_WIDTH;
extern const unsigned int SCR_HEIGHT;
