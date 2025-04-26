#include "renderer.hpp"
#include <GLFW/glfw3.h>
#include <algorithm> // for std::min, std::max

const unsigned int SCR_WIDTH = 1200;
const unsigned int SCR_HEIGHT = 1000;

// Add coordinate axes with better visual indicators
//for european call with S_0 indicator
void renderAxes(float S_0, float min_strike, float max_strike) {
    // Draw the main axes with thicker lines
    glLineWidth(2.0f);
    glBegin(GL_LINES);
    // X-axis (red) - Strike - moved to front corner
    glColor3f(1.0f, 0.0f, 0.0f);
    glVertex3f(-1.0f, -0.1f, -1.0f);
    glVertex3f(1.0f, -0.1f, -1.0f);
    
    // Y-axis (green) - Price - moved to front corner
    glColor3f(0.0f, 1.0f, 0.0f);
    glVertex3f(-1.0f, -0.1f, -1.0f);
    glVertex3f(-1.0f, 1.0f, -1.0f);
    
    // Z-axis (blue) - Maturity - moved to front corner
    glColor3f(0.0f, 0.0f, 1.0f);
    glVertex3f(-1.0f, -0.1f, -1.0f);
    glVertex3f(-1.0f, -0.1f, 1.0f);
    glEnd();
    
    // Reset line width
    glLineWidth(1.0f);
    
    // Draw tick marks on axes (adjusted for new position)
    glBegin(GL_LINES);
    // X-axis ticks
    glColor3f(1.0f, 0.0f, 0.0f);
    for (float x = -1.0f; x <= 1.0f; x += 0.2f) {
        glVertex3f(x, -0.1f, -1.0f);
        glVertex3f(x, -0.15f, -1.0f);
    }
    
    // Z-axis ticks
    glColor3f(0.0f, 0.0f, 1.0f);
    for (float z = -1.0f; z <= 1.0f; z += 0.2f) {
        glVertex3f(-1.0f, -0.1f, z);
        glVertex3f(-1.0f, -0.15f, z);
    }
    
    // Y-axis ticks
    glColor3f(0.0f, 1.0f, 0.0f);
    for (float y = 0.0f; y <= 1.0f; y += 0.2f) {
        glVertex3f(-1.0f, y, -1.0f);
        glVertex3f(-1.05f, y, -1.0f);
    }
    glEnd();
    
    // Add special marker for S_0 on X-axis
    glLineWidth(3.0f);
    glBegin(GL_LINES);
    glColor3f(1.0f, 1.0f, 0.0f); // Yellow color for S_0
    // Normalize S_0 to [-1, 1] range
    float x_pos = -1.0f + 2.0f * (S_0 - min_strike) / (max_strike - min_strike);
    glVertex3f(x_pos, -0.1f, -1.0f);
    glVertex3f(x_pos, -0.25f, -1.0f); // Make it longer than regular ticks
    glEnd();
    glLineWidth(1.0f);
}


// Modified function to include indicators
//for american dividneds with indicators
void renderAxes(float S_0, const std::vector<double>& maturity_points, 
    float min_strike, float max_strike, float min_maturity, float max_maturity) {
    // Draw the main axes with thicker lines (keep your existing code)
    glLineWidth(2.0f);
    glBegin(GL_LINES);
    // X-axis (red) - Strike - moved to front corner
    glColor3f(1.0f, 0.0f, 0.0f);
    glVertex3f(-1.0f, -0.1f, -1.0f);
    glVertex3f(1.0f, -0.1f, -1.0f);

    // Y-axis (green) - Price - moved to front corner
    glColor3f(0.0f, 1.0f, 0.0f);
    glVertex3f(-1.0f, -0.1f, -1.0f);
    glVertex3f(-1.0f, 1.0f, -1.0f);

    // Z-axis (blue) - Maturity - moved to front corner
    glColor3f(0.0f, 0.0f, 1.0f);
    glVertex3f(-1.0f, -0.1f, -1.0f);
    glVertex3f(-1.0f, -0.1f, 1.0f);
    glEnd();

    // Reset line width
    glLineWidth(1.0f);

    // Draw tick marks on axes (adjusted for new position)
    glBegin(GL_LINES);
    // X-axis ticks
    glColor3f(1.0f, 0.0f, 0.0f);
    for (float x = -1.0f; x <= 1.0f; x += 0.2f) {
        glVertex3f(x, -0.1f, -1.0f);
        glVertex3f(x, -0.15f, -1.0f);
    }

    // Z-axis ticks
    glColor3f(0.0f, 0.0f, 1.0f);
    for (float z = -1.0f; z <= 1.0f; z += 0.2f) {
        glVertex3f(-1.0f, -0.1f, z);
        glVertex3f(-1.0f, -0.15f, z);
    }

    // Y-axis ticks
    glColor3f(0.0f, 1.0f, 0.0f);
    for (float y = 0.0f; y <= 1.0f; y += 0.2f) {
        glVertex3f(-1.0f, y, -1.0f);
        glVertex3f(-1.05f, y, -1.0f);
    }
    glEnd();

    // Add special marker for S_0 on X-axis
    glLineWidth(3.0f);
    glBegin(GL_LINES);
    glColor3f(1.0f, 1.0f, 0.0f); // Yellow color for S_0
    // Normalize S_0 to [-1, 1] range
    float x_pos = -1.0f + 2.0f * (S_0 - min_strike) / (max_strike - min_strike);
    glVertex3f(x_pos, -0.1f, -1.0f);
    glVertex3f(x_pos, -0.25f, -1.0f); // Make it longer than regular ticks
    glEnd();

    // Add markers for maturity points on Z-axis
    glBegin(GL_LINES);
    glColor3f(0.0f, 1.0f, 1.0f); // Cyan color for maturity points
    for (double maturity : maturity_points) {
        // Normalize maturity to [-1, 1] range
        float z_pos = -1.0f + 2.0f * (maturity - min_maturity) / (max_maturity - min_maturity);
        glVertex3f(-1.0f, -0.1f, z_pos);
        glVertex3f(-1.0f, -0.25f, z_pos); // Make it longer than regular ticks
    }
    glEnd();
    glLineWidth(1.0f);
}

// Modify rendering function to handle both surfaces
void renderSurfaceData(const std::vector<std::vector<float>>& surface, float scale) {
    int width = surface.size();
    int height = surface[0].size();
    
    // Draw lines along width
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

//for american divident with indicators
void renderBothSurfaces(
    const std::vector<std::vector<float>>& price_surface, 
    const std::vector<std::vector<float>>& iv_surface, 
    float rotationX, float rotationY,
    float S_0, const std::vector<double>& maturity_points,
    float min_strike, float max_strike, float min_maturity, float max_maturity) {
    
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
    renderSurfaceData(price_surface, 0.01f);
    
    // Draw the main axes with thicker lines
    glLineWidth(2.0f);
    glBegin(GL_LINES);
    // X-axis (red) - Strike - moved to front corner
    glColor3f(1.0f, 0.0f, 0.0f);
    glVertex3f(-1.0f, -0.1f, -1.0f);
    glVertex3f(1.0f, -0.1f, -1.0f);
    
    // Y-axis (green) - Price - moved to front corner
    glColor3f(0.0f, 1.0f, 0.0f);
    glVertex3f(-1.0f, -0.1f, -1.0f);
    glVertex3f(-1.0f, 1.0f, -1.0f);
    
    // Z-axis (blue) - Maturity - moved to front corner
    glColor3f(0.0f, 0.0f, 1.0f);
    glVertex3f(-1.0f, -0.1f, -1.0f);
    glVertex3f(-1.0f, -0.1f, 1.0f);
    glEnd();
    
    // Reset line width
    glLineWidth(1.0f);
    
    // Draw tick marks on axes (adjusted for new position)
    glBegin(GL_LINES);
    // X-axis ticks
    glColor3f(1.0f, 0.0f, 0.0f);
    for (float x = -1.0f; x <= 1.0f; x += 0.2f) {
        glVertex3f(x, -0.1f, -1.0f);
        glVertex3f(x, -0.15f, -1.0f);
    }
    
    // Z-axis ticks
    glColor3f(0.0f, 0.0f, 1.0f);
    for (float z = -1.0f; z <= 1.0f; z += 0.2f) {
        glVertex3f(-1.0f, -0.1f, z);
        glVertex3f(-1.0f, -0.15f, z);
    }
    
    // Y-axis ticks
    glColor3f(0.0f, 1.0f, 0.0f);
    for (float y = 0.0f; y <= 1.0f; y += 0.2f) {
        glVertex3f(-1.0f, y, -1.0f);
        glVertex3f(-1.05f, y, -1.0f);
    }
    glEnd();
    
    // Add special marker for S_0 on X-axis
    glLineWidth(3.0f);
    glBegin(GL_LINES);
    glColor3f(1.0f, 1.0f, 0.0f); // Yellow color for S_0
    // Normalize S_0 to [-1, 1] range
    float x_pos = -1.0f + 2.0f * (S_0 - min_strike) / (max_strike - min_strike);
    // Clamp to valid range
    x_pos = std::min(std::max(x_pos, -1.0f), 1.0f);
    glVertex3f(x_pos, -0.1f, -1.0f);
    glVertex3f(x_pos, -0.25f, -1.0f); // Make it longer than regular ticks
    glEnd();
    
    // Add markers for maturity points on Z-axis
    glBegin(GL_LINES);
    glColor3f(0.0f, 1.0f, 1.0f); // Cyan color for maturity points
    for (double maturity : maturity_points) {
        // Normalize maturity to [-1, 1] range
        float z_pos = -1.0f + 2.0f * (maturity - min_maturity) / (max_maturity - min_maturity);
        // Clamp to valid range
        z_pos = std::min(std::max(z_pos, -1.0f), 1.0f);
        glVertex3f(-1.0f, -0.1f, z_pos);
        glVertex3f(-1.0f, -0.25f, z_pos); // Make it longer than regular ticks
    }
    glEnd();
    glLineWidth(1.0f);
    
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
    renderSurfaceData(iv_surface, 2.0f);
    
    // Draw the main axes with thicker lines (for right surface)
    glLineWidth(2.0f);
    glBegin(GL_LINES);
    // X-axis (red) - Strike - moved to front corner
    glColor3f(1.0f, 0.0f, 0.0f);
    glVertex3f(-1.0f, -0.1f, -1.0f);
    glVertex3f(1.0f, -0.1f, -1.0f);
    
    // Y-axis (green) - Price - moved to front corner
    glColor3f(0.0f, 1.0f, 0.0f);
    glVertex3f(-1.0f, -0.1f, -1.0f);
    glVertex3f(-1.0f, 1.0f, -1.0f);
    
    // Z-axis (blue) - Maturity - moved to front corner
    glColor3f(0.0f, 0.0f, 1.0f);
    glVertex3f(-1.0f, -0.1f, -1.0f);
    glVertex3f(-1.0f, -0.1f, 1.0f);
    glEnd();
    
    // Reset line width
    glLineWidth(1.0f);
    
    // Draw tick marks on axes (adjusted for new position)
    glBegin(GL_LINES);
    // X-axis ticks
    glColor3f(1.0f, 0.0f, 0.0f);
    for (float x = -1.0f; x <= 1.0f; x += 0.2f) {
        glVertex3f(x, -0.1f, -1.0f);
        glVertex3f(x, -0.15f, -1.0f);
    }
    
    // Z-axis ticks
    glColor3f(0.0f, 0.0f, 1.0f);
    for (float z = -1.0f; z <= 1.0f; z += 0.2f) {
        glVertex3f(-1.0f, -0.1f, z);
        glVertex3f(-1.0f, -0.15f, z);
    }
    
    // Y-axis ticks
    glColor3f(0.0f, 1.0f, 0.0f);
    for (float y = 0.0f; y <= 1.0f; y += 0.2f) {
        glVertex3f(-1.0f, y, -1.0f);
        glVertex3f(-1.05f, y, -1.0f);
    }
    glEnd();
    
    // Add special marker for S_0 on X-axis (right surface)
    glLineWidth(3.0f);
    glBegin(GL_LINES);
    glColor3f(1.0f, 1.0f, 0.0f); // Yellow color for S_0
    glVertex3f(x_pos, -0.1f, -1.0f);
    glVertex3f(x_pos, -0.25f, -1.0f); // Make it longer than regular ticks
    glEnd();
    
    // Add markers for maturity points on Z-axis (right surface)
    glBegin(GL_LINES);
    glColor3f(0.0f, 1.0f, 1.0f); // Cyan color for maturity points
    for (double maturity : maturity_points) {
        // Normalize maturity to [-1, 1] range
        float z_pos = -1.0f + 2.0f * (maturity - min_maturity) / (max_maturity - min_maturity);
        // Clamp to valid range
        z_pos = std::min(std::max(z_pos, -1.0f), 1.0f);
        glVertex3f(-1.0f, -0.1f, z_pos);
        glVertex3f(-1.0f, -0.25f, z_pos); // Make it longer than regular ticks
    }
    glEnd();
    glLineWidth(1.0f);
    
    // Reset viewport to full window
    glViewport(0, 0, SCR_WIDTH, SCR_HEIGHT);
}

//for european call without indicators
void renderBothSurfaces(
    const std::vector<std::vector<float>>& price_surface, 
    const std::vector<std::vector<float>>& iv_surface, 
    float S_0, float min_strike, float max_strike,
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
    renderSurfaceData(price_surface, 0.01f);
    renderAxes(S_0, min_strike, max_strike);
    
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
    renderSurfaceData(iv_surface, 2.0f);
    renderAxes(S_0, min_strike, max_strike);
    
    // Reset viewport to full window
    glViewport(0, 0, SCR_WIDTH, SCR_HEIGHT);
}



