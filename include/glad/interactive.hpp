#pragma once

#include <vector>


//Interactive Implied vol surfaces for european without and american call options with dividends
void interactive_european(
    double theta,
    int m1, int m2, int total_size,
    int width, int height,
    float min_strike, float max_strike,
    float min_maturity, float max_maturity,
    float S0, float v0,
    float& kappa, float& eta, float& sigma, float& rho, float& r_d, float& q
);

void interactive_american(
    double theta,
    int m1, int m2, int total_size,
    int width, int height,
    float min_strike, float max_strike,
    float min_maturity, float max_maturity,
    float S0, float v0,
    float& kappa, float& eta, float& sigma, float& rho, float& r_d, float& q,
    const std::vector<double>& dividend_dates,
    const std::vector<double>& dividend_amounts,
    const std::vector<double>& dividend_percentages
);

//Interactive calibration surfaces by Levenberg-Marquart 
void interactive_calibration_european(
    double theta,
    int m1, int m2, int total_size,
    int width, int height,
    float min_strike, float max_strike,
    float min_maturity, float max_maturity,
    float S_0, float v0,
    float& kappa, float& eta, float& sigma, float& rho, float& r_d, float& q
);

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
);