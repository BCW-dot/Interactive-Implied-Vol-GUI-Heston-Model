#pragma once

#include "device_solver.hpp"

struct CalibrationPoint {
    double strike;
    double maturity;
    int time_steps;      // N for this maturity
    double delta_t;      // Time step for this maturity
    int global_index;    // Flat index in the global arrays
};


// Compute implied volatility surface from option prices
void compute_implied_vol_surface(
    // Market parameters
    const double S_0, const double r_d,
    // Numerical parameters
    const int width, const int height,
    // Pre-computed data structures
    const Kokkos::View<CalibrationPoint*>& d_calibration_points,
    // Input prices and output implied volatilities
    const Kokkos::View<double*>& base_prices,
    Kokkos::View<double*>& implied_vols,
    // Team policy for parallelization
    const Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace>& policy
);

//Solves an implicit 5x5 matrix on the gpu
void solve_5x5_device(
    const Kokkos::View<double**> &A_device,  // shape (5,5)
    const Kokkos::View<double*>  &b_device,  // shape (5)
    const Kokkos::View<double*>  &x_device   // shape (5)
);

//Perfoms the LVBM steps
void compute_parameter_update_on_device(
    const Kokkos::View<double**>& J,        // [num_data x 5]
    const Kokkos::View<double*>&  residual, // [num_data]
    const double                  lambda,
    Kokkos::View<double*>&        delta     // [5]
);


void compute_base_prices_multi_maturity(
    // Market/model parameters
    const double S_0, const double V_0,
    const double r_d, const double r_f,
    const double rho, const double sigma, const double kappa, const double eta,
    // Numerical parameters
    const int m1, const int m2, const int total_size,
    const double theta,
    // Calibration points
    const Kokkos::View<CalibrationPoint*>& d_calibration_points,
    // Pre-computed data structures
    const int total_calibration_size,
    const Kokkos::View<Device_A0_heston<Kokkos::DefaultExecutionSpace>*>& A0_solvers,
    const Kokkos::View<Device_A1_heston<Kokkos::DefaultExecutionSpace>*>& A1_solvers,
    const Kokkos::View<Device_A2_shuffled_heston<Kokkos::DefaultExecutionSpace>*>& A2_solvers,
    const Kokkos::View<Device_BoundaryConditions<Kokkos::DefaultExecutionSpace>*>& bounds_d,
    const Kokkos::View<GridViews*>& deviceGrids,
    DO_Workspace<Kokkos::DefaultExecutionSpace>& workspace,
    Kokkos::View<double*>& base_prices,
    const Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace>& policy
);

void compute_base_prices_multi_maturity_american_dividends(
    // Market/model parameters
    const double S_0, const double V_0,
    const double r_d, const double r_f,
    const double rho, const double sigma, const double kappa, const double eta,
    // Numerical parameters
    const int m1, const int m2, const int total_size,
    const double theta,
    // Calibration points
    const Kokkos::View<CalibrationPoint*>& d_calibration_points,
    // Pre-computed data structures
    const int total_calibration_size,
    const Kokkos::View<Device_A0_heston<Kokkos::DefaultExecutionSpace>*>& A0_solvers,
    const Kokkos::View<Device_A1_heston<Kokkos::DefaultExecutionSpace>*>& A1_solvers,
    const Kokkos::View<Device_A2_shuffled_heston<Kokkos::DefaultExecutionSpace>*>& A2_solvers,
    const Kokkos::View<Device_BoundaryConditions<Kokkos::DefaultExecutionSpace>*>& bounds_d,
    const Kokkos::View<GridViews*>& deviceGrids,
    const Kokkos::View<double**>& U_0,
    DO_Workspace<Kokkos::DefaultExecutionSpace>& workspace,
    // Dividend specifics
    const int num_dividends,                        
    const Kokkos::View<double*>& dividend_dates,    
    const Kokkos::View<double*>& dividend_amounts,  
    const Kokkos::View<double*>& dividend_percentages,
    Kokkos::View<double*>& base_prices,
    const Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace>& policy
);


// Function to compute Jacobian matrix of a european options at multiply maturities
void compute_jacobian_multi_maturity(
    // Market/model parameters
    const double S_0, const double V_0,
    const double r_d, const double r_f,
    const double rho, const double sigma, const double kappa, const double eta,
    // Numerical parameters
    const int m1, const int m2, const int total_size,
    const double theta,
    // Calibration points
    const Kokkos::View<CalibrationPoint*>& d_calibration_points,  // Note the & here
    // Pre-computed data structures
    const int total_calibration_size,
    const Kokkos::View<Device_A0_heston<Kokkos::DefaultExecutionSpace>*>& A0_solvers,
    const Kokkos::View<Device_A1_heston<Kokkos::DefaultExecutionSpace>*>& A1_solvers,
    const Kokkos::View<Device_A2_shuffled_heston<Kokkos::DefaultExecutionSpace>*>& A2_solvers,
    const Kokkos::View<Device_BoundaryConditions<Kokkos::DefaultExecutionSpace>*>& bounds_d,
    const Kokkos::View<GridViews*>& deviceGrids,
    const Kokkos::View<double**>& U_0,
    DO_Workspace<Kokkos::DefaultExecutionSpace>& workspace,
    // Output matrix
    Kokkos::View<double**>& J,
    Kokkos::View<double*>& base_prices,
    const Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace>& policy,
    // Optional: perturbation size
    const double eps = 1e-6
);

// Function to compute Jacobian matrix for american options with dividends at multiple maturities
void compute_jacobian_multi_maturity_american_dividends(
    // Market/model parameters
    const double S_0, const double V_0,
    const double r_d, const double r_f,
    const double rho, const double sigma, const double kappa, const double eta,
    // Numerical parameters
    const int m1, const int m2, const int total_size,
    const double theta,
    // Calibration points
    const Kokkos::View<CalibrationPoint*>& d_calibration_points,
    // Pre-computed data structures
    const int total_calibration_size,
    const Kokkos::View<Device_A0_heston<Kokkos::DefaultExecutionSpace>*>& A0_solvers,
    const Kokkos::View<Device_A1_heston<Kokkos::DefaultExecutionSpace>*>& A1_solvers,
    const Kokkos::View<Device_A2_shuffled_heston<Kokkos::DefaultExecutionSpace>*>& A2_solvers,
    const Kokkos::View<Device_BoundaryConditions<Kokkos::DefaultExecutionSpace>*>& bounds_d,
    const Kokkos::View<GridViews*>& deviceGrids,
    const Kokkos::View<double**>& U_0,
    DO_Workspace<Kokkos::DefaultExecutionSpace>& workspace,
    // Dividend specifics
    const int num_dividends,                        
    const Kokkos::View<double*>& dividend_dates,    
    const Kokkos::View<double*>& dividend_amounts,  
    const Kokkos::View<double*>& dividend_percentages,
    // Output matrix
    Kokkos::View<double**>& J,
    Kokkos::View<double*>& base_prices,
    const Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace>& policy,
    // Optional: perturbation size
    const double eps = 1e-6
);