#include "base_prices.hpp"


//Computes the base prices of european call options on the GPU
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
) {
    //using Device = Kokkos::DefaultExecutionSpace;
    // Create team policy
    //using team_policy = Kokkos::TeamPolicy<>;
    //team_policy policy(total_calibration_size, Kokkos::AUTO);

    // Main computation kernel 
    Kokkos::parallel_for("Base_Price_computation", policy,
        KOKKOS_LAMBDA(const typename Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace>::member_type& team) {
            const int instance = team.league_rank();
            
            // Get calibration point data
            const CalibrationPoint& point = d_calibration_points(instance);
            const int N = point.time_steps;
            const double delta_t = point.delta_t;
            
            // Setup workspace views
            auto U_i = Kokkos::subview(workspace.U, instance, Kokkos::ALL);
            auto Y_0_i = Kokkos::subview(workspace.Y_0, instance, Kokkos::ALL);
            auto Y_1_i = Kokkos::subview(workspace.Y_1, instance, Kokkos::ALL);
            auto A0_result_i = Kokkos::subview(workspace.A0_result, instance, Kokkos::ALL);
            auto A1_result_i = Kokkos::subview(workspace.A1_result, instance, Kokkos::ALL);
            auto A2_result_unshuf_i = Kokkos::subview(workspace.A2_result_unshuf, instance, Kokkos::ALL);
            
            auto U_shuffled_i = Kokkos::subview(workspace.U_shuffled, instance, Kokkos::ALL);
            auto Y_1_shuffled_i = Kokkos::subview(workspace.Y_1_shuffled, instance, Kokkos::ALL);
            auto A2_result_shuffled_i = Kokkos::subview(workspace.A2_result_shuffled, instance, Kokkos::ALL);
            auto U_next_shuffled_i = Kokkos::subview(workspace.U_next_shuffled, instance, Kokkos::ALL);
            
            GridViews grid_i = deviceGrids(instance);

            grid_i.rebuild_variance_views(V_0, 5.0, 5.0/500, team);
            
            bounds_d(instance).initialize(grid_i, team);
            auto bounds = bounds_d(instance);
            
            A0_solvers(instance).build_matrix(grid_i, rho, sigma, team);
            A1_solvers(instance).build_matrix(grid_i, r_d, r_f, theta, delta_t, team);
            A2_solvers(instance).build_matrix(grid_i, r_d, kappa, eta, sigma, theta, delta_t, team);

            // Use instance-specific time steps and delta_t
            device_DO_timestepping<Kokkos::DefaultExecutionSpace, decltype(U_i)>(
                m1, m2, N, delta_t, theta, r_f,
                A0_solvers(instance), A1_solvers(instance), A2_solvers(instance),
                bounds,
                U_i, Y_0_i, Y_1_i,
                A0_result_i, A1_result_i, A2_result_unshuf_i,
                U_shuffled_i, Y_1_shuffled_i, A2_result_shuffled_i, U_next_shuffled_i,
                team
            );
            
            // Find spot price index
            int index_s = -1;
            for(int i = 0; i <= m1; i++) {
                if(Kokkos::abs(grid_i.device_Vec_s(i) - S_0) < 1e-10) {
                    index_s = i;
                    break;
                }
            }
            const int index_v = grid_i.find_v0_index(V_0);

            // Store base price
            base_prices(instance) = U_i(index_s + index_v*(m1+1));
        });
    Kokkos::fence();
}


//Computes the base prices of an american call with a divident paying stock
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
) {
    // Main computation kernel 
    Kokkos::parallel_for("Base_Price_computation", policy,
        KOKKOS_LAMBDA(const typename Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace>::member_type& team) {
            const int instance = team.league_rank();
            
            // Get calibration point data
            const CalibrationPoint& point = d_calibration_points(instance);
            const int N = point.time_steps;
            const double delta_t = point.delta_t;
            
            // Setup workspace views
            auto U_i = Kokkos::subview(workspace.U, instance, Kokkos::ALL);
            auto Y_0_i = Kokkos::subview(workspace.Y_0, instance, Kokkos::ALL);
            auto Y_1_i = Kokkos::subview(workspace.Y_1, instance, Kokkos::ALL);
            auto A0_result_i = Kokkos::subview(workspace.A0_result, instance, Kokkos::ALL);
            auto A1_result_i = Kokkos::subview(workspace.A1_result, instance, Kokkos::ALL);
            auto A2_result_unshuf_i = Kokkos::subview(workspace.A2_result_unshuf, instance, Kokkos::ALL);
            
            auto U_shuffled_i = Kokkos::subview(workspace.U_shuffled, instance, Kokkos::ALL);
            auto Y_1_shuffled_i = Kokkos::subview(workspace.Y_1_shuffled, instance, Kokkos::ALL);
            auto A2_result_shuffled_i = Kokkos::subview(workspace.A2_result_shuffled, instance, Kokkos::ALL);
            auto U_next_shuffled_i = Kokkos::subview(workspace.U_next_shuffled, instance, Kokkos::ALL);

            auto U_0_i = Kokkos::subview(U_0, instance, Kokkos::ALL);  // Get initial condition
            
            // American specific
            auto lambda_bar_i = Kokkos::subview(workspace.lambda_bar, instance, Kokkos::ALL);
            
            // Dividend specifics
            auto U_temp_i = Kokkos::subview(workspace.U_temp, instance, Kokkos::ALL);

            GridViews grid_i = deviceGrids(instance);
            auto device_Vec_s_i = grid_i.device_Vec_s;

            grid_i.rebuild_variance_views(V_0, 5.0, 5.0/500, team);
            
            bounds_d(instance).initialize(grid_i, team);
            auto bounds = bounds_d(instance);
            
            A0_solvers(instance).build_matrix(grid_i, rho, sigma, team);
            A1_solvers(instance).build_matrix(grid_i, r_d, r_f, theta, delta_t, team);
            A2_solvers(instance).build_matrix(grid_i, r_d, kappa, eta, sigma, theta, delta_t, team);

            // Use instance-specific time steps and delta_t
            device_DO_timestepping_american_dividend<Kokkos::DefaultExecutionSpace, decltype(U_i)>(
                m1, m2, N, delta_t, theta, r_f,
                A0_solvers(instance), A1_solvers(instance), A2_solvers(instance),
                bounds,
                U_i, Y_0_i, Y_1_i,
                A0_result_i, A1_result_i, A2_result_unshuf_i,
                U_shuffled_i, Y_1_shuffled_i, A2_result_shuffled_i, U_next_shuffled_i,
                lambda_bar_i, U_0_i,
                num_dividends,
                dividend_dates,     
                dividend_amounts,   
                dividend_percentages, 
                device_Vec_s_i,        
                U_temp_i,  
                team
            );
            
            // Find spot price index
            int index_s = -1;
            for(int i = 0; i <= m1; i++) {
                if(Kokkos::abs(grid_i.device_Vec_s(i) - S_0) < 1e-10) {
                    index_s = i;
                    break;
                }
            }
            const int index_v = grid_i.find_v0_index(V_0);

            // Store base price
            base_prices(instance) = U_i(index_s + index_v*(m1+1));
        });
    Kokkos::fence();
}



//gets the base prices and computes for each the implied volatility at this point on the gpu
void compute_implied_vol_surface(
    const double S_0, const double r_d,
    const int width, const int height,
    const Kokkos::View<CalibrationPoint*>& d_calibration_points,
    const Kokkos::View<double*>& base_prices,
    Kokkos::View<double*>& implied_vols,
    const Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace>& policy
) {
    using Device = Kokkos::DefaultExecutionSpace;
    
    // Run the computation in parallel for each strike/maturity pair
    Kokkos::parallel_for("ImpliedVolComputation", policy,
        KOKKOS_LAMBDA(const typename Kokkos::TeamPolicy<Device>::member_type& team) {
            const int idx = team.league_rank();
            
            // Get calibration point information
            CalibrationPoint point = d_calibration_points(idx);
            double K = point.strike;
            double T = point.maturity;
            
            // Get the option price
            double option_price = base_prices(idx);
            
            // Newton-Raphson method for implied volatility with fallback to bisection
            double vol = 0.0;
            
            // Initial guess for volatility (midpoint of reasonable range)
            double x = 0.3;
            
            // Target option price
            double C_target = option_price;
            
            // Tolerance for convergence
            const double epsilon = 1e-6;
            
            // Maximum iterations
            const int MAX_ITER = 50;
            
            // Newton-Raphson iterations
            bool fail = false;
            
            // Price and vega for current volatility estimate
            double C = 0.0;
            double V = 0.0;
            
            for (int iter = 0; iter < MAX_ITER && !fail; ++iter) {
                // Black-Scholes call price
                double d1 = (Kokkos::log(S_0 / K) + (r_d + 0.5 * x * x) * T) / (x * Kokkos::sqrt(T));
                double d2 = d1 - x * Kokkos::sqrt(T);
                
                // Using cumulative normal distribution approximation
                double nd1 = 0.5 * (1.0 + Kokkos::erf(d1 / Kokkos::sqrt(2.0)));
                double nd2 = 0.5 * (1.0 + Kokkos::erf(d2 / Kokkos::sqrt(2.0)));
                
                C = S_0 * nd1 - K * Kokkos::exp(-r_d * T) * nd2;
                
                // Break if we're within tolerance
                if (Kokkos::abs(C - C_target) < epsilon) {
                    break;
                }
                
                // Calculate vega
                V = S_0 * Kokkos::sqrt(T / (2.0 * M_PI)) * Kokkos::exp(-d1 * d1 / 2.0);
                
                // Check if vega is too small
                if (Kokkos::abs(V) < 1e-10) {
                    fail = true;
                    break;
                }
                
                // Update volatility estimate
                x = x - (C - C_target) / V;
                
                // Ensure volatility stays in a reasonable range
                x = Kokkos::max(0.001, Kokkos::min(2.0, x));
            }
            
            // If Newton-Raphson failed, use bisection method
            if (fail) {
                // Bisection parameters
                double a = 0.001;  // Min volatility
                double b = 2.0;    // Max volatility
                
                // Bisection iterations
                for (int iter = 0; iter < MAX_ITER; ++iter) {
                    x = (a + b) / 2.0;
                    
                    // Calculate BS price at midpoint
                    double d1 = (Kokkos::log(S_0 / K) + (r_d + 0.5 * x * x) * T) / (x * Kokkos::sqrt(T));
                    double d2 = d1 - x * Kokkos::sqrt(T);
                    
                    double nd1 = 0.5 * (1.0 + Kokkos::erf(d1 / Kokkos::sqrt(2.0)));
                    double nd2 = 0.5 * (1.0 + Kokkos::erf(d2 / Kokkos::sqrt(2.0)));
                    
                    C = S_0 * nd1 - K * Kokkos::exp(-r_d * T) * nd2;
                    
                    // Break if we're within tolerance
                    if (Kokkos::abs(C - C_target) < epsilon) {
                        break;
                    }
                    
                    // Update interval
                    if (C > C_target) {
                        b = x;
                    } else {
                        a = x;
                    }
                }
            }
            
            // Store the implied volatility
            implied_vols(idx) = x;
        });
    
    Kokkos::fence();
}
