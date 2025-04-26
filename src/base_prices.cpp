#include "base_prices.hpp"

#include <KokkosBlas2_gemv.hpp> // for gemv
#include <KokkosBlas3_gemm.hpp> // for gemm


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


void solve_5x5_device(
    const Kokkos::View<double**> &A_device,  // shape (5,5)
    const Kokkos::View<double*>  &b_device,  // shape (5)
    const Kokkos::View<double*>  &x_device   // shape (5)
){
  // We run one kernel with a single iteration so that everything is done on GPU.
  Kokkos::parallel_for("solve_5x5", Kokkos::RangePolicy<>(0, 1),
    KOKKOS_LAMBDA(const int /*dummy*/)
    {
      constexpr int N = 5;

      // 1) Copy A_device, b_device into local arrays in GPU registers/shared memory.
      double A[25]; // row-major: A[i*N + j]
      double b[5];
      for (int i = 0; i < N; i++) {
        b[i] = b_device(i);
        for (int j = 0; j < N; j++) {
          A[i*N + j] = A_device(i,j);
        }
      }

      // 2) Perform partial pivot Gaussian elimination in-place.
      //    We'll pivot on the largest absolute value in the column for each step.
      for (int k = 0; k < N; k++) {

        // 2a) Find pivot row = row p in [k..N-1] with max |A[p,k]|.
        double maxA = Kokkos::abs(A[k*N + k]);
        int pivotRow = k;
        for(int p = k+1; p < N; p++){
          double val = Kokkos::abs(A[p*N + k]);
          if(val > maxA){
            maxA = val;
            pivotRow = p;
          }
        }
        // If pivotRow != k, swap the two rows in A and the entries in b.
        if(pivotRow != k){
          for(int col = 0; col < N; col++){
            double tmp = A[k*N + col];
            A[k*N + col] = A[pivotRow*N + col];
            A[pivotRow*N + col] = tmp;
          }
          double tmpb = b[k];
          b[k] = b[pivotRow];
          b[pivotRow] = tmpb;
        }

        // 2b) Divide pivot row by pivot
        double pivot = A[k*N + k];
        // (Assume matrix is non-singular, pivot != 0)
        for(int col = k+1; col < N; col++){
          A[k*N + col] /= pivot;
        }
        b[k] /= pivot;
        A[k*N + k] = 1.0;

        // 2c) Eliminate below pivot
        for(int i = k+1; i < N; i++){
          double factor = A[i*N + k];
          for(int col = k+1; col < N; col++){
            A[i*N + col] -= factor * A[k*N + col];
          }
          b[i] -= factor * b[k];
          A[i*N + k] = 0.0;
        }
      }

      // 3) Back-substitution
      for(int k = N-1; k >= 0; k--){
        double val = b[k];
        for(int col = k+1; col < N; col++){
          val -= A[k*N + col] * b[col];
        }
        b[k] = val;
      }

      // 4) Now b[] holds the solution. Copy to x_device.
      for(int i = 0; i < N; i++){
        x_device(i) = b[i];
      }
    }
  ); // parallel_for

  Kokkos::fence();
}


void solve_5x5_host(
    const Kokkos::View<double**> &A_device,  // shape (5,5)
    const Kokkos::View<double*>  &b_device,  // shape (5)
    const Kokkos::View<double*>  &x_device   // shape (5)
) {
    constexpr int N = 5;

    // 1) Deep copy A_device, b_device into host memory
    auto A_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), A_device);
    auto b_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), b_device);
    auto x_host = Kokkos::create_mirror_view(Kokkos::HostSpace(), x_device); // x_host to store the result

    // 2) Local arrays for solving
    double A[25]; // row-major: A[i*N + j]
    double b[5];

    for (int i = 0; i < N; i++) {
        b[i] = b_host(i);
        for (int j = 0; j < N; j++) {
            A[i*N + j] = A_host(i,j);
        }
    }

    // 3) Perform Gaussian elimination with partial pivoting
    for (int k = 0; k < N; k++) {
        // Find pivot
        double maxA = std::abs(A[k*N + k]);
        int pivotRow = k;
        for (int p = k+1; p < N; p++) {
            double val = std::abs(A[p*N + k]);
            if (val > maxA) {
                maxA = val;
                pivotRow = p;
            }
        }

        // Swap rows if needed
        if (pivotRow != k) {
            for (int col = 0; col < N; col++) {
                double tmp = A[k*N + col];
                A[k*N + col] = A[pivotRow*N + col];
                A[pivotRow*N + col] = tmp;
            }
            double tmpb = b[k];
            b[k] = b[pivotRow];
            b[pivotRow] = tmpb;
        }

        // Normalize pivot row
        double pivot = A[k*N + k];
        for (int col = k+1; col < N; col++) {
            A[k*N + col] /= pivot;
        }
        b[k] /= pivot;
        A[k*N + k] = 1.0;

        // Eliminate below
        for (int i = k+1; i < N; i++) {
            double factor = A[i*N + k];
            for (int col = k+1; col < N; col++) {
                A[i*N + col] -= factor * A[k*N + col];
            }
            b[i] -= factor * b[k];
            A[i*N + k] = 0.0;
        }
    }

    // 4) Back-substitution
    for (int k = N-1; k >= 0; k--) {
        double val = b[k];
        for (int col = k+1; col < N; col++) {
            val -= A[k*N + col] * b[col];
        }
        b[k] = val;
    }

    // 5) Copy solution into x_host
    for (int i = 0; i < N; i++) {
        x_host(i) = b[i];
    }

    // 6) Deep copy x_host back to x_device
    Kokkos::deep_copy(x_device, x_host);
}


// This is the same as compute_parameter_update, but no KokkosBatched:
void compute_parameter_update_on_device(
    const Kokkos::View<double**>& J,        // [num_data x 5]
    const Kokkos::View<double*>&  residual, // [num_data]
    const double                  lambda,
    Kokkos::View<double*>&        delta     // [5]
){
    constexpr int N = 5;

    // 1. Build J^T J => [5 x 5]
    Kokkos::View<double**> JTJ("JTJ", N, N);
    KokkosBlas::gemm("T", "N", 1.0, J, J, 0.0, JTJ);

    // 2. Add lambda on diagonal
    Kokkos::parallel_for("add_lambda_diag", N, KOKKOS_LAMBDA(const int i){
        JTJ(i,i) *= (1.0 + lambda);
    });

    // 3. Build J^T r => [5]
    Kokkos::View<double*> JTr("JTr", N);
    KokkosBlas::gemv("T", 1.0, J, residual, 0.0, JTr);

    // 4. Solve (JTJ) * delta = (JTr) on device with our manual 5x5 routine
    solve_5x5_device(JTJ, JTr, delta);
    //std::cout<< "Solving on host";
    //solve_5x5_host(JTJ, JTr, delta);
}


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
            const double K = point.strike;
            
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
            grid_i.rebuild_stock_views(S_0, 8*K, K, K/5, team);
            
            bounds_d(instance).initialize(grid_i, r_d, r_f, team);
            auto bounds = bounds_d(instance);
            
            A0_solvers(instance).build_matrix(grid_i, rho, sigma, team);
            A1_solvers(instance).build_matrix(grid_i, r_d, r_f, theta, delta_t, team);
            A2_solvers(instance).build_matrix(grid_i, r_d, kappa, eta, sigma, theta, delta_t, team);

            // Initialize U_i with payoff directly on device
            Kokkos::parallel_for(Kokkos::TeamThreadRange(team, m2+1), 
            [&](const int j) {
                for(int i = 0; i <= m1; i++) {
                    // Compute max(S-K, 0) for each point
                    double s_value = grid_i.device_Vec_s(i);
                    double payoff = Kokkos::max(s_value - K, 0.0);
                    U_i(i + j*(m1+1)) = payoff;
                }
            });
            team.team_barrier();

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
            
            // Find spot and variance price index
            const int index_s = grid_i.find_s0_index(S_0);
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
            const double K = point.strike;
            
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
            grid_i.rebuild_stock_views(S_0, 8*K, K, K/5, team);
            grid_i.rebuild_variance_views(V_0, 5.0, 5.0/500, team);
            auto device_Vec_s_i = grid_i.device_Vec_s;
            
            bounds_d(instance).initialize(grid_i, r_d, r_f, team);
            auto bounds = bounds_d(instance);
            
            A0_solvers(instance).build_matrix(grid_i, rho, sigma, team);
            A1_solvers(instance).build_matrix(grid_i, r_d, r_f, theta, delta_t, team);
            A2_solvers(instance).build_matrix(grid_i, r_d, kappa, eta, sigma, theta, delta_t, team);

            // Initialize U_i with payoff directly on device
            Kokkos::parallel_for(Kokkos::TeamThreadRange(team, m2+1), 
            [&](const int j) {
                for(int i = 0; i <= m1; i++) {
                    // Compute max(S-K, 0) for each point
                    double s_value = grid_i.device_Vec_s(i);
                    double payoff = Kokkos::max(s_value - K, 0.0);
                    U_i(i + j*(m1+1)) = payoff;
                    U_0_i(i + j*(m1+1)) = payoff;
                }
            });
            team.team_barrier();

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
            
            // Find spot and variance price index
            const int index_s = grid_i.find_s0_index(S_0);
            const int index_v = grid_i.find_v0_index(V_0);

            // Store base price
            base_prices(instance) = U_i(index_s + index_v*(m1+1));
        });
    Kokkos::fence();
}



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
    const double eps
) {
    // Create team policy for the total number of calibration points
    //using Device = Kokkos::DefaultExecutionSpace;
    //using team_policy = Kokkos::TeamPolicy<>;
    //team_policy policy(total_calibration_size, Kokkos::AUTO);


    Kokkos::parallel_for("Jacobian_computation", policy,
        KOKKOS_LAMBDA(const typename Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace>::member_type& team) {
            const int instance = team.league_rank();
            
            // Get calibration point data
            const CalibrationPoint& point = d_calibration_points(instance);
            const int N = point.time_steps;
            const double delta_t = point.delta_t;
            const double K = point.strike;
    
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
            grid_i.rebuild_stock_views(S_0, 8*K, K, K/5, team);
            
            bounds_d(instance).initialize(grid_i, r_d, r_f, team);
            auto bounds = bounds_d(instance);
            
            A0_solvers(instance).build_matrix(grid_i, rho, sigma, team);
            A1_solvers(instance).build_matrix(grid_i, r_d, r_f, theta, delta_t, team);
            A2_solvers(instance).build_matrix(grid_i, r_d, kappa, eta, sigma, theta, delta_t, team);
            
            // Initialize U_i with payoff directly on device
            Kokkos::parallel_for(Kokkos::TeamThreadRange(team, m2+1), 
            [&](const int j) {
                for(int i = 0; i <= m1; i++) {
                    // Compute max(S-K, 0) for each point
                    double s_value = grid_i.device_Vec_s(i);
                    double payoff = Kokkos::max(s_value - K, 0.0);
                    U_i(i + j*(m1+1)) = payoff;
                }
            });
            team.team_barrier();
            
            // Use instance-specific time steps and delta_t from calibration point
            device_DO_timestepping<Kokkos::DefaultExecutionSpace, decltype(U_i)>(
                m1, m2, N, delta_t, theta, r_f,
                A0_solvers(instance), A1_solvers(instance), A2_solvers(instance),
                bounds,
                U_i, Y_0_i, Y_1_i,
                A0_result_i, A1_result_i, A2_result_unshuf_i,
                U_shuffled_i, Y_1_shuffled_i, A2_result_shuffled_i, U_next_shuffled_i,
                team
            );
            
            // Find spot and variance price index
            const int index_s = grid_i.find_s0_index(S_0);
            const int index_v = grid_i.find_v0_index(V_0);

            // Store base price
            base_prices(instance) = U_i(index_s + index_v*(m1+1));
    
            // Loop over parameters for finite differences
            for(int param = 0; param < 4; param++) {
                // Handle parameters other than V0
                double kappa_p = kappa;
                double eta_p = eta;
                double sigma_p = sigma;
                double rho_p = rho;
    
                switch(param) {
                    case 0: kappa_p += eps; break;
                    case 1: eta_p += eps; break;
                    case 2: sigma_p += eps; break;
                    case 3: rho_p += eps; break;
                }
    
                // Initialize U_i with payoff directly on device
                Kokkos::parallel_for(Kokkos::TeamThreadRange(team, m2+1), 
                [&](const int j) {
                    for(int i = 0; i <= m1; i++) {
                        // Compute max(S-K, 0) for each point
                        double s_value = grid_i.device_Vec_s(i);
                        double payoff = Kokkos::max(s_value - K, 0.0);
                        U_i(i + j*(m1+1)) = payoff;
                    }
                });
                team.team_barrier();
    
                // Rebuild matrices with perturbed parameter
                A0_solvers(instance).build_matrix(grid_i, rho_p, sigma_p, team);
                A1_solvers(instance).build_matrix(grid_i, r_d, r_f, theta, delta_t, team);
                A2_solvers(instance).build_matrix(grid_i, r_d, kappa_p, eta_p, sigma_p, theta, delta_t, team);
    
                // Compute perturbed solution with maturity-specific time steps
                device_DO_timestepping<Kokkos::DefaultExecutionSpace, decltype(U_i)>(
                    m1, m2, N, delta_t, theta, r_f,
                    A0_solvers(instance), A1_solvers(instance), A2_solvers(instance),
                    bounds, U_i, Y_0_i, Y_1_i,
                    A0_result_i, A1_result_i, A2_result_unshuf_i,
                    U_shuffled_i, Y_1_shuffled_i, A2_result_shuffled_i, U_next_shuffled_i,
                    team
                );
    
                // Store results
                double pert_price = U_i(index_s + index_v*(m1+1));
                J(instance, param) = (pert_price - base_prices(instance)) / eps;
            }
    
            // Special handling for V0 (param == 4)
            const int param = 4;
            // Initialize U_i with payoff directly on device
            Kokkos::parallel_for(Kokkos::TeamThreadRange(team, m2+1), 
            [&](const int j) {
                for(int i = 0; i <= m1; i++) {
                    // Compute max(S-K, 0) for each point
                    double s_value = grid_i.device_Vec_s(i);
                    double payoff = Kokkos::max(s_value - K, 0.0);
                    U_i(i + j*(m1+1)) = payoff;
                }
            });
            team.team_barrier();
    
            // Rebuild variance views with perturbed V0
            grid_i.rebuild_variance_views(V_0 + eps, 5.0, 5.0/500, team);
            const int index_v_pertubed = grid_i.find_v0_index(V_0 + eps);
            
            // Rebuild matrices with updated grid
            A0_solvers(instance).build_matrix(grid_i, rho, sigma, team);
            A1_solvers(instance).build_matrix(grid_i, r_d, r_f, theta, delta_t, team);
            A2_solvers(instance).build_matrix(grid_i, r_d, kappa, eta, sigma, theta, delta_t, team);
    
            // Compute perturbed solution
            device_DO_timestepping<Kokkos::DefaultExecutionSpace, decltype(U_i)>(
                m1, m2, N, delta_t, theta, r_f,
                A0_solvers(instance), A1_solvers(instance), A2_solvers(instance),
                bounds, U_i, Y_0_i, Y_1_i,
                A0_result_i, A1_result_i, A2_result_unshuf_i,
                U_shuffled_i, Y_1_shuffled_i, A2_result_shuffled_i, U_next_shuffled_i,
                team
            );
    
            // Get price and compute gradient
            double pert_price = U_i(index_s + index_v_pertubed*(m1+1));
            J(instance, param) = (pert_price - base_prices(instance)) / eps;
        });
    Kokkos::fence();
}

//americans
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
    const double eps
) {
    Kokkos::parallel_for("Jacobian_computation", policy,
        KOKKOS_LAMBDA(const typename Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace>::member_type& team) {
            const int instance = team.league_rank();
            
            // Get calibration point data
            const CalibrationPoint& point = d_calibration_points(instance);
            const int N = point.time_steps;
            const double delta_t = point.delta_t;
            const double K = point.strike;
    
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
            grid_i.rebuild_stock_views(S_0, 8*K, K, K/5, team);
            
            bounds_d(instance).initialize(grid_i, r_d, r_f, team);
            auto bounds = bounds_d(instance);
            
            A0_solvers(instance).build_matrix(grid_i, rho, sigma, team);
            A1_solvers(instance).build_matrix(grid_i, r_d, r_f, theta, delta_t, team);
            A2_solvers(instance).build_matrix(grid_i, r_d, kappa, eta, sigma, theta, delta_t, team);

            // Initialize U_i with payoff directly on device
            Kokkos::parallel_for(Kokkos::TeamThreadRange(team, m2+1), 
            [&](const int j) {
                for(int i = 0; i <= m1; i++) {
                    // Compute max(S-K, 0) for each point
                    double s_value = grid_i.device_Vec_s(i);
                    double payoff = Kokkos::max(s_value - K, 0.0);
                    U_i(i + j*(m1+1)) = payoff;
                    U_0_i(i + j*(m1+1)) = payoff;
                }
            });
            team.team_barrier();

    
            // Call device timestepping for American options with dividends
            device_DO_timestepping_american_dividend<Kokkos::DefaultExecutionSpace, decltype(U_i)>(
                m1, m2, N, delta_t, theta, r_f,
                A0_solvers(instance), A1_solvers(instance), A2_solvers(instance),
                bounds,
                U_i, Y_0_i, Y_1_i,
                A0_result_i, A1_result_i, A2_result_unshuf_i,
                U_shuffled_i, Y_1_shuffled_i, A2_result_shuffled_i, U_next_shuffled_i,
                lambda_bar_i, U_0_i,
                // Dividend - Use device views
                num_dividends,
                dividend_dates,     
                dividend_amounts,   
                dividend_percentages, 
                device_Vec_s_i,        
                U_temp_i,  
                team
            );
            
            // Find spot and variance price index
            const int index_s = grid_i.find_s0_index(S_0);
            const int index_v = grid_i.find_v0_index(V_0);
    
            // Store base price
            base_prices(instance) = U_i(index_s + index_v*(m1+1));
    
            // Loop over parameters for finite differences
            for(int param = 0; param < 4; param++) {
                // Handle parameters other than V0
                double kappa_p = kappa;
                double eta_p = eta;
                double sigma_p = sigma;
                double rho_p = rho;
    
                switch(param) {
                    case 0: kappa_p += eps; break;
                    case 1: eta_p += eps; break;
                    case 2: sigma_p += eps; break;
                    case 3: rho_p += eps; break;
                }
    
                // Reset initial condition
                Kokkos::parallel_for(Kokkos::TeamThreadRange(team, m2+1), 
                [&](const int j) {
                    for(int i = 0; i <= m1; i++) {
                        // Compute max(S-K, 0) for each point
                        double s_value = grid_i.device_Vec_s(i);
                        double payoff = Kokkos::max(s_value - K, 0.0);
                        U_i(i + j*(m1+1)) = payoff;
                        U_0_i(i + j*(m1+1)) = payoff;
                    }
                });
                team.team_barrier();
    
                // Reset lambda (needed for American option)
                for(int idx = 0; idx < total_size; idx++) {
                    lambda_bar_i(idx) = 0.0;
                }
    
                // Rebuild matrices with perturbed parameter
                A0_solvers(instance).build_matrix(grid_i, rho_p, sigma_p, team);
                A1_solvers(instance).build_matrix(grid_i, r_d, r_f, theta, delta_t, team);
                A2_solvers(instance).build_matrix(grid_i, r_d, kappa_p, eta_p, sigma_p, theta, delta_t, team);
    
                // Compute perturbed solution
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
    
                // Store results
                double pert_price = U_i(index_s + index_v*(m1+1));
                J(instance, param) = (pert_price - base_prices(instance)) / eps;
            }
    
            // Special handling for V0 (param == 4)
            const int param = 4;
            Kokkos::parallel_for(Kokkos::TeamThreadRange(team, m2+1), 
            [&](const int j) {
                for(int i = 0; i <= m1; i++) {
                    // Compute max(S-K, 0) for each point
                    double s_value = grid_i.device_Vec_s(i);
                    double payoff = Kokkos::max(s_value - K, 0.0);
                    U_i(i + j*(m1+1)) = payoff;
                    U_0_i(i + j*(m1+1)) = payoff;
                }
            });
            team.team_barrier();
            
            // Reset lambda (needed for American option)
            for(int idx = 0; idx < total_size; idx++) {
                lambda_bar_i(idx) = 0.0;
            }
    
            // Rebuild variance views with perturbed V0
            grid_i.rebuild_variance_views(V_0 + eps, 5.0, 5.0/500, team);
            const int index_v_pertubed = grid_i.find_v0_index(V_0 + eps);
            
            // Rebuild matrices with updated grid
            A0_solvers(instance).build_matrix(grid_i, rho, sigma, team);
            A1_solvers(instance).build_matrix(grid_i, r_d, r_f, theta, delta_t, team);
            A2_solvers(instance).build_matrix(grid_i, r_d, kappa, eta, sigma, theta, delta_t, team);
    
            // Compute perturbed solution
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
    
            // Get price and compute gradient
            double pert_price = U_i(index_s + index_v_pertubed*(m1+1));
            J(instance, param) = (pert_price - base_prices(instance)) / eps;
        });
    Kokkos::fence();
}
