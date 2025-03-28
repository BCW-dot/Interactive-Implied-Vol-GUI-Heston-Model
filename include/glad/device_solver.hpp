#pragma once
#include <Kokkos_Core.hpp>

#include "hes_a0_kernels.hpp"
#include "hes_a1_kernels.hpp"
#include "hes_a2_shuffled_kernels.hpp"
#include "hes_boundary_kernels.hpp"

#include "DO_solver_workspace.hpp"

/*

This is a device callable time stepping of the heston pde

*/
template<class Device, class ViewType>  // Add ViewType template parameter
KOKKOS_FUNCTION 
void device_DO_timestepping(
    // Grid dimensions
    const int m1,
    const int m2,
    // Time discretization
    const int N,
    const double delta_t,
    const double theta,
    const double r_f,
    // Problem components
    Device_A0_heston<Device>& A0,
    Device_A1_heston<Device>& A1,
    Device_A2_shuffled_heston<Device>& A2,
    const Device_BoundaryConditions<Device>& bounds,
    // Workspace views for this instance - now using ViewType
    ViewType& U_i,
    ViewType& Y_0_i,
    ViewType& Y_1_i,
    ViewType& A0_result_i,
    ViewType& A1_result_i,
    ViewType& A2_result_unshuf_i,
    ViewType& U_shuffled_i,
    ViewType& Y_1_shuffled_i,
    ViewType& A2_result_shuffled_i,
    ViewType& U_next_shuffled_i,
    // Team handle
    const typename Kokkos::TeamPolicy<>::member_type& team
) {
    const int total_size = (m1+1)*(m2+1);

    for(int n = 1; n <= N; n++) {
        // Step 1: Y0 computation
        A0.multiply_parallel_v(U_i, A0_result_i, team);
        A1.multiply_parallel_v(U_i, A1_result_i, team);
        
        device_shuffle_vector(U_i, U_shuffled_i, m1, m2, team);
        A2.multiply_parallel_s(U_shuffled_i, A2_result_shuffled_i, team);
        device_unshuffle_vector(A2_result_shuffled_i, A2_result_unshuf_i, m1, m2, team);

        // Y0 computation with boundary terms
        Kokkos::parallel_for(Kokkos::TeamThreadRange(team, total_size), 
            [&](const int i) {
                double exp_factor = std::exp(r_f * delta_t * (n-1));
                Y_0_i(i) = U_i(i) + delta_t * (A0_result_i(i) + A1_result_i(i) + 
                          A2_result_unshuf_i(i) + bounds.b_(i) * exp_factor);
            });

        // Step 2: A1 implicit solve
        Kokkos::parallel_for(Kokkos::TeamThreadRange(team, total_size),
            [&](const int i) {
                double exp_factor_n = std::exp(r_f * delta_t * n);
                double exp_factor_nm1 = std::exp(r_f * delta_t * (n-1));
                Y_0_i(i) = Y_0_i(i) + theta * delta_t * (bounds.b1_(i) * exp_factor_n - 
                          (A1_result_i(i) + bounds.b1_(i) * exp_factor_nm1));
            });
        A1.solve_implicit_parallel_v(Y_1_i, Y_0_i, team);

        // Step 3: A2 shuffled implicit solve
        Kokkos::parallel_for(Kokkos::TeamThreadRange(team, total_size),
            [&](const int i) {
                double exp_factor_n = std::exp(r_f * delta_t * n);
                double exp_factor_nm1 = std::exp(r_f * delta_t * (n-1));
                Y_1_i(i) = Y_1_i(i) + theta * delta_t * (bounds.b2_(i) * exp_factor_n - 
                          (A2_result_unshuf_i(i) + bounds.b2_(i) * exp_factor_nm1));
            });

        device_shuffle_vector(Y_1_i, Y_1_shuffled_i, m1, m2, team);
        A2.solve_implicit_parallel_s(U_next_shuffled_i, Y_1_shuffled_i, team);
        device_unshuffle_vector(U_next_shuffled_i, U_i, m1, m2, team);
    }
}

