#pragma once

#include <cstddef>

#if defined(_WIN32)
#  ifdef NDYN_EXPORTS
#    define NDYN_API __declspec(dllexport)
#  else
#    define NDYN_API __declspec(dllimport)
#  endif
#else
#  define NDYN_API __attribute__((visibility("default")))
#endif

extern "C" {

// Compute M and F for a batch of parts.
// Inputs:
//  N: number of items
//  I_batch: pointer to N*9 doubles (row-major 3x3 for each item)
//  m_batch: pointer to N doubles
//  omega, dot_omega, ddpg: pointers to N*3 doubles
//  g: pointer to 3 doubles
// Outputs:
//  M_out, F_out: pointers to N*3 doubles (pre-allocated by caller)
// Returns 0 on success, non-zero on invalid args
NDYN_API int dyn_compute_mf_batch(
    int N,
    const double* I_batch,
    const double* m_batch,
    const double* omega,
    const double* dot_omega,
    const double* ddpg,
    const double* g,
    double* M_out,
    double* F_out);

// Compute joint torques for a kinematic chain using accumulated M and F.
// tau_j = sum_{i>=j} M_i + sum_{i>=j} ((r_gs[i]-p1s[j]) x F_i) - tau_E - ((r_x - p1s[j]) x f_E)
// Inputs:
//  N: number of parts
//  Ms, Fs, r_gs, p1s: pointers to N*3 doubles
//  tau_E, f_E, r_x: pointers to 3 doubles
// Output:
//  tau_out: N*3 doubles
// Returns 0 on success
NDYN_API int dyn_compute_tau_chain(
    int N,
    const double* Ms,
    const double* Fs,
    const double* r_gs,
    const double* p1s,
    const double* tau_E,
    const double* f_E,
    const double* r_x,
    double* tau_out);

}
