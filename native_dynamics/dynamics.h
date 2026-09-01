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

// Exponential low-pass filter with forward-backward pass (zero-phase like).
// Inputs:
//  N: number of samples
//  x: input array (N)
//  dt: sample period [s]
//  fc: cutoff frequency [Hz]
//  passes: number of forward-backward repetitions (>=1)
// Output:
//  y_out: filtered array (N)
// Returns 0 on success
NDYN_API int dyn_lpf_exp_fb(
    int N,
    const double* x,
    double dt,
    double fc,
    int passes,
    double* y_out);

// Batch triangulation + coordinate transform with finite checks.
// Inputs:
//  N: number of keypoints
//  P0, P1: projection matrices (3x4, row-major, 12 doubles each)
//  pts0, pts1: 2D points arrays (N*2): [u, v, u, v, ...]
//  scale: scale factor from triangulated space to output space (e.g., 0.01)
// Output:
//  out_xyz: transformed 3D points (N*3). Invalid points are written as NaN.
// Returns 0 on success.
NDYN_API int dyn_triangulate_transform_batch(
    int N,
    const double* P0,
    const double* P1,
    const double* pts0,
    const double* pts1,
    double scale,
    double* out_xyz);

}
