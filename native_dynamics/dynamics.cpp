#ifndef NDYN_EXPORTS
#define NDYN_EXPORTS 1
#endif
#include "dynamics.h"
#include <cmath>
#include <algorithm>
#include <vector>

static inline void mat3_mul_vec3_rowmajor(const double* M9, const double* v3, double* out3) {
    // M9 = [m00 m01 m02 m10 m11 m12 m20 m21 m22] (row-major)
    out3[0] = M9[0]*v3[0] + M9[1]*v3[1] + M9[2]*v3[2];
    out3[1] = M9[3]*v3[0] + M9[4]*v3[1] + M9[5]*v3[2];
    out3[2] = M9[6]*v3[0] + M9[7]*v3[1] + M9[8]*v3[2];
}

static inline void cross3(const double* a, const double* b, double* out) {
    out[0] = a[1]*b[2] - a[2]*b[1];
    out[1] = a[2]*b[0] - a[0]*b[2];
    out[2] = a[0]*b[1] - a[1]*b[0];
}

static inline void sub3(const double* a, const double* b, double* out) {
    out[0] = a[0] - b[0];
    out[1] = a[1] - b[1];
    out[2] = a[2] - b[2];
}

static inline void add3_inplace(double* a, const double* b) {
    a[0] += b[0]; a[1] += b[1]; a[2] += b[2];
}

static inline void copy3(const double* src, double* dst) {
    dst[0] = src[0]; dst[1] = src[1]; dst[2] = src[2];
}

static inline void scale3_inplace(double* v, double s) {
    v[0]*=s; v[1]*=s; v[2]*=s;
}

extern "C" {

NDYN_API int dyn_compute_mf_batch(
    int N,
    const double* I_batch,
    const double* m_batch,
    const double* omega,
    const double* dot_omega,
    const double* ddpg,
    const double* g,
    double* M_out,
    double* F_out)
{
    if (N < 0 || !I_batch || !m_batch || !omega || !dot_omega || !ddpg || !g || !M_out || !F_out) return -1;
    for (int i=0; i<N; ++i) {
        const double* Ii = I_batch + i*9;
        const double* wi = omega + i*3;
        const double* dwi = dot_omega + i*3;
        const double* ai = ddpg + i*3;
        double* Mi = M_out + i*3;
        double* Fi = F_out + i*3;

        double Iw[3];
        mat3_mul_vec3_rowmajor(Ii, wi, Iw);

        double tmp[3];
        mat3_mul_vec3_rowmajor(Ii, dwi, tmp);
        copy3(tmp, Mi); // Mi = I * dot_omega

        double w_cross_Iw[3];
        cross3(wi, Iw, w_cross_Iw);
        add3_inplace(Mi, w_cross_Iw); // Mi += omega x (I * omega)

        Fi[0] = ai[0] - g[0];
        Fi[1] = ai[1] - g[1];
        Fi[2] = ai[2] - g[2];
        scale3_inplace(Fi, m_batch[i]);
    }
    return 0;
}

NDYN_API int dyn_compute_tau_chain(
    int N,
    const double* Ms,
    const double* Fs,
    const double* r_gs,
    const double* p1s,
    const double* tau_E,
    const double* f_E,
    const double* r_x,
    double* tau_out)
{
    if (N < 0 || !Ms || !Fs || !r_gs || !p1s || !tau_E || !f_E || !r_x || !tau_out) return -1;

    // Precompute cumulative sums from end to start to speed up Σ_{i>=j} M_i
    // However, the term sum_{i>=j} ((r_gs[i] - p1s[j]) x F_i) depends on j through p1s[j],
    // so we must loop j and i for that part.

    // Compute cumulative M starting from N-1 downwards
    // cumM[j] = sum_{i>=j} M_i
    if (N == 0) return 0;
    // allocate temporary cumM vector of size N*3
    std::vector<double> cumM(static_cast<size_t>(N)*3, 0.0);
    // start from last
    copy3(Ms + (static_cast<size_t>(N)-1)*3, &cumM[(static_cast<size_t>(N)-1)*3]);
    for (int j = N-2; j >= 0; --j) {
        const double* Mj = Ms + j*3;
        double* cmj = &cumM[j*3];
        copy3(&cumM[(j+1)*3], cmj);
        cmj[0] += Mj[0]; cmj[1] += Mj[1]; cmj[2] += Mj[2];
    }

    // Compute tau for each j
    for (int j=0; j<N; ++j) {
        const double* p1j = p1s + j*3;
        double tauj[3] = { cumM[j*3+0], cumM[j*3+1], cumM[j*3+2] };

        // sum_{i>=j} ((r_gs[i] - p1s[j]) x F_i)
        for (int i=j; i<N; ++i) {
            const double* rgi = r_gs + i*3;
            const double* Fi = Fs + i*3;
            double rdiff[3];
            sub3(rgi, p1j, rdiff);
            double rcrossF[3];
            cross3(rdiff, Fi, rcrossF);
            tauj[0] += rcrossF[0];
            tauj[1] += rcrossF[1];
            tauj[2] += rcrossF[2];
        }

        // - tau_E - ((r_x - p1s[j]) x f_E)
        tauj[0] -= tau_E[0];
        tauj[1] -= tau_E[1];
        tauj[2] -= tau_E[2];

        double rdiff2[3];
        sub3(r_x, p1j, rdiff2);
        double rcrossfE[3];
        cross3(rdiff2, f_E, rcrossfE);
        tauj[0] -= rcrossfE[0];
        tauj[1] -= rcrossfE[1];
        tauj[2] -= rcrossfE[2];

        // store
        double* outj = tau_out + j*3;
        outj[0] = tauj[0]; outj[1] = tauj[1]; outj[2] = tauj[2];
    }

    return 0;
}

} // extern "C"
