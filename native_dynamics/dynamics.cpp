#ifndef NDYN_EXPORTS
#define NDYN_EXPORTS 1
#endif
#include "dynamics.h"
#include <cmath>
#include <algorithm>
#include <vector>
#include <limits>

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

static inline bool solve3x3(double A[3][3], const double b[3], double x[3]) {
    double M[3][4] = {
        {A[0][0], A[0][1], A[0][2], b[0]},
        {A[1][0], A[1][1], A[1][2], b[1]},
        {A[2][0], A[2][1], A[2][2], b[2]},
    };

    for (int col = 0; col < 3; ++col) {
        int piv = col;
        double best = std::fabs(M[col][col]);
        for (int r = col + 1; r < 3; ++r) {
            const double v = std::fabs(M[r][col]);
            if (v > best) {
                best = v;
                piv = r;
            }
        }
        if (best < 1e-12) return false;
        if (piv != col) {
            for (int c = col; c < 4; ++c) {
                std::swap(M[col][c], M[piv][c]);
            }
        }

        const double d = M[col][col];
        for (int c = col; c < 4; ++c) M[col][c] /= d;

        for (int r = 0; r < 3; ++r) {
            if (r == col) continue;
            const double f = M[r][col];
            if (std::fabs(f) < 1e-18) continue;
            for (int c = col; c < 4; ++c) {
                M[r][c] -= f * M[col][c];
            }
        }
    }

    x[0] = M[0][3];
    x[1] = M[1][3];
    x[2] = M[2][3];
    return std::isfinite(x[0]) && std::isfinite(x[1]) && std::isfinite(x[2]);
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

NDYN_API int dyn_lpf_exp_fb(
    int N,
    const double* x,
    double dt,
    double fc,
    int passes,
    double* y_out)
{
    if (N < 1 || !x || !y_out) return -1;
    if (!(std::isfinite(dt)) || dt <= 0.0) return -2;
    if (!(std::isfinite(fc)) || fc <= 0.0) return -3;
    if (passes < 1) passes = 1;

    const double pi = 3.14159265358979323846;
    const double rc = 1.0 / (2.0 * pi * fc);
    const double alpha = dt / (rc + dt);
    if (!(std::isfinite(alpha)) || alpha <= 0.0) {
        // practically no filtering -> copy
        for (int i = 0; i < N; ++i) y_out[i] = x[i];
        return 0;
    }

    std::vector<double> a(static_cast<size_t>(N));
    std::vector<double> b(static_cast<size_t>(N));
    for (int i = 0; i < N; ++i) a[static_cast<size_t>(i)] = x[i];

    for (int p = 0; p < passes; ++p) {
        // forward
        b[0] = a[0];
        for (int i = 1; i < N; ++i) {
            b[static_cast<size_t>(i)] = b[static_cast<size_t>(i - 1)] + alpha * (a[static_cast<size_t>(i)] - b[static_cast<size_t>(i - 1)]);
        }
        // backward
        a[static_cast<size_t>(N - 1)] = b[static_cast<size_t>(N - 1)];
        for (int i = N - 2; i >= 0; --i) {
            a[static_cast<size_t>(i)] = a[static_cast<size_t>(i + 1)] + alpha * (b[static_cast<size_t>(i)] - a[static_cast<size_t>(i + 1)]);
        }
    }

    for (int i = 0; i < N; ++i) y_out[i] = a[static_cast<size_t>(i)];
    return 0;
}

NDYN_API int dyn_triangulate_transform_batch(
    int N,
    const double* P0,
    const double* P1,
    const double* pts0,
    const double* pts1,
    double scale,
    double* out_xyz)
{
    if (N < 0 || !P0 || !P1 || !pts0 || !pts1 || !out_xyz) return -1;
    if (!(std::isfinite(scale)) || scale == 0.0) return -2;

    const double qnan = std::numeric_limits<double>::quiet_NaN();
    for (int i = 0; i < N; ++i) {
        out_xyz[i*3 + 0] = qnan;
        out_xyz[i*3 + 1] = qnan;
        out_xyz[i*3 + 2] = qnan;
    }

    for (int i = 0; i < N; ++i) {
        const double u0 = pts0[i*2 + 0];
        const double v0 = pts0[i*2 + 1];
        const double u1 = pts1[i*2 + 0];
        const double v1 = pts1[i*2 + 1];

        if (!(std::isfinite(u0) && std::isfinite(v0) && std::isfinite(u1) && std::isfinite(v1))) continue;
        if (u0 < 0.0 || v0 < 0.0 || u1 < 0.0 || v1 < 0.0) continue;

        double rows[4][4];
        for (int k = 0; k < 4; ++k) {
            rows[0][k] = u0 * P0[8 + k] - P0[0 + k];
            rows[1][k] = v0 * P0[8 + k] - P0[4 + k];
            rows[2][k] = u1 * P1[8 + k] - P1[0 + k];
            rows[3][k] = v1 * P1[8 + k] - P1[4 + k];
        }

        double AtA[3][3] = {{0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}};
        double Atb[3] = {0.0, 0.0, 0.0};

        for (int r = 0; r < 4; ++r) {
            const double a0 = rows[r][0];
            const double a1 = rows[r][1];
            const double a2 = rows[r][2];
            const double d  = rows[r][3];

            AtA[0][0] += a0 * a0; AtA[0][1] += a0 * a1; AtA[0][2] += a0 * a2;
            AtA[1][0] += a1 * a0; AtA[1][1] += a1 * a1; AtA[1][2] += a1 * a2;
            AtA[2][0] += a2 * a0; AtA[2][1] += a2 * a1; AtA[2][2] += a2 * a2;

            Atb[0] += -a0 * d;
            Atb[1] += -a1 * d;
            Atb[2] += -a2 * d;
        }

        double X[3];
        if (!solve3x3(AtA, Atb, X)) continue;

        const double tx = -X[0] * scale;
        const double ty = -X[2] * scale;
        const double tz = -X[1] * scale;
        if (!(std::isfinite(tx) && std::isfinite(ty) && std::isfinite(tz))) continue;

        out_xyz[i*3 + 0] = tx;
        out_xyz[i*3 + 1] = ty;
        out_xyz[i*3 + 2] = tz;
    }
    return 0;
}

} // extern "C"
