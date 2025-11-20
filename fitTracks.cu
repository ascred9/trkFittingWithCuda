#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

#include "cuda.h"

#define M_PI 3.14159265358979323846
#define THREADS_PER_BLOCK 1024

#define ADD_CONSTRAINT false
#define BEAM_ORBIT 333 // mm

#define ADD_ALIGNMENT false 

#define ADD_V_CONSTR false

#define FIX_VERT true 

cudaError_t errCode;

typedef struct{
    float elements[5]; // [d/dx0, d/dy0, d/dz0, d/dR, d/dV] column
} Jacobian;

typedef struct{
    // [d2/(dx0 dx0) d2/(dx0 dy0) d2/(dx0 dz0) d2/(dx0 dR) d2/(dx0 dV)]
    // [d2/(dy0 dx0) d2/(dy0 dy0) ...]
    float elements[5][5]; 
} Hessian;

__device__ float getT(float xi, float yi, float zi, float x0, float y0, float z0, float R, float V, bool& intersection, bool& plus);

__device__ void correctPeriod(float zi, float z0, float V, float& t);

__device__ bool calculateCircle(float x1, float y1, float x2, float y2, float x3, float y3, float& x0, float& y0, float& R)
{
    constexpr float epsilon = 1e-5;

    float dist12 = (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2);
    float dist13 = (x1 - x3) * (x1 - x3) + (y1 - y3) * (y1 - y3);
    float dist23 = (x2 - x3) * (x2 - x3) + (y2 - y3) * (y2 - y3);

    if (dist12 < epsilon || dist13 < epsilon || dist23 < epsilon)
        return false;

    float A = x2 - x1;
    float B = y2 - y1;
    float C = x3 - x1;
    float D = y3 - y1;
    float E = A * (x1 + x2) + B * (y1 + y2);
    float F = C * (x1 + x3) + D * (y1 + y3);
    float G = 2 * (A * (y3 - y2) - B * (x3 - x2));
    if (abs(G) < epsilon)
        return false;

    x0 = (D * E - B * F) / G;
    y0 = (A * F - C * E) / G;
    R = sqrtf(powf(x1  - x0, 2) + powf(y1 - y0, 2));
    return R > 1 && R < 400;
}

// get preliminary parameters of helix
// get center of helix, radii and z-velocity
__device__ bool calculatePrelimPars(int tId, const float* data, float* output)
{
    constexpr float epsilon = 1e-2;
    constexpr int nPars = 5;
    constexpr int nVars = 12;
    constexpr int nPoints = 4;

    int count = 0;
    float x0 = 0, y0 = 0, R = 0;
    float tmpX0, tmpY0, tmpR;
    for (int i = 0; i < nPoints - 2; i++)
    {
        float xi = data[nVars * tId + 3 * i], yi = data[nVars * tId + 3 * i + 1];
        for (int j = i + 1; j < nPoints - 1; j++)
        {
            float xj = data[nVars * tId + 3 * j], yj = data[nVars * tId + 3 * j + 1];
            for (int k = j + 1; k < nPoints; k++)
            {
                float xk = data[nVars * tId + 3 * k], yk = data[nVars * tId + 3 * k + 1];
                if (calculateCircle(xi, yi, xj, yj, xk, yk, tmpX0, tmpY0, tmpR))
                {
                    x0 += tmpX0;
                    y0 += tmpY0;
                    R += tmpR;
                    count++;
                }
            }
        }
    }

    if (count == 0)
        return false;

    x0 /= count;
    y0 /= count;
    R /= count;

    count = 0;
    float V = 0;
    bool intersection, plus;
    for (int i = 0; i < nPoints - 1; i++)
    {
        float xi = data[nVars * tId + 3 * i], yi = data[nVars * tId + 3 * i + 1], zi = data[nVars * tId + 3 * i + 2];
        for (int j = i + 1; j < nPoints; j++)
        {
            float xj = data[nVars * tId + 3 * j], yj = data[nVars * tId + 3 * j + 1], zj = data[nVars * tId + 3 * j + 2];

            float ti = getT(xi, yi, zi, x0, y0, 0, R, 0, intersection, plus);
            float tj = getT(xj, yj, zj, x0, y0, 0, R, 0, intersection, plus);

            float deltaT = tj - ti;
            if (abs(deltaT) < epsilon)
                continue;
            
            if (deltaT < 0)
                deltaT += 2 * M_PI;

            V += (zj - zi) / deltaT;

            count++;
        }
    }

    if (count == 0)
        return false;

    V /= count;

    count = 0;
    float z0 = 0;
    for (int i = 0; i < nPoints; i++)
    {
        float xi = data[nVars * tId + 3 * i], yi = data[nVars * tId + 3 * i + 1], zi = data[nVars * tId + 3 * i + 2];
        //float phi = -atan2f(yi - y0, xi - x0); // [-pi, +pi], but reverse parametrization
        float t = getT(xi, yi, zi, x0, y0, 0, R, 0, intersection, plus);
        z0 += zi - V * t;
        count++;
    }

    if (count == 0)
        return false;

    z0 /= count;

    output[nPars * tId] = x0;
    output[nPars * tId + 1] = y0;
    output[nPars * tId + 2] = z0;
    output[nPars * tId + 3] = R;
    output[nPars * tId + 4] = V;
    return true;
} 

// return parameter of helix t in [-pi, +pi] range,
// TODO:: pull up z(ti) by period
__device__ float getT(float xi, float yi, float zi, float x0, float y0, float z0, float R, float V, bool& intersection, bool& plus)
{
    float vanePhi = atan2f(yi, xi); // [-pi, +pi]
    float tau = x0 * cosf(vanePhi) + y0 * sinf(vanePhi);
    float d2 = powf(x0 - tau * cosf(vanePhi), 2) + powf(y0 - tau * sinf(vanePhi), 2);
    float t = 0;
    if (d2 >= R*R)
    {
        // no intersection -> closest point between vane and circle
        intersection = false;
        plus = false;
        t = -atan2f(tau * sinf(vanePhi) - y0, tau * cosf(vanePhi) - x0); // dt/x0 = 0 and dt/dy = 0
    }
    else
    {
        // intersection
        intersection = true;
        float t1 = asin( (y0 * cosf(vanePhi) - x0 * sinf(vanePhi)) / R ) - vanePhi;
        float t2 = M_PI - asin( (y0 * cosf(vanePhi) - x0 * sinf(vanePhi)) / R ) - vanePhi;
        if ((x0 - xi) * cosf(-t1) + (y0 - yi) * sin(-t1) < (x0 - xi) * cosf(-t2) + (y0 - yi) * sin(-t2))
        {
            t = t1;
            plus = true;
        }
        else
        {
            t = t2;
            plus = false;
        }

        if (t < -M_PI)
            t += 2 * M_PI;
        else if (t > M_PI)
            t -= 2 * M_PI;
    }

    return t;
}

__device__ void correctPeriod(float zi, float z0, float V, float& t)
{
    int period = 0;
    float delta = abs(zi - z0);
    for (int n = -1; n < 1; n++)
    {
        float tmp = abs(zi - z0 - V * (t + 2 * M_PI * n));
        if (tmp < delta)
        {
            delta = tmp;
            period = n;
        }
    }

    t += 2 * M_PI * period;
}

__device__ float getChiSquare(int tId, const float* data, const float* output, const float constr_resolution)
{
    // sum_i( (xi - x(ti))^2 + (yi - y(ti))^2 + (zi - z(ti))^2)
    constexpr int nPars = 5;
    constexpr int nPoints = 4;
    constexpr int nVars = nPoints * 3;
    constexpr float resolution = 3.4641; // mm

    float x0 = output[nPars * tId];
    float y0 = output[nPars * tId + 1];
    float z0 = output[nPars * tId + 2];
    float R = output[nPars * tId + 3];
    float V = output[nPars * tId + 4];

    float chi = 0;
    float t, dx2, dy2, dz2;
    bool intersection, plus;
    for (int i = 0; i < nPoints; i++)
    {
        float xi = data[nVars * tId + i * 3];
        float yi = data[nVars * tId + i * 3 + 1];
        float zi = data[nVars * tId + i * 3 + 2];

        t = getT(xi, yi, zi, x0, y0, z0, R, V, intersection, plus);
        correctPeriod(zi, z0, V, t);

        dx2 = (x0 + R * cosf(-t) - xi) * (x0 + R * cosf(-t) - xi);
        dy2 = (y0 + R * sinf(-t) - yi) * (y0 + R * sinf(-t) - yi);
        dz2 = (z0 + V * t - zi) * (z0 + V * t - zi);

        chi += dx2 + dy2 + dz2;
    }

    int ndf = 3 * nPoints - nPars - 1;
    chi /= (ndf * resolution * resolution);

    if (ADD_CONSTRAINT)
    {
        float delta = (BEAM_ORBIT * BEAM_ORBIT - (x0 * x0 + y0 * y0 + R * R)) / (constr_resolution * constr_resolution);
        if (delta > 0)
            chi += delta;
    }

    if (ADD_ALIGNMENT)
    {
        float chiPart2 = 0;
        for (int i = 0; i < nPoints; i++)
        {
            float xi = data[nVars * tId + i * 3];
            float yi = data[nVars * tId + i * 3 + 1];

            chiPart2 += ( R - sqrtf((xi - x0)*(xi - x0) + (yi - y0)*(yi - y0)) );
        }
        chiPart2 *= chiPart2 / (ndf * resolution * resolution);
        chi += chiPart2;
    }

    if (ADD_V_CONSTR)
    {    
        chi -= V * V / (constr_resolution * constr_resolution);
    }

    return chi;
}

__device__ bool calcJacobian(int tId, const float* data, const float* output, const float constr_resolution, Jacobian& jac)
{
    constexpr int nPars = 5;
    constexpr int nPoints = 4;
    constexpr int nVars = nPoints * 3;
    constexpr float resolution = 3.4641; // mm
    
    float x0 = output[nPars * tId];
    float y0 = output[nPars * tId + 1];
    float z0 = output[nPars * tId + 2];
    float R = output[nPars * tId + 3];
    float V = output[nPars * tId + 4];

    for (int i = 0; i < nPars; i++)
        jac.elements[i] = 0;

    for (int i = 0; i < nPoints; i++)
    {
        float xi = data[nVars * tId + i * 3];
        float yi = data[nVars * tId + i * 3 + 1];
        float zi = data[nVars * tId + i * 3 + 2];

        bool intersection, plus;
        float t = getT(xi, yi, zi, x0, y0, z0, R, V, intersection, plus);
        correctPeriod(zi, z0, V, t);

        float x = x0 + R * cos(-t);
        float y = y0 + R * sin(-t);
        float z = z0 + V * t;

        jac.elements[0] += 2 * (x - xi); // dchi/dx0
        jac.elements[1] += 2 * (y - yi); // dchi/dy0
        jac.elements[2] += 2 * (z - zi); // dchi/dz0
        jac.elements[3] += 2 * (x - xi) * cos(-t) + 2 * (y - yi) * sin(-t); // dchi/dR
        jac.elements[4] += 2 * (z - zi) * t; // dchi/dV

        if (!intersection)
            continue;

        float dchidt = 2 * (x - xi) * ( R * sin(-t)) +
                        2 * (y - yi) * ( - R * cos(-t)) +
                        2 * (z - zi) * V;

        float vanePhi = atan2f(yi, xi); // [-pi, +pi]

        float dtds = (plus ? 1. : -1.) * 1 / sqrtf(1 - (y0 * cosf(vanePhi) - x0 * sinf(vanePhi)) * (y0 * cosf(vanePhi) - x0 * sinf(vanePhi)) / (R*R));

        jac.elements[0] += dchidt * dtds * (-sinf(vanePhi) / R);
        jac.elements[1] += dchidt * dtds * ( cosf(vanePhi) / R);
        jac.elements[3] += dchidt * dtds * (-1 * (y0 * cosf(vanePhi) - x0 * sinf(vanePhi))) / (R * R);
    }

    int ndf = 3 * nPoints - nPars - 1;
    for (int i = 0; i < nPars; i++)
        jac.elements[i] /= (ndf * resolution * resolution);

    if (ADD_CONSTRAINT)
    {
        float delta = (BEAM_ORBIT * BEAM_ORBIT - (x0 * x0 + y0 * y0 + R * R)) / (constr_resolution * constr_resolution);
        if (delta > 0)
        {
            jac.elements[0] += - 2 * x0 / (constr_resolution * constr_resolution);
            jac.elements[1] += - 2 * y0 / (constr_resolution * constr_resolution);
            jac.elements[3] += - 2 * R /  (constr_resolution * constr_resolution);
        }
    }
    
    if (ADD_ALIGNMENT)
    {
        float chiPart2 = 0;
        float x0Part = 0, y0Part = 0;
        for (int i = 0; i < nPoints; i++)
        {
            float xi = data[nVars * tId + i * 3];
            float yi = data[nVars * tId + i * 3 + 1];

            chiPart2 += ( R - sqrtf((xi - x0)*(xi - x0) + (yi - y0)*(yi - y0)) );
            x0Part += (x0 - xi) / sqrtf((xi - x0)*(xi - x0) + (yi - y0)*(yi - y0));
            y0Part += (y0 - yi) / sqrtf((xi - x0)*(xi - x0) + (yi - y0)*(yi - y0));
        }
        chiPart2 *= 2. / (ndf * resolution * resolution * 0.1);
        jac.elements[0] += chiPart2 * x0Part;
        jac.elements[1] += chiPart2 * y0Part;
        jac.elements[3] -= chiPart2 * nPoints;
    }

    if (ADD_V_CONSTR)
    {
        float minT = 0, maxT = 0;
        for (int i = 0; i < nPoints; i++)
        {
            float xi = data[nVars * tId + i * 3];
            float yi = data[nVars * tId + i * 3 + 1];
            float zi = data[nVars * tId + i * 3 + 2];

            bool intersection, plus;
            float t = getT(xi, yi, zi, x0, y0, z0, R, V, intersection, plus);
            correctPeriod(zi, z0, V, t);
            if (minT > t || i == 0)
                minT = t;

            if (maxT < t || i == 0)
                maxT = t;
        }

        int nPeriods = (maxT - minT) / (2 * M_PI);
        jac.elements[4] += (V  * V * nPeriods * nPeriods) / (constr_resolution * constr_resolution);
    }


    return true;
}

__device__ bool calcHessian(int tId, const float* data, const float* output, Hessian& hes)
{
    constexpr int nPars = 5;
    constexpr int nPoints = 4;
    constexpr float resolution = 3.4641; // mm
    
    float x0 = output[nPars * tId];
    float y0 = output[nPars * tId + 1];
    float z0 = output[nPars * tId + 2];
    float R = output[nPars * tId + 3];
    float V = output[nPars * tId + 4];

    for (int i = 0; i < nPars; i++)
        for (int j = 0; j < nPars; j++)
            hes.elements[i][j] = 0;

    for (int i = 0; i < nPoints; i++)
    {
        float xi = data[12 * tId + i * 3];
        float yi = data[12 * tId + i * 3 + 1];
        float zi = data[12 * tId + i * 3 + 2];

        bool intersection, plus;
        float t = getT(xi, yi, zi, x0, y0, z0, R, V, intersection, plus);

        float x = x0 + R * cos(-t);
        float y = y0 + R * sin(-t);
        float z = z0 + V * t;
    
        hes.elements[0][0] += 2; // d2/(dx0 dx0)
        hes.elements[0][3] += 2 * cosf(-t); // d2/(dx0 dR)
        hes.elements[1][1] += 2; // d2/(dy0 dy0)
        hes.elements[1][3] += 2 * sinf(-t); // d2/(dy0 dR)
        hes.elements[2][2] += 2; // d2/(dz0 dz0)
        hes.elements[2][4] += 2 * t; // d2/(dz0 dV)
        hes.elements[3][0] += 2 * cosf(-t); // d2/(dR dx0)
        hes.elements[3][1] += 2 * sinf(-t); // d2/(dR dy0)
        hes.elements[3][3] += 2; // d2/(dR dR)
        hes.elements[4][2] += 2 * t; // d2/(dV dz0)
        hes.elements[4][4] += 2; // d2/(dV dV)

        if (!intersection)
            continue;

        float dchidt = 2 * (x - xi) * ( R * sin(-t)) +
                        2 * (y - yi) * ( - R * cos(-t)) +
                        2 * (z - zi) * V;

        float d2chidt2 = 2 * ( R * R ) + 2 * (x - xi) * ( - R * cos(-t))
                                        + 2 * (y - yi) * ( - R * sin(-t)) +
                          2 * V * V;

        float vanePhi = atan2f(yi, xi); // [-pi, +pi]

        float s = (y0 * cosf(vanePhi) - x0 * sinf(vanePhi)) / R;
        float dtds = (plus ? 1. : -1.) * 1 / sqrtf(1 - s * s);
        float d2tds2 = (plus ? 1. : -1.) * s / powf(sqrtf(1 - s * s), 1.5);

        hes.elements[0][0] += 2 * dtds * (-sinf(vanePhi) / R) + (d2chidt2 * dtds * dtds + dchidt * d2tds2) * (sinf(vanePhi) * sinf(vanePhi)) / (R * R);
        hes.elements[0][1] += 2 * dtds * (-sinf(vanePhi) / R) + (d2chidt2 * dtds * dtds + dchidt * d2tds2) * (-sinf(vanePhi) * cosf(vanePhi)) / (R * R);
        hes.elements[0][2] += 2 * dtds * (-sinf(vanePhi) / R);
        hes.elements[0][3] += 2 * ((x - xi) * sinf(-t) + (y - yi) * (- cosf(-t))) * dtds * (-sinf(vanePhi)) / (R) +
                              (d2chidt2 * dtds * dtds + dchidt * d2tds2) * (-s / R) * (-sinf(vanePhi)) / (R) + 
                              dchidt * dtds * (sinf(vanePhi) / (R * R));
        hes.elements[0][4] += 2 * t * dtds * (-sinf(vanePhi) / R);

        hes.elements[1][0] = hes.elements[0][1];
        hes.elements[1][1] += 2 * dtds * (cosf(vanePhi) / R) + (d2chidt2 * dtds * dtds + dchidt * d2tds2) * (cosf(vanePhi) * cosf(vanePhi)) / (R * R);
        hes.elements[1][2] += 2 * dtds * (cosf(vanePhi) / R);
        hes.elements[1][3] += 2 * ((x - xi) * sinf(-t) + (y - yi) * (- cosf(-t))) * dtds * (cosf(vanePhi)) / (R) +
                              (d2chidt2 * dtds * dtds + dchidt * d2tds2) * (-s / R) * (cosf(vanePhi)) / (R) + 
                              dchidt * dtds * (-cosf(vanePhi) / (R * R));
        hes.elements[1][4] += 2 * t * dtds * (cosf(vanePhi) / R);

        hes.elements[2][0] = hes.elements[0][2];
        hes.elements[2][1] = hes.elements[1][2];

        hes.elements[3][0] = hes.elements[0][3];
        hes.elements[4][0] = hes.elements[0][4];
    }

    int ndf = 3 * nPoints - nPars - 1;
    for (int i = 0; i < nPars; i++)
        for (int j = 0; j < nPars; j++)
            hes.elements[i][j] /= (ndf * resolution * resolution);

    return true;
}

__global__ void prepareTracks_kernel(const float* data, int size, float* output, float* info)
{
    // We will use the following parametrization:
    // C(t) = (x0 + R cos(-t), y0 + R sin(-t), z0 + V t)
    int tId = blockDim.x * blockIdx.x + threadIdx.x;
    if (tId >= size)
        return;

    const int ninfo = 3;
    if (!calculatePrelimPars(tId, data, output))
    {
        info[ninfo * tId] = -1;
        info[ninfo * tId + 1] = -1;
        return;
    }

    if (tId == 90) printf("%f\n", output[5 * 90 + 4]);
}
    
__global__ void fitTracks_kernel(const float* data, int size, float* output, float* info)
{
    constexpr int Niter = 1e5;
    __shared__ Jacobian jacobian[THREADS_PER_BLOCK];

    int tId = blockDim.x * blockIdx.x + threadIdx.x;
    if (tId >= size)
        return;

    // bad initilization
    const int ninfo = 3;
    if (info[ninfo * tId] == -1)
        return;

    float step = 1e-1;
    float constr_res = 1e2;

    const int nPars = 5;
    float chiInit = getChiSquare(tId, data, output, constr_res);
    float chiPrev = chiInit, chiCur = 0;

    for (int iter = 0; iter < Niter; iter++)
    {
        if (!calcJacobian(tId, data, output, constr_res, jacobian[threadIdx.x]))
            return;

        for (int i = 0; i < nPars; i++)
            output[nPars * tId + i] -= step * jacobian[threadIdx.x].elements[i];

        if (output[nPars * tId + 3] < 0) // R < 0 ?
            break;
        
        chiCur = getChiSquare(tId, data, output, constr_res);
        if (chiCur > chiPrev)
            step /= M_E;

        if (ADD_CONSTRAINT)
        {
            // increase the importance of constraint
            constr_res -= 1e3 / Niter;
            if (constr_res < 1) // reduce before 1 mm resolution
                constr_res = 1;
        }
    }
    
    if (FIX_VERT)
    {
        int nPoints = 4;
        int nVars = 12;
        float tMax, tMin, zMax, zMin;
        bool intersection, plus;
        for (int i = 0; i < nPoints; i++)
        {
            float x0 = output[nPars * tId + 0], y0 = output[nPars * tId + 1], z0 = output[nPars * tId + 2];
            float R = output[nPars * tId + 3], V = output[nPars * tId + 4];
            float xi = data[nVars * tId + 3 * i], yi = data[nVars * tId + 3 * i + 1], zi = data[nVars * tId + 3 * i + 2];
            float t = getT(xi, yi, zi, x0, y0, z0, R, V, intersection, plus);
            correctPeriod(zi, z0, V, t);
            if (i == 0 || tMin > t)
            {
                tMin = t;
                zMin = zi;
            }

            if (i == 0 || tMax < t)
            {
                tMax = t;
                zMax = zi;
            }
        }

        float dt = (tMax - tMin) - int( (tMax - tMin) / (2 * M_PI) ) * 2 * M_PI;
        output[nPars * tId + 4] = (zMax - zMin) / dt;
        output[nPars * tId + 2] = zMax - tMax * output[nPars * tId + 4];
        chiCur = getChiSquare(tId, data, output, constr_res);
    }

    info[ninfo * tId] = chiCur;
    info[ninfo * tId + 2] = chiInit;

    info[ninfo * tId + 1] = 0;
    for (int i = 0; i < nPars; i++)
        info[ninfo * tId + 1] += abs(jacobian[threadIdx.x].elements[i]);
}

bool error()
{
    if ( (errCode = cudaGetLastError()) != cudaError::cudaSuccess)
    {
        printf("Error: %s", cudaGetErrorString(errCode));
        return false;
    }
    return true;
}

// data: input with 4 points with 3 coordiantes of hits
// size: number of track candidates
// output: 5 parameters (x0, y0, z0, R, V): center of circle, some start z coord of helix, radii and velocity along z-axis
// info: status of the fitter, chiSqaure
void fitTracks(const float* data, const int size, float* output, float* info)
{
    const int nVars = 12;
    const int nPars = 5;
    const int nInfo = 3;
    // Perform as one-dimensioanl task

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    if (!error())
        return;

    const int maxThreadsPerBlock = prop.maxThreadsPerBlock;
    const int maxGridSize = prop.maxGridSize[0];
    if (size > maxGridSize)
    {
        printf("Number of tracks is much more than grid size.\n");
        return;
    }

    const int threadsPerBlock = THREADS_PER_BLOCK;//maxThreadsPerBlock;
    const int numBlocks = size / threadsPerBlock + 1;
    printf("Blocks: %d, threads per block: %d\n", numBlocks, threadsPerBlock);

    int dataSize = size * nVars * sizeof(float);
    int outSize = size * nPars * sizeof(float);
    int infoSize = size * nInfo * sizeof(float);

    float* d_data;
    cudaMalloc( (void**)&d_data, dataSize); // TODO: make it constant
    if ( !error())
        return;

    float* d_out;
    cudaMalloc( (void**)&d_out, outSize);
    if ( !error())
        return;

    float* d_info;
    cudaMalloc( (void**)&d_info, infoSize);
    if ( !error())
        return;

    cudaMemcpy(d_data, data, dataSize, cudaMemcpyHostToDevice);
    if ( !error())
        return;

    printf("Start calculation of preliminary parameters.\n");
    prepareTracks_kernel<<<numBlocks, maxThreadsPerBlock>>>(d_data, size, d_out, d_info);
    cudaDeviceSynchronize();
    if (!error())
        return;

    printf("Start fitting.\n");
    fitTracks_kernel<<<numBlocks, maxThreadsPerBlock, sizeof(Jacobian) * THREADS_PER_BLOCK>>>(d_data, size, d_out, d_info);
    cudaDeviceSynchronize();
    if (!error())
        return;

    cudaMemcpy(output, d_out, outSize, cudaMemcpyDeviceToHost);
    if ( !error())
        return;

    cudaMemcpy(info, d_info, infoSize, cudaMemcpyDeviceToHost);
    if ( !error())
        return;

    cudaFree(d_data);
    cudaFree(d_out);
    cudaFree(d_info);
    if (!error())
        return;
}

void printDeviceInfo(int i = 0)
{
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, i);
    if (!error())
        return;

    printf("\n");
    printf("Device name: %s\n", prop.name);
    printf("Global memory: %lu\n", prop.totalGlobalMem);
    printf("Regs per block: %d\n", prop.regsPerBlock);
    printf("Warp size: %d\n", prop.warpSize);
    printf("Max threads per block: %d\n", prop.maxThreadsPerBlock);
    printf("Max threadsDim0: %d\n", prop.maxThreadsDim[0]);
    printf("Max threadsDim1: %d\n", prop.maxThreadsDim[1]);
    printf("Max threadsDim2: %d\n", prop.maxThreadsDim[2]);
    printf("Max grid size0: %d\n", prop.maxGridSize[0]);
    printf("Max grid size1: %d\n", prop.maxGridSize[1]);
    printf("Max grid size2: %d\n", prop.maxGridSize[2]);
    printf("Multi processor count: %d\n", prop.multiProcessorCount);
    printf("\n");
}
