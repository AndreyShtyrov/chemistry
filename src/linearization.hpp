#pragma once

#include "helper.h"

template<int N>
matrix<N, N> isqrt(matrix<N, N> m)
{
    for (int i = 0; i < N; i++)
        m(i, i) = 1. / sqrt(abs(m(i, i)));
    return m;
};

template<int N>
matrix<N, N> linearization(matrix<N, N> m)
{
    Eigen::JacobiSVD<matrix<N, N>> d(m, Eigen::ComputeFullU | Eigen::ComputeFullV);

    return d.matrixU() * isqrt(matrix<N, N>(d.singularValues().asDiagonal()));
}