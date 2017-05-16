#pragma once

#include "helper.h"

matrix isqrt(matrix m)
{
    for (int i = 0; i < m.rows(); i++)
        m(i, i) = 1. / sqrt(abs(m(i, i)));
    return m;
};

matrix linearization(matrix m)
{
    Eigen::JacobiSVD<matrix> d(m, Eigen::ComputeFullU | Eigen::ComputeFullV);
    LOG_INFO("singular values: {}", d.singularValues().transpose());

    return d.matrixU();
}

matrix linearizationNormalization(matrix m)
{
    Eigen::JacobiSVD<matrix> d(m, Eigen::ComputeFullU | Eigen::ComputeFullV);
    cout << "\t\t\tsingular values: " << d.singularValues().transpose() << endl;
    return d.matrixU() * isqrt(matrix(d.singularValues().asDiagonal()));
};
