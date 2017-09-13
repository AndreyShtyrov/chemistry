#include "linearAlgebraUtils.h"

matrix makeRandomMatrix(size_t rows, size_t cols)
{
    matrix matr(rows, cols);
    matr.setRandom();
    return matr;
};

vect makeRandomVect(size_t n)
{
    vect v(n);
    v.setRandom();
    return v;
};

vect makeRandomVect(vect const& lowerBound, vect const& upperBound)
{
    return lowerBound + (0.5 * (1 + makeRandomVect(lowerBound.rows()).array()) * (upperBound - lowerBound).array()).matrix();
}

vect makeConstantVect(size_t n, double constant)
{
    vect v(n);
    v.setConstant(constant);
    return v;
}

vect eye(size_t n, size_t i)
{
    vect result(n);
    result.setZero();
    result(i) = 1.;

    return result;
};

matrix makeConstantMatrix(size_t rows, size_t cols, double constant)
{
    matrix m(rows, cols);
    m.setConstant(constant);
    return m;
}

matrix identity(size_t nDims)
{
    return identity(nDims, nDims);
};

matrix identity(size_t rows, size_t cols)
{
    matrix result(rows, cols);
    result.setIdentity();
    return result;
};


matrix isqrt(matrix m, size_t removedCnt)
{
    for (size_t i = 0; i < m.rows() - removedCnt; i++)
        m(i, i) = 1. / sqrt(abs(m(i, i)));
    return m;
};

//m - symmetric matrix
matrix linearization(matrix m)
{
    return Eigen::JacobiSVD<matrix>(m, Eigen::ComputeFullU).matrixU();
}

//m - symmetric matrix
matrix linearizationNormalization(matrix m, size_t removedCnt)
{
    Eigen::JacobiSVD<matrix> d(m, Eigen::ComputeFullU);
    auto diagonal = d.singularValues();
    diagonal.block(diagonal.size() - removedCnt, 0, removedCnt, 1).setZero();
    return d.matrixU() * isqrt(matrix(diagonal.asDiagonal()), removedCnt);
};

matrix rotationMatrix(vect u, vect v, double alpha)
{
    return identity((size_t) u.rows()) + sin(alpha) * (u * v.transpose() - v * u.transpose()) +
           (cos(alpha) - 1) * (u * u.transpose() + v * v.transpose());
}

matrix rotationMatrix(vect from, vect to)
{
    vect v = to / to.norm();
    vect u = (from - v * from.dot(v));
    u /= u.norm();

    double alpha = acos(from.dot(to) / from.norm() / to.norm());

    return rotationMatrix(v, u, alpha);
}

vect randomVectOnSphere(size_t nDims, double r)
{
    vect v(nDims);
    normal_distribution<double> distribution(.0, 1.);
    for (size_t i = 0; i < nDims; i++)
        v(i) = distribution(randomGen);
    return v / v.norm() * r;
}

vect projection(vect wich, vect to)
{
    to /= to.norm();
    return wich.dot(to) * to;
}

matrix singularValues(matrix m)
{
    auto A = linearization(m);
    return (A.transpose() * m * A).diagonal().transpose();
}

vect toDistanceSpace(vect v, bool sorted)
{
    assert(v.rows() % 3 == 0);

    size_t n = (size_t) v.rows() / 3;

    vector<double> dists(n * (n - 1) / 2);
    for (size_t i = 0, num = 0; i < n;  i++)
        for (size_t j = i + 1; j < n; j++, num++)
            dists[num] = (v.block(i * 3, 0, 3, 1) - v.block(j * 3, 0, 3, 1)).norm();
    if (sorted)
        sort(dists.begin(), dists.end());

    vect res(dists.size());
    for (size_t i = 0; i < dists.size(); i++)
        res(i) = dists[i];

    return res;
};

vect normalized(vect const& v)
{
    return v / v.norm();
}

double distance(vect const& u, vect const& v)
{
    return (u - v).norm();
}

double angleCosine(vect const& u, vect const& v)
{
    return normalized(u).dot(normalized(v));
}

matrix verticalStack(matrix const& a, matrix const& b)
{
    assert(a.cols() == b.cols());

    matrix result(a.rows() + b.rows(), a.cols());
    result.block(0, 0, a.rows(), a.cols()) = a;
    result.block(a.rows(), 0, b.rows(), b.cols()) = b;
    return result;
}

matrix horizontalStack(matrix const& a, matrix const& b)
{
    assert(a.rows() == b.rows());

    matrix result(a.rows(), a.cols() + b.cols());
    result.block(0, 0, a.rows(), a.cols()) = a;
    result.block(0, a.cols(), b.rows(), b.cols()) = b;

    return result;
}