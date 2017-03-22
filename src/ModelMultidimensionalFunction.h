#pragma once

#include "helper.h"

template<int N_DIMS>
class ModelMultidimensionalFunction
{
public:
    static constexpr int N = N_DIMS;

    ModelMultidimensionalFunction()
    {
        for (size_t i = 0; i < N; i++)
            mCoeffs[i] = 2. * rand() / RAND_MAX - 1.;
    }

    double transform(vect<N> const& x)
    {
        double value = 0;
        for (size_t i = 0; i < (size_t) N; i++)
            value += mCoeffs[i] * x(i);
        return value;
    }

    double operator()(vect<N> const& x)
    {
        return exp(transform(x));
    }

    vect<N> grad(vect<N> const& x)
    {
        vect<N> grad;
        for (size_t i = 0; i < (size_t) N; i++)
            grad(i) = mCoeffs[i];

        return grad * operator()(x);
    }

    matrix<N, N> hess(vect<N> const& x)
    {
        matrix<N, N> hess;
        for (size_t i = 0; i < (size_t) N; i++)
            for (size_t j = 0; j < (size_t) N; j++)
                hess(i, j) = mCoeffs[i] * mCoeffs[j];

        return hess * operator()(x);
    };

private:
    array<double, N> mCoeffs;
};


template<int N_DIMS>
class ModelMultidimensionalZeroHessFunction
{
public:
    static constexpr int N = N_DIMS;

    ModelMultidimensionalZeroHessFunction()
    {
        for (size_t i = 0; i < N; i++)
            mCoeffs[i] = 2. * rand() / RAND_MAX - 1.;
    }

    double transform(vect<N> const& x)
    {
        double value = 0;
        for (size_t i = 0; i < (size_t) N; i++)
            value += mCoeffs[i] * x(i);
        return value;
    }

    double operator()(vect<N> const& x)
    {
        return transform(x);
    }

    vect<N> grad(vect<N> const& x)
    {
        vect<N> grad;
        for (size_t i = 0; i < (size_t) N; i++)
            grad(i) = mCoeffs[i];

        return grad;
    }

    matrix<N, N> hess(vect<N> const& x)
    {
        return makeConstantMatrix<N, N>(0);
    };

private:
    array<double, N> mCoeffs;
};