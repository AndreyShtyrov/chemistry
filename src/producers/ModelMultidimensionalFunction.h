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

    double transform(vect const& x)
    {
        double value = 0;
        for (size_t i = 0; i < (size_t) N; i++)
            value += mCoeffs[i] * x(i);
        return value;
    }

    double operator()(vect const& x)
    {
        return exp(transform(x));
    }

    vect grad(vect const& x)
    {
        vect grad;
        for (size_t i = 0; i < (size_t) N; i++)
            grad(i) = mCoeffs[i];

        return grad * operator()(x);
    }

    matrix hess(vect const& x)
    {
        matrix hess;
        for (size_t i = 0; i < (size_t) N; i++)
            for (size_t j = 0; j < (size_t) N; j++)
                hess(i, j) = mCoeffs[i] * mCoeffs[j];

        return hess * operator()(x);
    };

private:
    array<double, N> mCoeffs;
};


template<size_t N_DIMS>
class ModelMultidimensionalZeroHessFunction : FunctionProducer
{
public:
    ModelMultidimensionalZeroHessFunction() : FunctionProducer(N_DIMS)
    {
        for (size_t i = 0; i < nDims; i++)
            mCoeffs[i] = 2. * rand() / RAND_MAX - 1.;
    }

    double transform(vect const& x)
    {
        double value = 0;
        for (size_t i = 0; i < nDims; i++)
            value += mCoeffs[i] * x(i);
        return value;
    }

    double operator()(vect const& x)
    {
        return transform(x);
    }

    vect grad(vect const& x)
    {
        vect grad(nDims);
        for (size_t i = 0; i < nDims; i++)
            grad(i) = mCoeffs[i];

        return grad;
    }

    matrix hess(vect const& x)
    {
        return makeConstantMatrix(nDims, nDims, 0);
    };

private:
    array<double, N_DIMS> mCoeffs;
};