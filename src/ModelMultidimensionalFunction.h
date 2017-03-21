#pragma once

#include "helper.h"

class ModelMultidimensionalFunction
{
public:
    static constexpr int N = 20;

    double operator()(vect<N> const& x)
    {
        double value = 0;
        for (size_t i = 0; i < (size_t) N; i++)
            value += (i % 4 - 1.5) * x(i);
        return value;
    }

    vect<N> grad(vect<N> const& x)
    {
        auto grad = makeConstantVect<N>(0);
        for (size_t i = 0; i < (size_t) N; i++)
            grad(i) = i % 4 - 1.5;
        return grad;
    }

    matrix<N, N> hess(vect<N> const& x)
    {
        assert(false);
    };
};