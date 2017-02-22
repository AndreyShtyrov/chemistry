#pragma once

#include "helper.h"

#include "FunctionProducer.h"

template<int N_DIMS>
class SqrNorm : public FunctionProducer<N_DIMS>
{
public:
    using FunctionProducer<N_DIMS>::N;

    virtual double operator()(vect<N> const& x) override
    {
        return x.transpose() * x;
    }

    virtual vect<N> grad(vect<N> const& x) override
    {
        return 2 * x;
    }

    virtual matrix<N, N> hess(vect<N> const& x) override
    {
        return 2. * matrix<N, N>::Identity();
    }
};
