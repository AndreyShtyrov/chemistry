#pragma once

#include "helper.h"

#include "FunctionProducer.h"

template<int N_DIMS>
class Constant : public FunctionProducer<N_DIMS>
{
public:
    using FunctionProducer<N_DIMS>::N;

    Constant(double value) : mValue(value)
    { }

    virtual double operator()(vect<N> const& x)
    {
        return mValue;
    }

    virtual vect<N> grad(vect<N> const& x)
    {
        vect<N> grad;
        grad.setZero();
        return grad;
    }

    virtual matrix<N, N> hess(vect<N> const& x)
    {
        matrix<N, N> hess;
        hess.setZero();
        return hess;
    };

private:
    double mValue;
};
