#pragma once

#include "helper.h"

#include "FunctionProducer.h"

class SqrNorm : public FunctionProducer
{
public:
//    using FunctionProducer<N_DIMS>::N;

    SqrNorm(size_t nDims) : FunctionProducer(nDims)
    { }

    virtual double operator()(vect const& x) override
    {
        return x.transpose() * x;
    }

    virtual vect grad(vect const& x) override
    {
        return 2 * x;
    }

    virtual matrix hess(vect const& x) override
    {
        return 2. * matrix::Identity(x.rows(), x.rows());
    }
};
