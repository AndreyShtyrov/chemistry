#pragma once

#include "helper.h"

#include "FunctionProducer.h"

class Constant : public FunctionProducer
{
public:
//    using FunctionProducer<N_DIMS>::N;

    Constant(int nDims, double value) : FunctionProducer(nDims), mValue(value)
    { }

    virtual double operator()(vect const& x)
    {
        return mValue;
    }

    virtual vect grad(vect const& x)
    {
        vect grad(x.rows());
        grad.setZero();
        return grad;
    }

    virtual matrix hess(vect const& x)
    {
        matrix hess(x.rows(), x.rows());
        hess.setZero();
        return hess;
    };

private:
    double mValue;
};
