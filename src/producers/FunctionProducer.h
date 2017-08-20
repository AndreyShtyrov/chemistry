#pragma once

#include "helper.h"

class FunctionProducer
{
public:
    FunctionProducer(size_t nDims) : nDims(nDims)
    { }

    virtual ~FunctionProducer() = default;

    virtual double       operator()(vect const& x) = 0;
    virtual vect      grad(vect const& x) = 0;
    virtual matrix hess(vect const& x) = 0;

    virtual tuple<double, vect> valueGrad(vect const& x);
    virtual tuple<double, vect, matrix> valueGradHess(vect const& x);

    const size_t nDims;
};
