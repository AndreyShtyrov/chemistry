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

    const size_t nDims;
};
