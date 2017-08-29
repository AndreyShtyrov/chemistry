#pragma once

#include "helper.h"

#include "FunctionProducer.h"

class SqrNorm : public FunctionProducer
{
public:
    explicit SqrNorm(size_t nDims);

    double operator()(vect const& x) override;
    vect grad(vect const& x) override;
    matrix hess(vect const& x) override;
    tuple<double, vect> valueGrad(vect const& x) override;
    tuple<double, vect, matrix> valueGradHess(vect const& x) override;
};
