#pragma once

#include "helper.h"

#include <producers/FunctionProducer.h>

class OnSphereCosineSupplement : public FunctionProducer
{
public:
    OnSphereCosineSupplement(size_t nDims, vect direction, double value);

    double operator()(vect const& x) override;
    vect grad(vect const& x) override;
    matrix hess(vect const& x) override;
    tuple<double, vect> valueGrad(vect const& x) override;
    tuple<double, vect, matrix> valueGradHess(vect const& x) override;

private:
    double getCos(vect const& x) const;

    vect mDirection;
    double mValue;
};
