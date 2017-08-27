#pragma once

#include "helper.h"

#include <producers/FunctionProducer.h>
#include <producers/OnSphereCosineSupplement.h>

class Cosine3OnSPhereInterpolation : public FunctionProducer
{
public:
    Cosine3OnSPhereInterpolation(size_t nDims, vector<double> const& values, vector<vect> const& directions);

    double operator()(vect const& x) override;
    vect grad(vect const& x) override;
    matrix hess(vect const& x) override;
    tuple<double, vect> valueGrad(vect const& x) override;
    tuple<double, vect, matrix> valueGradHess(vect const& x) override;

private:
    vector<OnSphereCosineSupplement> mSupplements;
};
