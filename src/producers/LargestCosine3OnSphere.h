#pragma once

#include "helper.h"

#include <producers/FunctionProducer.h>
#include <producers/OnSphereCosineSupplement.h>

class LargestCosine3OnSphere : public FunctionProducer
{
public:
    LargestCosine3OnSphere(size_t nDims, vector<double> values, vector<vect> directions);

    double operator()(vect const& x) override;
    vect grad(vect const& x) override;
    matrix hess(vect const& x) override;
    tuple<double, vect> valueGrad(vect const& x) override;
    tuple<double, vect, matrix> valueGradHess(vect const& x) override;

private:
    vector<double> mValues;
    vector<vect> mDirections;
    vector<OnSphereCosineSupplement> mSupplements;

    size_t getLargest(vect const& x);
};
