#include "Cosine3OnSphereInterpolation.h"

#include "linearAlgebraUtils.h"

Cosine3OnSphereInterpolation::Cosine3OnSphereInterpolation(size_t nDims, vector<double> const& values, vector<vect> const& directions)
        : FunctionProducer(nDims)
{
    assert(values.size() == directions.size());
    for (size_t i = 0; i < values.size(); i++) {
        mSupplements.emplace_back(directions[i], values[i]);
    }
}

double Cosine3OnSphereInterpolation::operator()(vect const& x)
{
    double value = 0;
    for (auto& supplement : mSupplements)
        value += supplement(x);
}

vect Cosine3OnSphereInterpolation::grad(vect const& x)
{
    auto grad = makeConstantVect(nDims, 0);
    for (auto& supplement : mSupplements)
        grad += supplement.grad(x);
    return grad;
}

matrix Cosine3OnSphereInterpolation::hess(vect const& x)
{
    auto hess = makeConstantMatrix(nDims, nDims, 0);
    for (auto& supplement : mSupplements)
        hess += supplement.hess(x);
    return hess;
}

tuple<double, vect> Cosine3OnSphereInterpolation::valueGrad(vect const& x)
{
    return FunctionProducer::valueGrad(x);
};

tuple<double, vect, matrix> Cosine3OnSphereInterpolation::valueGradHess(vect const& x)
{
    return FunctionProducer::valueGradHess(x);
};
