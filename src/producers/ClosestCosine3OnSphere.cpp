#include "ClosestCosine3OnSphere.h"

#include "linearAlgebraUtils.h"

ClosestCosine3OnSphere::ClosestCosine3OnSphere(size_t nDims, vector<double> values, vector<vect> directions)
   : FunctionProducer(nDims), mValues(move(values)), mDirections(move(directions))
{
    assert(mValues.size() == mDirections.size());
    for (size_t i = 0; i < mValues.size(); i++) {
        mSupplements.emplace_back(mDirections[i], mValues[i]);
    }
}

double ClosestCosine3OnSphere::operator()(vect const& x)
{
    size_t closest = getClosest(x);
    if (closest != (size_t) -1)
        return mSupplements[closest](x);

    return 0.;
}

vect ClosestCosine3OnSphere::grad(vect const& x)
{
    size_t closest = getClosest(x);
    if (closest != (size_t) -1)
        return mSupplements[closest].grad(x);

    return makeConstantVect(nDims, 0.);
}

matrix ClosestCosine3OnSphere::hess(vect const& x)
{
    size_t closest = getClosest(x);
    if (closest != (size_t) -1)
        return mSupplements[closest].hess(x);

    return makeConstantMatrix(nDims, nDims, 0.);
}

tuple<double, vect> ClosestCosine3OnSphere::valueGrad(vect const& x)
{
    return FunctionProducer::valueGrad(x);
};

tuple<double, vect, matrix> ClosestCosine3OnSphere::valueGradHess(vect const& x)
{
    return FunctionProducer::valueGradHess(x);
};

size_t ClosestCosine3OnSphere::getClosest(vect const& x)
{
    auto closest = (size_t) -1;
    double value = 0;

    for (size_t i = 0; i < mDirections.size(); i++)
        if (closest == (size_t) -1 || mDirections[i].dot(x) > value) {
            closest = i;
            value = mDirections[i].dot(x);
        }

    return closest;
}

