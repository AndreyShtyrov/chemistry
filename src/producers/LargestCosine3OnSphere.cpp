#include "LargestCosine3OnSphere.h"

#include "linearAlgebraUtils.h"

LargestCosine3OnSphere::LargestCosine3OnSphere(size_t nDims, vector<double> values, vector<vect> directions)
   : FunctionProducer(nDims), mValues(move(values)), mDirections(move(directions))
{
    assert(mValues.size() == mDirections.size());
    for (size_t i = 0; i < mValues.size(); i++) {
        mSupplements.emplace_back(mDirections[i], mValues[i]);
    }
}

double LargestCosine3OnSphere::operator()(vect const& x)
{
    if (!mValues.empty())
        return mSupplements[getLargest(x)](x);

    return 0.;
}

vect LargestCosine3OnSphere::grad(vect const& x)
{
    if (!mValues.empty())
        return mSupplements[getLargest(x)].grad(x);

    return makeConstantVect(nDims, 0);
}

matrix LargestCosine3OnSphere::hess(vect const& x)
{
    if (!mValues.empty())
        return mSupplements[getLargest(x)].hess(x);

    return makeConstantMatrix(nDims, nDims, 0);
}

tuple<double, vect> LargestCosine3OnSphere::valueGrad(vect const& x)
{
    return FunctionProducer::valueGrad(x);
};

tuple<double, vect, matrix> LargestCosine3OnSphere::valueGradHess(vect const& x)
{
    return FunctionProducer::valueGradHess(x);
};

size_t LargestCosine3OnSphere::getLargest(vect const& x)
{
    double value = 0;
    size_t closest = 0;

    for (size_t i = 0; i < mDirections.size(); i++) {
        double current = mSupplements[i](x);
        if (value < current) {
            closest = i;
            value = current;
        }
    }

    return closest;
}

