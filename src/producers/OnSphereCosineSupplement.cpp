#include "OnSphereCosineSupplement.h"

OnSphereCosineSupplement::OnSphereCosineSupplement(size_t nDims, vect direction, double value)
        : FunctionProducer(nDims), mDirection(move(direction)), mValue(value)
{
    mDirection.normalize();
}

double OnSphereCosineSupplement::operator()(vect const& x)
{
    double c = x.dot(mDirection);
    return mValue * c * c * c;
}

vect OnSphereCosineSupplement::grad(vect const& x)
{
    double c = x.dot(mDirection);
    return 3 * c * c * mDirection;
}

matrix OnSphereCosineSupplement::hess(vect const& x)
{
    double c = x.dot(mDirection);
    return 6 * c * mDirection * mDirection.transpose();
}

tuple<double, vect> OnSphereCosineSupplement::valueGrad(vect const& x)
{
    return FunctionProducer::valueGrad(x);
};

tuple<double, vect, matrix> OnSphereCosineSupplement::valueGradHess(vect const& x)
{
    return FunctionProducer::valueGradHess(x);
};


double OnSphereCosineSupplement::getCos(vect const& x) const
{
    return mDirection.dot(x / x.norm());
}
