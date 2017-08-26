#include "OnSphereCosineSupplement.h"

OnSphereCosineSupplement::OnSphereCosineSupplement(vect direction, double value)
        : FunctionProducer((size_t) direction.size()), mDirection(move(direction)), mValue(value)
{
    mDirection.normalize();
}

double OnSphereCosineSupplement::operator()(vect const& x)
{
    double c = getCosine(x);
    return mValue * c * c * c;
}

vect OnSphereCosineSupplement::grad(vect const& x)
{
    double c = getCosine(x);
    return mValue * 3 * c * c * mDirection;
}

matrix OnSphereCosineSupplement::hess(vect const& x)
{
    double c = getCosine(x);
    return mValue * 6 * c * mDirection * mDirection.transpose();
}

tuple<double, vect> OnSphereCosineSupplement::valueGrad(vect const& x)
{
    return FunctionProducer::valueGrad(x);
};

tuple<double, vect, matrix> OnSphereCosineSupplement::valueGradHess(vect const& x)
{
    return FunctionProducer::valueGradHess(x);
};


double OnSphereCosineSupplement::getCosine(vect const& x) const
{
    return max(0., mDirection.dot(x));
}
