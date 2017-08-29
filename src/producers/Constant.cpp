#include "producers/Constant.h"

#include "linearAlgebraUtils.h"

Constant::Constant(size_t nDims, double value) : FunctionProducer(nDims), mValue(value)
{ }

double Constant::operator()(vect const& x)
{
    return mValue;
}

vect Constant::grad(vect const& x)
{
    return makeConstantVect(nDims, 0.);
}

matrix Constant::hess(vect const& x)
{
    return makeConstantMatrix(nDims, nDims, 0.);
};

tuple<double, vect> Constant::valueGrad(vect const& x)
{
    return FunctionProducer::valueGrad(x);
};

tuple<double, vect, matrix> Constant::valueGradHess(vect const& x)
{
    return FunctionProducer::valueGradHess(x);
};
