#include "SqrNorm.h"

SqrNorm::SqrNorm(size_t nDims) : FunctionProducer(nDims)
{ }

double SqrNorm::operator()(vect const& x)
{
    return x.transpose() * x;
}

vect SqrNorm::grad(vect const& x)
{
    return 2 * x;
}

matrix SqrNorm::hess(vect const& x)
{
    return 2. * matrix::Identity(x.rows(), x.rows());
}

tuple<double, vect> SqrNorm::valueGrad(vect const& x)
{
    return FunctionProducer::valueGrad(x);
};

tuple<double, vect, matrix> SqrNorm::valueGradHess(vect const& x)
{
    return FunctionProducer::valueGradHess(x);
};
