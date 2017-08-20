#include "FunctionProducer.h"

tuple<double, vect> FunctionProducer::valueGrad(vect const& x)
{
    return make_tuple((*this)(x), grad(x));
}

tuple<double, vect, matrix> FunctionProducer::valueGradHess(vect const& x)
{
    return make_tuple((*this)(x), grad(x), hess(x));
}