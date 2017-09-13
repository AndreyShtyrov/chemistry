//
// Created by george on 10.09.17.
//

#include "SecondOrderFunction.h"

SecondOrderFunction::SecondOrderFunction(double value, vect grad, matrix hess) :
   FunctionProducer(grad.size()), mValue(value), mGrad(move(grad)), mHess(move(hess))
{
    assert(grad.size() == hess.rows() && hess.rows() == hess.cols());
}

double SecondOrderFunction::operator()(vect const& x)
{
    return mValue + x.transpose() * mGrad + .5 * x.transpose() * mHess * x;
}

vect SecondOrderFunction::grad(vect const& x)
{
    return mGrad + mHess * x;
}

matrix SecondOrderFunction::hess(vect const& x)
{
    return mHess;
}

tuple<double, vect> SecondOrderFunction::valueGrad(vect const& x)
{
    return FunctionProducer::valueGrad(x);
};

tuple<double, vect, matrix> SecondOrderFunction::valueGradHess(vect const& x)
{
    return FunctionProducer::valueGradHess(x);
};

