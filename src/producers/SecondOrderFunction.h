#pragma once

#include "helper.h"
#include "FunctionProducer.h"

class SecondOrderFunction : public FunctionProducer
{
public:
    SecondOrderFunction(double value, vect grad, matrix hess);

    double operator()(vect const& x);
    vect grad(vect const& x);
    matrix hess(vect const& x);

    tuple<double, vect> valueGrad(vect const& x);
    tuple<double, vect, matrix> valueGradHess(vect const& x);


private:
    double mValue;
    vect mGrad;
    matrix mHess;
};