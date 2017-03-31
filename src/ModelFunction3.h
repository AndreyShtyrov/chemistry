#pragma once

#include "helper.h"

#include "FunctionProducer.h"


class ModelFunction3 : public FunctionProducer
{
public:
//    using FunctionProducer<2>::N;

    ModelFunction3(double t=1.) : FunctionProducer(2), mT(t)
    { }

    virtual double operator()(vect const& x)
    {
        return 0.5 * sqr(x(0)) + 0.5 * sqr(x(1)) - mT * sqr(x(0)) * x(0) / 6;
    }

    virtual vect grad(vect const& x)
    {
        return makeVect(x(0) - mT * 0.5 * sqr(x(0)), x(1));
    }

    virtual matrix hess(vect const& x)
    {
        matrix hess(2, 2);
        hess(0, 0) = 1. - mT * x(0);
        hess(1, 0) = hess(0, 1) = 0.;
        hess(1, 1) = 1.;

        return hess;
    };

private:
    double mT;
};

//class ModelFunction4 : public FunctionProducer<3>
//{
//public:
//    using FunctionProducer<2>::N;
//
//    virtual double operator()(vect const& x)
//    {
//        return x(0) + sqr(x(1)) - mT * sqr(x(0)) * x(0) / 6;
//    }
//
//    virtual vect grad(vect const& x)
//    {
//        return make_vect(x(0) - mT * 0.5 * sqr(x(0)), x(1));
//    }
//
//    virtual matrix hess(vect const& x)
//    {
//        matrix hess;
//        hess(0, 0) = 1. - mT * x(0);
//        hess(1, 0) = hess(0, 1) = 0.;
//        hess(1, 1) = 1.;
//
//        return hess;
//    };
//
//private:
//    double mT;
//};