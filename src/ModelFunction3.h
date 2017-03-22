#pragma once

#include "helper.h"

#include "FunctionProducer.h"


class ModelFunction3 : public FunctionProducer<2>
{
public:
    using FunctionProducer<2>::N;

    ModelFunction3(double t=1.) : mT(t)
    { }

    virtual double operator()(vect<N> const& x)
    {
        return 0.5 * sqr(x(0)) + 0.5 * sqr(x(1)) - mT * sqr(x(0)) * x(0) / 6;
    }

    virtual vect<N> grad(vect<N> const& x)
    {
        return make_vect(x(0) - mT * 0.5 * sqr(x(0)), x(1));
    }

    virtual matrix<N, N> hess(vect<N> const& x)
    {
        matrix<N, N> hess;
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
//    virtual double operator()(vect<N> const& x)
//    {
//        return x(0) + sqr(x(1)) - mT * sqr(x(0)) * x(0) / 6;
//    }
//
//    virtual vect<N> grad(vect<N> const& x)
//    {
//        return make_vect(x(0) - mT * 0.5 * sqr(x(0)), x(1));
//    }
//
//    virtual matrix<N, N> hess(vect<N> const& x)
//    {
//        matrix<N, N> hess;
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