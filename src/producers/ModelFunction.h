#pragma once

#include "helper.h"

#include "FunctionProducer.h"

class ModelFunction : public FunctionProducer
{
public:
    explicit ModelFunction(double a = 1., double b = 1., double c = 1.)
            : FunctionProducer(2), mA(a), mB(b), mC(c)
    { }

    double operator()(vect const& p) override
    {
        double x2 = sqr(p(0));
        double y2 = sqr(p(1));

        return (mA - mB * y2) * x2 * exp(-x2) + 0.5 * mC * y2;
    }

    vect grad(vect const& p) override
    {
        double x = p(0);
        double y = p(1);
        double x2 = sqr(x);

        vect res(2);
        res(0) = 2 * (mA - mB * sqr(y)) * exp(-x2) * (x - x * x2);
        res(1) = (mC - 2 * mB * x2 * exp(-x2)) * y;

        return res;
    }

    matrix hess(vect const& p) override
    {
        double x = p(0);
        double y = p(1);
        double x2 = sqr(x);
        double y2 = sqr(y);
        double e = exp(-x2);

        double xx = 2 * (mA - mB * y2) * e * (2 * sqr(x2) - 5 * x2 + 1);
        double xy = 4 * mB * y * e * x * (x - 1) * (x + 1);
        double yy = mC - 2 * mB * x2 * e;

        matrix hess(2, 2);
        hess(0, 0) = xx;
        hess(0, 1) = hess(1, 0) = xy;
        hess(1, 1) = yy;

        return hess;
    }

private:
    double mA, mB, mC;
};


class SecondModelFunction : public FunctionProducer
{
public:
    SecondModelFunction () : FunctionProducer(2)
    { }

    double operator()(vect const& p) override
    {
        double x = p(0);
        double y = p(1);

        return sin(0.5 * sqr(x) - 0.25 * sqr(y) + 3) * cos(2 * x + 1 - exp(y));
    }

    vect grad(vect const& p) override
    {
        double x = p(0);
        double y = p(1);
        double x2 = sqr(x);
        double y2 = sqr(y);

        vect res;
        res(0) = x * cos(0.5 * x2 - 0.25 * y2 + 3) * cos(2 * x + 1 - exp(y)) - 2 * sin(0.5 * x2 - 0.25 * y2 +  3) * sin(2 * x + 1 - exp(y));
        res(1) = 0.5 * y * cos(0.5 * x2 - 0.25 * y2 + 3) * cos(2 * x + 1 - exp(y)) - exp(y) * sin(0.5 * x2 - 0.25 * y2 +  3) * sin(2 * x + 1 - exp(y));

        return res;
    }

    matrix hess(vect const& p) override
    {
        assert(false);
//        double x = p(0);
//        double y = p(1);
//        double x2 = sqr(x);
//        double y2 = sqr(y);
//        double e = exp(-x2);
//
//        double xx = 2 * (mA - mB * y2) * e * (2 * sqr(x2) - 5 * x2 + 1);
//        double xy = 4 * mB * y * e * x * (x - 1) * (x + 1);
//        double yy = mC - 2 * mB * x2 * e;
//
//        matrix<2, 2> hess;
//        hess(0, 0) = xx;
//        hess(0, 1) = hess(1, 0) = xy;
//        hess(1, 1) = yy;
//
//        return hess;
    }
};