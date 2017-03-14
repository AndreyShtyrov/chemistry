#pragma once

#include "helper.h"

#include "FunctionProducer.h"

template<typename FuncT>
class InPolar : public FunctionProducer<FuncT::N - 1>
{
public:
    static constexpr int N = FuncT::N - 1;

    InPolar(FuncT const& func, double r)
            : mFunc(func), mR(r)
    { }

    virtual double operator()(vect<N> const& phi) override
    {
        return mFunc(transform(phi));
    }

    virtual vect<N> grad(vect<N> const& phi) override
    {
        auto grad = mFunc.grad(transform(phi));

        vect<N> result;
        result.setZero();

        double firstSinProduct = 1.;
        for (size_t i = 0; i < N; i++) {
            double secondSinProduct = 1.;
            for (size_t j = i; j < N + 1; j++) {
                double fact = 0;
                if (i == j) {
                    fact = -firstSinProduct * sin(phi(j));
                }
                else {
                    fact = firstSinProduct * cos(phi(i)) * secondSinProduct;
                    if (j < N)
                        fact *= cos(phi(j));
                }

                result(i) += mR * fact * grad(j);
                if (i < j && j < N)
                    secondSinProduct *= sin(phi(j));
            }

            firstSinProduct *= sin(phi(i));
        }

        return result;
    }

    virtual matrix<N, N> hess(vect<N> const& x) override
    {
        assert(false);
    }

    vect<N + 1> transform(vect<N> const &phi)
    {
        vect<N + 1> x;
        double sinProduct = 1;
        for (size_t i = 0; i < N; i++) {
            x(i) = mR * sinProduct * cos(phi[i]);
            sinProduct *= sin(phi[i]);
        }
        x(N) = mR * sinProduct;

        return x;
    }

public:
    FuncT mFunc;
    double mR;
};

template<typename FuncT>
InPolar<FuncT> make_polar(FuncT const& func, double r)
{
    return InPolar<FuncT>(func, r);
}
