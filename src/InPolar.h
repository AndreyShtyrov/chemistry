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

//        vect<N> result;
//        result.setZero();
//
//        double firstSinProduct = 1.;
//        for (size_t i = 0; i < N; i++) {
//            double secondSinProduct = 1.;
//            for (size_t j = i; j < N + 1; j++) {
//                double fact = 0;
//                if (i == j) {
//                    fact = -firstSinProduct * sin(phi(j));
//                }
//                else {
//                    fact = firstSinProduct * cos(phi(i)) * secondSinProduct;
//                    if (j < N)
//                        fact *= cos(phi(j));
//                }
//
//                result(i) += mR * fact * grad(j);
//                if (i < j && j < N)
//                    secondSinProduct *= sin(phi(j));
//            }
//
//            firstSinProduct *= sin(phi(i));
//        }

//        array<array<double, N + 1>, N + 1> products;
//        for (size_t i)

        matrix<N, N + 1> m;
        for (size_t i = 0; i < N + 1; i++)
            for (size_t j = 0; j < N; j++)  {
                if (i < j) {
                    m(j, i) = 0;
                    continue;
                }

                double product = 1.;

                for (size_t k = 0; k < min(i + 1, (size_t) N); k++)
                    if (k == j && k == i)
                        product *= -sin(phi(k));
                    else if (k == j || k == i)
                        product *= cos(phi(k));
                    else if (k <= i)
                        product *= sin(phi(k));

                m(j, i) = product;
            }

        return mR * m * grad;
    }

    virtual matrix<N, N> hess(vect<N> const& x) override
    {
        assert(false);
    }

    vect<N + 1> transform(vect<N> const &phi) const
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

    FuncT const& getInnerFunction() const
    {
        return mFunc;
    }

public:
    FuncT mFunc;
    double mR;
};

template<typename FuncT>
InPolar<FuncT> makePolar(FuncT const& func, double r)
{
    return InPolar<FuncT>(func, r);
}
