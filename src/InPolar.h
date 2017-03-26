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

        return grad.transpose() * calculateDerivatives(phi);
    }

    virtual matrix<N, N> hess(vect<N> const& phi) override
    {
        auto grad = mFunc.grad(transform(phi));
        auto hess = mFunc.hess(transform(phi));

        auto m = calculateDerivatives(phi);

        matrix<N, N> result;
        result.setConstant(0.);

        for (size_t i = 0; i < N; i++) {
            for (size_t j = 0; j < N; j++) {
                for (size_t k = 0; k < N + 1; k++) {
                    matrix<1, 1> val = hess.row(k) * m.col(j) * m(k, i);
                    result(i, j) += val(0, 0);
                    result(i, j) += grad(k) * calculateSecondDerivatives(phi, i)(k, j);
                }
            }
        }

        return result;
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

    }

public:
    matrix<N + 1, N> calculateDerivatives(vect<N> const& phi)
    {
        matrix<N + 1, N> m;
        for (size_t i = 0; i < N + 1; i++)
            for (size_t j = 0; j < N; j++)  {
                if (i < j) {
                    m(i, j) = 0;
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

                m(i, j) = mR * product;
            }
        return m;
    }

    matrix<N + 1, N> calculateSecondDerivatives(vect<N> const& phi, size_t j)
    {
        matrix<N + 1, N> m;
        for (size_t i = 0; i < N + 1; i++)
            for (size_t k = 0; k < N; k++)  {
                if (i < j || i < k) {
                    m(i, k) = 0;
                    continue;
                }

                double product = 1.;

                for (size_t l = 0; l < min(i + 1, (size_t) N); l++)
                    if (l == i) {
                        if (l == j && l == k)
                            product *= -cos(phi(l));
                        else if (l == j || l == k)
                            product *= -sin(phi(l));
                        else
                            product *= cos(phi(l));
                    }
                    else {
                        if (l == j && l == k)
                            product *= -sin(phi(l));
                        else if (l == j || l == k)
                            product *= cos(phi(l));
                        else
                            product *= sin(phi(l));
                    }

                m(i, k) = mR * product;
            }

        return m;
    };

    FuncT mFunc;
    double mR;
};

template<typename FuncT>
InPolar<FuncT> makePolar(FuncT const& func, double r)
{
    return InPolar<FuncT>(func, r);
}

template<int N>
vect<N> polarVectLowerBound()
{
    return makeConstantVect<N>(0.);
}

template<int N>
vect<N> polarVectUpperBound()
{
    vect<N> upperBound;
    upperBound.template block<N - 1, 1>(0, 0).setConstant(2 * M_PI);
    upperBound.template block<1, 1>(N - 1, 0).setConstant(M_PI);

    return upperBound;
}

template<>
vect<1> polarVectUpperBound<1>()
{
    return makeConstantVect<1>(2 * M_PI);
}