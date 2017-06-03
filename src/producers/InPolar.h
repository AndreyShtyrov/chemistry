#pragma once

#include "helper.h"

#include "FunctionProducer.h"
#include "linearAlgebraUtils.h"

template<typename FuncT>
class InPolar : public FunctionProducer
{
public:
    InPolar(FuncT func, double r)
            : FunctionProducer(func.nDims - 1), mFunc(move(func)), mR(r)
    { }

    virtual double operator()(vect const& phi) override
    {
        return mFunc(transform(phi));
    }

    virtual vect grad(vect const& phi) override
    {
        auto grad = mFunc.grad(transform(phi));
        return grad.transpose() * calculateDerivatives(phi);
    }

    virtual matrix hess(vect const& phi) override
    {
        auto grad = mFunc.grad(transform(phi));
        auto hess = mFunc.hess(transform(phi));

        auto m = calculateDerivatives(phi);

        matrix result(nDims, nDims);
        result.setConstant(0.);

        for (size_t i = 0; i < nDims; i++) {
            for (size_t j = 0; j < nDims; j++) {
                for (size_t k = 0; k < nDims + 1; k++) {
                    result(i, j) += (hess.row(k) * m.col(j) * m(k, i))(0, 0);
                    result(i, j) += grad(k) * calculateSecondDerivatives(phi, i)(k, j);
                }
            }
        }

        return result;
    }

    vect transform(vect const &phi) const
    {
        vect x(nDims + 1);
        double sinProduct = 1;
        for (size_t i = 0; i < nDims; i++) {
            x(i) = mR * sinProduct * cos(phi[i]);
            sinProduct *= sin(phi[i]);
        }
        x(nDims) = mR * sinProduct;

        return x;
    }

    vect fullTransform(vect const& phi) const
    {
        return mFunc.fullTransform(transform(phi));
    }

    FuncT const& getInnerFunction() const
    {
        return mFunc;
    }

public:
    matrix calculateDerivatives(vect const& phi)
    {
        matrix m(nDims + 1, nDims);
        for (size_t i = 0; i < nDims + 1; i++)
            for (size_t j = 0; j < nDims; j++)  {
                if (i < j) {
                    m(i, j) = 0;
                    continue;
                }

                double product = 1.;

                for (size_t k = 0; k < min(i + 1, nDims); k++)
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

    matrix calculateSecondDerivatives(vect const& phi, size_t j)
    {
        matrix m(nDims + 1, nDims);
        for (size_t i = 0; i < nDims + 1; i++)
            for (size_t k = 0; k < nDims; k++)  {
                if (i < j || i < k) {
                    m(i, k) = 0;
                    continue;
                }

                double product = 1.;

                for (size_t l = 0; l < min(i + 1, nDims); l++)
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

vect polarVectLowerBound(int rows)
{
    return makeConstantVect(rows, 0.);
}

vect polarVectUpperBound(int rows)
{
    if (rows == 1)
        return makeConstantVect(1, 2 * M_PI);

    vect upperBound;
    upperBound.block(0, 0, rows - 1, 1).setConstant(2 * M_PI);
    upperBound.block(rows - 1, 0, 1, 1).setConstant(M_PI);

    return upperBound;
}

vect randomPolarPoint(size_t nDims)
{
    vect v(nDims);
    normal_distribution<double> distribution(.0, 1.);
    for (size_t i = 0; i < nDims; i++)
        v(i) = distribution(randomGen);
    v /= v.norm();

    vect phi(nDims - 1);
    double sinProduct = 1;
    for (size_t i = 0; i < nDims - 1; i++) {
        double cur = acos(v(i) / sinProduct);
        sinProduct *= sin(cur);
        phi(i) = cur;
    }

    return phi;
}
