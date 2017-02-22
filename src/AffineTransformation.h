#pragma once

#include "helper.h"

#include "FunctionProducer.h"

template<typename FuncT>
class AffineTransformation : public FunctionProducer<FuncT::N>
{
public:
    using FunctionProducer<FuncT::N>::N;

    AffineTransformation(FuncT const& func, matrix<N, N> const& basis)
            : mFunc(func),
              mBasis(basis), mBasisT(basis.transpose())
    { }

    virtual double operator()(vect<N> const& x) override
    {
        return mFunc(mBasis * x);
    }

    virtual vect<N> grad(vect<N> const& x) override
    {
        return mBasisT * mFunc.grad(mBasis * x);
    }

    virtual matrix<N, N> hess(vect<N> const& x) override
    {
        return mBasisT * mFunc.hess(mBasis * x) * mBasis;
    }

private:
    FuncT mFunc;
    matrix<N, N> mBasis;
    matrix<N, N> mBasisT;
};

template <typename FuncT>
AffineTransformation<FuncT> make_affine_transfomation(FuncT const& func, matrix<FuncT::N, FuncT::N> const& A)
{
    return AffineTransformation<FuncT>(func, A);
}
