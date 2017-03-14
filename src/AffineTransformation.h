#pragma once

#include "helper.h"

#include "FunctionProducer.h"

template<typename FuncT>
class AffineTransformation : public FunctionProducer<FuncT::N>
{
public:
    using FunctionProducer<FuncT::N>::N;

    AffineTransformation(FuncT const& func, vect<N> const& delta, matrix<N, N> const& basis)
            : mFunc(func), mDelta(delta), mBasis(basis), mBasisT(basis.transpose())
    { }

    virtual double operator()(vect<N> const& x) override
    {
        return mFunc(transform(x));
    }

    virtual vect<N> grad(vect<N> const& x) override
    {
        return mBasisT * mFunc.grad(transform(x));
    }

    virtual matrix<N, N> hess(vect<N> const& x) override
    {
        return mBasisT * mFunc.hess(transform(x)) * mBasis;
    }

    vect<N> transform(vect<N> const& x)
    {
        return mBasis * x + mDelta;
    }

private:
    FuncT mFunc;
    vect<N> mDelta;
    matrix<N, N> mBasis;
    matrix<N, N> mBasisT;
};

template <typename FuncT>
AffineTransformation<FuncT> make_affine_transfomation(FuncT const& func, vect<FuncT::N> const& delta, matrix<FuncT::N, FuncT::N> const& A)
{
    return AffineTransformation<FuncT>(func, delta, A);
}
