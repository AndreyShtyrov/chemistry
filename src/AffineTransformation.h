#pragma once

#include "helper.h"

#include "FunctionProducer.h"
#include "linearization.h"

template<typename FuncT>
class AffineTransformation : public FunctionProducer<FuncT::N>
{
public:
    using FunctionProducer<FuncT::N>::N;

    AffineTransformation(FuncT func, vect<N> delta, matrix<N, N> const& basis) : mFunc(move(func)), mDelta(move(delta)),
                                                                                 mBasis(basis),
                                                                                 mBasisT(basis.transpose())
    {}

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

    vect<N> transform(vect<N> const& x) const
    {
        return mBasis * x + mDelta;
    }

    FuncT const& getInnerFunction() const
    {
        return mFunc;
    }

private:
    FuncT mFunc;
    vect<N> mDelta;
    matrix<N, N> mBasis;
    matrix<N, N> mBasisT;
};

template<typename FuncT>
auto makeAffineTransfomation(FuncT&& func, vect<decay_t<FuncT>::N> delta,
                             matrix<decay_t<FuncT>::N, decay_t<FuncT>::N> const& A)
{
    return AffineTransformation<decay_t<FuncT>>(forward<FuncT>(func), move(delta), A);
}

template<typename FuncT>
auto prepareForPolar(FuncT&& func, vect<decay_t<FuncT>::N> const& v)
{
    auto A = linearizationNormalization(func.hess(v));
    return makeAffineTransfomation(func, v, A);
}
