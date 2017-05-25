#pragma once

#include "helper.h"

#include "FunctionProducer.h"
#include "linearAlgebraUtils.h"

template<typename FuncT>
class AffineTransformation : public FunctionProducer
{
public:
    AffineTransformation(FuncT func, vect delta, matrix const& basis) : FunctionProducer(func.nDims), mFunc(move(func)), mDelta(move(delta)),
                                                                                 mBasis(basis),
                                                                                 mBasisT(basis.transpose())
    {}

    virtual double operator()(vect const& x) override
    {
        return mFunc(transform(x));
    }

    virtual vect grad(vect const& x) override
    {
        return mBasisT * mFunc.grad(transform(x));
    }

    virtual matrix hess(vect const& x) override
    {
        return mBasisT * mFunc.hess(transform(x)) * mBasis;
    }

    vect transform(vect const& x) const
    {
        return mBasis * x + mDelta;
    }

    vect fullTransform(vect const& x) const
    {
        return mFunc.fullTransform(transform(x));
    }

    FuncT const& getInnerFunction() const
    {
        return mFunc;
    }

private:
    FuncT mFunc;

    size_t mN;
    vect mDelta;
    matrix mBasis;
    matrix mBasisT;
};

template<typename FuncT>
auto makeAffineTransfomation(FuncT&& func, vect delta)
{
    return AffineTransformation<decay_t<FuncT>>(forward<FuncT>(func), move(delta), identity(func.nDims));
}

template<typename FuncT>
auto makeAffineTransfomation(FuncT&& func, vect delta, matrix const& A)
{
    return AffineTransformation<decay_t<FuncT>>(forward<FuncT>(func), move(delta), A);
}

template<typename FuncT>
auto makeAffineTransfomation(FuncT&& func, matrix const& A)
{
    return AffineTransformation<decay_t<FuncT>>(forward<FuncT>(func), makeConstantVect(func.nDims, 0), A);
}

template<typename FuncT>
auto prepareForPolar(FuncT&& func, vect const& v)
{
    auto A = linearizationNormalization(func.hess(v));
    return makeAffineTransfomation(func, v, A);
}
