#pragma once

#include "helper.h"

#include "FunctionProducer.h"
#include "linearAlgebraUtils.h"

template<typename FuncT>
class AffineTransformation : public FunctionProducer
{
public:
    AffineTransformation(FuncT func, vect delta, matrix basis) : FunctionProducer((size_t) basis.cols()),
                                                                 mFunc(move(func)), mDelta(move(delta)),
                                                                 mBasis(move(basis))
    { }

    double operator()(vect const& x) override
    {
        assert((size_t) x.rows() == nDims);

        return mFunc(transform(x));
    }

    vect grad(vect const& x) override
    {
        assert((size_t) x.rows() == nDims);

        return transformGrad(mFunc.grad(transform(x)));
    }

    matrix hess(vect const& x) override
    {
        assert((size_t) x.rows() == nDims);

        return transformHess(mFunc.hess(transform(x)));
    }

    tuple<double, vect> valueGrad(vect const& x) override
    {
        assert((size_t) x.rows() == nDims);

        auto result = mFunc.valueGrad(transform(x));
        return make_tuple(get<0>(result), transformGrad(get<1>(result)));
    };

    tuple<double, vect, matrix> valueGradHess(vect const& x) override
    {
        assert((size_t) x.rows() == nDims);

        auto result = mFunc.valueGradHess(transform(x));
        return make_tuple(get<0>(result), transformGrad(get<1>(result)), transformHess(get<2>(result)));
    };

    vect transform(vect const& x) const
    {
        assert((size_t) x.rows() == nDims);

        return mBasis * x + mDelta;
    }

    vect fullTransform(vect const& x) const
    {
        assert((size_t) x.size() == nDims);
        return mFunc.fullTransform(transform(x));
    }

    vect backTransform(vect const& x) const
    {
        assert(x.size() == mDelta.size());
        return mBasis.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(x - mDelta);
//        return mBasis.inverse() * (x - mDelta);
    }

    FuncT const& getInnerFunction() const
    {
        return mFunc;
    }

    auto& getFullInnerFunction()
    {
        return mFunc.getFullInnerFunction();
    }

    auto const& getFullInnerFunction() const
    {
        return mFunc.getFullInnerFunction();
    }

    matrix const& getBasis() const
    {
        return mBasis;
    }

private:
    FuncT mFunc;

    vect mDelta;
    matrix mBasis;

    vect transformGrad(vect const& grad)
    {
        return mBasis.transpose() * grad;
    }

    matrix transformHess(matrix const& hess)
    {
        return mBasis.transpose() * hess * mBasis;
    }
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
    return AffineTransformation<decay_t<FuncT>>(forward<FuncT>(func), makeConstantVect((size_t) A.rows(), 0), A);
}

template<typename FuncT>
auto normalizeForPolar(FuncT&& func, vect const& v)
{
    return makeAffineTransfomation(forward<FuncT>(func), v, linearizationNormalization(func.hess(v)));
}

template<typename FuncT>
auto toNormalCooridnates(FuncT&& func, vect const& v)
{
    auto A = linearization(func.hess(v));

    return makeAffineTransfomation(forward<FuncT>(func), v, linearizationNormalization(func.hess(v)));
}
