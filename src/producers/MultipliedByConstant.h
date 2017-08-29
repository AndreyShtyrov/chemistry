#pragma once

#include "helper.h"

#include "FunctionProducer.h"

template<typename FuncT>
class MultipliedByConstant : public FunctionProducer
{
public:
    MultipliedByConstant(FuncT func, double factor) : FunctionProducer(func.nDims), mFunc(func), mFactor(factor)
    { }

    double operator()(vect const& x) override
    {
        return mFactor * mFunc(x);
    }

    vect grad(vect const& x) override
    {
        return mFactor * mFunc.grad(x);
    }

    matrix hess(vect const& x) override
    {
        return mFactor * mFunc.hess(x);
    };

    tuple<double, vect> valueGrad(vect const& x) override
    {
        auto valueGrad = mFunc.valueGrad(x);
        return make_tuple(mFactor * get<0>(valueGrad), mFactor * get<1>(valueGrad));
    };

    tuple<double, vect, matrix> valueGradHess(vect const& x) override
    {
        auto valueGradHess = mFunc.valueGradHess(x);
        return make_tuple(mFactor * get<0>(valueGradHess), mFactor * get<1>(valueGradHess), mFactor * get<2>(valueGradHess));
    };

private:
    FuncT mFunc;
    double mFactor;
};

template<typename FuncT>
enable_if_t<is_base_of<FunctionProducer, decay_t<FuncT>>::value, MultipliedByConstant<decay_t<FuncT>>>
operator*(FuncT&& func, double value)
{
    return MultipliedByConstant<decay_t<FuncT>>(forward<FuncT>(func), value);
};

template<typename FuncT>
enable_if_t<is_base_of<FunctionProducer, decay_t<FuncT>>::value, MultipliedByConstant<decay_t<FuncT>>>
operator*(double value, FuncT&& func)
{
    return MultipliedByConstant<decay_t<FuncT>>(forward<FuncT>(func), value);
};