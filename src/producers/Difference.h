#pragma once

#include "helper.h"

#include "FunctionProducer.h"

template<typename Func1T, typename Func2T>
class Difference : public FunctionProducer
{
public:
    Difference(Func1T func1, Func2T func2) : FunctionProducer(func1.nDims), mFunc1(move(func1)), mFunc2(move(func2))
    {
        assert(mFunc1.nDims == mFunc2.nDims);
    }

    double operator()(vect const& x) override
    {
        return mFunc1(x) - mFunc2(x);
    }

    vect grad(vect const& x) override
    {
        return mFunc1.grad(x) - mFunc2.grad(x);
    }

    matrix hess(vect const& x) override
    {
        return mFunc1.hess(x) - mFunc2.hess(x);
    };

    tuple<double, vect> valueGrad(vect const& x) override
    {
        auto first  = mFunc1.valueGrad(x);
        auto second = mFunc2.valueGrad(x);

        return make_tuple(get<0>(first) - get<0>(second), get<1>(first) - get<1>(second));
    };

    tuple<double, vect, matrix> valueGradHess(vect const& x) override
    {
        auto first  = mFunc1.valueGradHess(x);
        auto second = mFunc2.valueGradHess(x);

        return make_tuple(get<0>(first) - get<0>(second), get<1>(first) - get<1>(second), get<2>(first) - get<2>(second));
    };

private:
    Func1T mFunc1;
    Func2T mFunc2;
};

template<typename Func1T, typename Func2T>
typename enable_if<is_base_of<FunctionProducer, decay_t<Func1T>>::value && is_base_of<FunctionProducer, decay_t<Func2T>>::value, Difference<decay_t<Func1T>, decay_t<Func2T>>>::type
operator-(Func1T&& func1, Func2T&& func2)
{
    return Difference<decay_t<Func1T>, decay_t<Func2T>>(forward<Func1T>(func1), forward<Func2T>(func2));
};
