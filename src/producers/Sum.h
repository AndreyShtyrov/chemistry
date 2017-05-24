#pragma once

#include "helper.h"

#include "FunctionProducer.h"

template<typename Func1T, typename Func2T>
class Sum : public FunctionProducer
{
public:
    Sum(Func1T const& func1, Func2T const& func2) : FunctionProducer(func1.nDims), mFunc1(func1), mFunc2(func2)
    {
        assert(func1.nDims == func2.nDims);
    }

    virtual double operator()(vect const& x)
    {
        return mFunc1(x) + mFunc2(x);
    }

    virtual vect grad(vect const& x)
    {
        return mFunc1.grad(x) + mFunc2.grad(x);
    }

    virtual matrix hess(vect const& x)
    {
        return mFunc1.hess(x) + mFunc2.hess(x);
    };

private:
    Func1T mFunc1;
    Func2T mFunc2;
};

template<typename Func1T, typename Func2T>
typename enable_if<is_base_of<FunctionProducer, Func1T>::value && is_base_of<FunctionProducer, Func2T>::value, Sum<Func1T, Func2T>>::type
operator+(Func1T&& func1, Func2T&& func2)
{
    return Sum<Func1T, Func2T>(forward<Func1T>(func1), forward<Func2T>(func2));
};
