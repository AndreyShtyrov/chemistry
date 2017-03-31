#pragma once

#include "helper.h"

#include "FunctionProducer.h"

template<typename Func1T, typename Func2T>
class Product : public FunctionProducer
{
public:
    Product(Func1T const& func1, Func2T const& func2) : mFunc1(func1), mFunc2(func2)
    { }

    virtual double operator()(vect const& x)
    {
        return mFunc1(x) * mFunc2(x);
    }

    virtual vect grad(vect const& x)
    {
        return mFunc1(x) * mFunc2.grad(x) + mFunc1.grad(x) * mFunc2(x);
    }

    virtual matrix hess(vect const& x)
    {
        return mFunc1(x) * mFunc2.hess(x) + 2 * mFunc2.grad(x) * mFunc1.grad(x).transpose() + mFunc2(x) * mFunc1.hess(x);
    };

private:
    Func1T mFunc1;
    Func2T mFunc2;
};

template<typename Func1T, typename Func2T>
typename enable_if<is_base_of<FunctionProducer, Func1T>::value && is_base_of<FunctionProducer, Func2T>::value, Product<Func1T, Func2T>>::type
operator*(Func1T&& func1, Func2T&& func2)
{
    return Product<Func1T, Func2T>(forward<Func1T>(func1), forward<Func2T>(func2));
};

