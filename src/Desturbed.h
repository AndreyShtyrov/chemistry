#pragma once

#include "helper.h"

#include "FunctionProducer.h"

template<typename FuncT>
class Desturbed : public FunctionProducer
{
public:
    Desturbed(FuncT func) : FunctionProducer(func.nDims), mFunc(move(func))
    { }

    virtual double operator()(vect const& x)
    {
        return mFunc(transformCoordinates(x));
    }

    virtual vect grad(vect const& x)
    {
        auto grad = mFunc.grad(transformCoordinates(x));
        grad(0) += grad(1) * cos(x(0));
        return grad;
    }

    virtual matrix hess(vect const& x)
    {
        auto p = transformCoordinates(x);
        auto grad = mFunc.grad(p);
        auto hess = mFunc.hess(p);
        auto c = cos(x(0));

        hess(0, 0) = hess(0, 0) + 2 * c * hess(0, 1) + sqr(c) * hess(1, 1) - grad(1) * sin(x(0));
        for (size_t i = 1; i < nDims; i++)
            hess(0, i) = hess(i, 0) = hess(0, i) + c * hess(1, i);

        return hess;
    };

private:
    FuncT mFunc;

    vect transformCoordinates(vect x)
    {
        x(1) += sin(x(0));
        return x;
    }
};

template<typename FuncT>
Desturbed<FuncT> make_desturbed(FuncT const& func)
{
    return Desturbed<FuncT>(func);
}
