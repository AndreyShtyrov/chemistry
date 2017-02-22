#pragma once

#include "helper.h"

#include "FunctionProducer.h"

template<typename FuncT, typename ConstraintT>
class LagrangeMultiplier : public FunctionProducer<FuncT::N + 1>
{
public:
    using FunctionProducer<FuncT::N + 1>::N;

    template<typename FuncArgT, typename ConstraintArgT>
    LagrangeMultiplier(FuncArgT&& func, ConstraintArgT&& constraint)
            : mFunc(forward<FuncArgT>(func))
            , mConstraint(forward<ConstraintArgT>(constraint))
    { };

    virtual double operator()(vect<N> const& x)
    {
        vect<N - 1> p = x.head(N - 1);
        return mFunc(p) + x(N - 1) * mConstraint(p);
    }

    virtual vect<N> grad(vect<N> const& x)
    {
        vect<N - 1> p = x.head(N - 1);
        vect<N - 1> grad = mFunc.grad(p) + x(N - 1) * mConstraint.grad(p);
        auto lgrad = mConstraint(p);

        vect<N> result;
        result << grad, lgrad;
        return result;
    }

    virtual matrix<N, N> hess(vect<N> const& x)
    {
        vect<N - 1> p = x.head(N - 1);

        matrix<N, N> result;
        matrix<N - 1, N - 1> hess = mFunc.hess(p) + x(N - 1) * mConstraint.hess(p);
        vect<N - 1> grad = mConstraint.grad(p);

        result << hess, grad, grad.transpose(), 0;
        return result;
    };

private:
    FuncT mFunc;
    ConstraintT mConstraint;
};

template<typename FuncT, typename ConstraintT, typename FT=typename remove_reference<FuncT>::type, typename CT=typename remove_reference<ConstraintT>::type>
LagrangeMultiplier<FT, CT> make_lagrange(FuncT&& func, ConstraintT&& constraint)
{
    return LagrangeMultiplier<FT, CT>(forward<FuncT>(func), forward<ConstraintT>(constraint));
};