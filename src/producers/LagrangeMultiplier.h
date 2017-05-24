#pragma once

#include "helper.h"

#include "FunctionProducer.h"

template<typename FuncT, typename ConstraintT>
class LagrangeMultiplier : public FunctionProducer
{
public:
    template<typename FuncArgT, typename ConstraintArgT>
    LagrangeMultiplier(FuncArgT&& func, ConstraintArgT&& constraint)
            : FunctionProducer(func.nDims + 1), mFunc(forward<FuncArgT>(func))
            , mConstraint(forward<ConstraintArgT>(constraint))
    { };

    virtual double operator()(vect const& x)
    {
        vect p = x.head(nDims - 1);
        return mFunc(p) + x(nDims - 1) * mConstraint(p);
    }

    virtual vect grad(vect const& x)
    {
        vect p = x.head(nDims - 1);
        vect grad = mFunc.grad(p) + x(nDims - 1) * mConstraint.grad(p);
        auto lgrad = mConstraint(p);

        vect result(nDims);
        result << grad, lgrad;
        return result;
    }

    virtual matrix hess(vect const& x)
    {
        vect p = x.head(nDims - 1);

        Eigen::Matrix3d result(nDims, nDims);
        matrix hess = mFunc.hess(p) + x(nDims - 1) * mConstraint.hess(p);
        vect grad = mConstraint.grad(p);

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