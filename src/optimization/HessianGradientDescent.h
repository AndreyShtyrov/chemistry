#pragma once

#include "helper.h"

namespace optimization
{
    template<int N>
    class HessianGradientDescent
    {
    public:
        constexpr static double EPS = 1e-7;

        HessianGradientDescent(double speed=1.) : mSpeed(speed)
        { }

        vector<vect<N>> operator()(FunctionProducer<N>& func, vect<N> p0)
        {
            history.clear();

            vector<vect<N>> path;

            for (int i = 0; ; i++) {
                path.push_back(p0);
                history.push(p0, func(p0), func.grad(p0), func.hess(p0));

                if (p0.norm() > 100000)
                    break;

                auto grad = func.grad(p0);
                auto hess = func.hess(p0);

                if (grad.norm() < EPS)
                    break;
                p0 -= mSpeed * hess.inverse() * grad;
            }

            return path;
        }

        History<N> history;

    private:
        double mSpeed;
    };
}