#pragma once

#include "FunctionProducer.h"
#include "History.h"

namespace optimization
{
    template<int N>
    class MomentumGradientDescent
    {
    public:
        constexpr static double EPS = 1e-7;

        MomentumGradientDescent(double speed = 1.0, double momentum = .9) : mS(speed), mM(momentum)
        {}

        vector<vect<N>> operator()(FunctionProducer<N> &func, vect<N> p0)
        {
            history.clear();

            vector<vect<N>> path;

            vect<N> g;
            g.setZero();

            for (int i = 0;; i++)
            {
                cerr << p0.transpose() << " : " << endl;
                for (auto val : history.vals)
                {
                    cerr << val << ' ';
                }
                cerr << endl;

                path.push_back(p0);
                history.push(func(p0), func.grad(p0), func.hess(p0));

                auto grad = func.grad(p0);
                if (grad.norm() < EPS || i > 100)
                    break;

                g = mM * g + mS * grad;
                p0 -= g;
            }

            return path;
        }

        History <N> history;

    private:
        double mS;
        double mM;
    };

}