#pragma once

namespace optimization
{
    template<int N>
    class NesterovGradientDescent
    {
    public:
        constexpr static double EPS = 1e-7;

        NesterovGradientDescent(double speed=0.5, double momentum=.9) : mS(speed), mM(momentum)
        { }

        vector<vect<N>> operator()(FunctionProducer<N>& func, vect p0)
        {
            history.clear();

            vector<vect<N>> path;

            vect g;
            g.setZero();

            for (int i = 0; ; i++) {
                path.push_back(p0);
                history.push(p0, func(p0), func.grad(p0), func.hess(p0));

                auto grad = func.grad(p0 - mM * g);
                if (grad.norm() < EPS || i > 100)
                    break;

                g = mM * g + mS * grad;
                p0 -= g;
            }

            return path;
        }

        History<N> history;

    private:
        double mS;
        double mM;
    };
}