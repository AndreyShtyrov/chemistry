#pragma once

#include "helper.h"

namespace optimization
{
    template<int N>
    class Adadelta
    {
    public:
        constexpr static double EPS = 1e-7;
        constexpr static double E = 1e-9;

        Adadelta(double decay=.99) : mD(decay)
        {
            mMeanGrad.setZero();
            mMeanDelta.setZero();
        }

        vector<vect<N>> operator()(FunctionProducer<N>& func, vect p0)
        {
            history.clear();

            vector<vect<N>> path;

            cerr << endl;
            for (int i = 0; ; i++) {
                cerr << p0 << endl;
                path.push_back(p0);
                history.push_back(func(p0));

                auto grad = func.grad(p0);
                if (grad.norm() < EPS)
                    break;

                if (!i)
                    mMeanGrad = grad.array() * grad.array();
                mMeanGrad = mD * mMeanGrad + (1 - mD) * grad.array() * grad.array();
                cerr << (grad.array() * grad.array()) << ' ' << mMeanGrad << ' ' << mMeanDelta << endl;
                Eigen::Array<double, N, 1> delta = sqrt(mMeanDelta + (double) E) / sqrt(mMeanGrad + (double) E) * grad.array();
                p0 -= delta.matrix();
                if (!i)
                    mMeanDelta = sqr(delta);
                mMeanDelta = mD * mMeanDelta + (1 - mD) * sqr(delta);
            }

            return path;
        }

        vector<double> history;

    private:
        double mD;
        Eigen::Array<double, N, 1> mMeanGrad;
        Eigen::Array<double, N, 1>  mMeanDelta;
    };
}