#pragma once

#include "helper.h"

namespace optimization
{
    template<int N>
    class Adam
    {
    public:
        constexpr static double EPS = 1e-7;

        explicit Adam(double speed=2., double beta1=.9, double beta2=0.999, double eps=1e-8)
           : mSpeed(speed), mBeta1(beta1), mBeta2(beta2), mEps(eps)
        {
            mMean.setZero();
            mStd2.setZero();
        }

        vector<vect<N>> operator()(FunctionProducer<N>& func, vect<N> p0)
        {
            history.clear();

            vector<vect<N>> path;

            cerr << endl;
            for (int i = 0; ; i++) {
//            cerr << p0 << ' ' << mMean << ' ' << mStd2 << endl;
                path.push_back(p0);
                history.push_back(func(p0));

                auto grad = func.grad(p0);
                if (grad.norm() < EPS)
                    break;

                if (!i)
                    mMean = grad.array(), mStd2 = grad.array() * grad.array();
                mMean = mBeta1 * mMean + (1 - mBeta1) * grad.array();
                mStd2 = mBeta2 * mStd2 + (1 - mBeta2) * grad.array() * grad.array();
                p0 -= (mMean / (sqrt(mStd2) + mEps) * grad.array()).matrix();
            }

            return path;
        }

        vector<double> history;

    private:
        double mSpeed;
        double mBeta1;
        double mBeta2;
        double mEps;
        Eigen::Array<double, N, 1> mMean;
        Eigen::Array<double, N, 1>  mStd2;
    };
}