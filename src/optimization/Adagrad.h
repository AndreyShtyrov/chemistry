#pragma once

#include "helper.h"

namespace optimization {
    template<int N>
    class Adagrad {
    public:
        constexpr static double EPS = 1e-7;
        constexpr static double E = 1e-8;

        Adagrad(double speed = 1.0) : mS(speed) {
            mG.setZero();
        }

        vector <vect<N>> operator()(FunctionProducer <N> &func, vect <N> p0) {
            history.clear();

            vector <vect<N>> path;

            for (int i = 0;; i++) {
                cerr << p0 << ' ';
                path.push_back(p0);
                history.push_back(func(p0));

                auto grad = func.grad(p0);
                if (grad.norm() < EPS)
                    break;

                double e = E;
                if (i)
                    p0 -= (1 / sqrt(mG + e) * grad.array()).matrix();
                mG += grad.array() * grad.array();
            }

            return path;
        }

        vector<double> history;

    private:
        double mS;
        double mM;
        Eigen::Array<double, N, 1> mG;
    };
}