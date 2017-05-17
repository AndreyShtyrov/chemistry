#pragma once

#include "helper.h"

namespace optimization {
    struct DFP {
        void update(matrix &B, vect const& dx, vect const& dy) {
            int n = B.rows();
            double denom = (dy.transpose() * dx);

            B = (identity(n, n) - dy * dx.transpose() / denom) * B *
                (identity(n, n) - dx * dy.transpose() / denom) + dy * dy.transpose() / denom;
        }
    };

    struct BFGS {
        void update(matrix &B, vect const& dx, vect const& dy) {
            B = B + dy * dy.transpose() / (dy.transpose() * dx) - B * dx * (B * dx).transpose() / (dx.transpose() * B * dx);
        }
    };

    struct Broyden {
        void update(matrix &B, vect const& dx, vect const& dy) {
            B = B + (dy - B * dx) / (dx.transpose() * dx) * dx.transpose();
        }
    };

    template<typename UpdaterT>
    class QuasiNewtonDeltaStrategy {
    public:
        QuasiNewtonDeltaStrategy()
        {
            mB.setIdentity();
        }

        QuasiNewtonDeltaStrategy(matrix hess) : mB(move(hess))
        {
            out.precision(15);
        }

        matrix makeGood(matrix const& m)
        {
            matrix A = linearization(m);
            matrix L = A.transpose() * m * A;
            for (size_t i = 0; i < (size_t) L.rows(); i++)
                L(i, i) = max(abs(L(i, i)), .01);
            auto Ai = A.inverse();

            return Ai.transpose() * L * Ai;
        };

        vect operator()(size_t iter, vect const &p, double value, vect const &grad) {
            if (iter)
                updater.update(mB, p - mLastP, grad - mLastGrad);

            LOG_INFO("B matrix values: {}", Eigen::JacobiSVD<matrix>(mB).singularValues().transpose());

            mLastP = p;
            mLastGrad = grad;

            if (iter < 4)
                return -grad;
            else
                return -mB.inverse() * grad;
        }

        void initializeHessian(matrix hess)
        {
            mB = move(hess);
        }

    private:
        matrix mB;
//        matrix mH;
        vect mLastDelta;
        vect mLastGrad;
        vect mLastP;
        UpdaterT updater;
    };
}