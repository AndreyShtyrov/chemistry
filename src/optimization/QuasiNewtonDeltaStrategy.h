#pragma once

#include "helper.h"

namespace optimization {
    struct DFP {
        template<int N>
        void update(matrix<N, N> &B, vect<N> const& dx, vect<N> const& dy) {
            double denom = (dy.transpose() * dx);

            B = (identity<N>() - dy * dx.transpose() / denom) * B *
                (identity<N>() - dx * dy.transpose() / denom) + dy * dy.transpose() / denom;
        }
    };

    struct BFGS {
        template<int N>
        void update(matrix<N, N> &B, vect<N> const& dx, vect<N> const& dy) {
            B = B + dy * dy.transpose() / (dy.transpose() * dx) - B * dx * (B * dx).transpose() / (dx.transpose() * B * dx);
        }
    };

    struct Broyden {
        template<int N>
        void update(matrix<N, N> &B, vect<N> const& dx, vect<N> const& dy) {
            B = B + (dy - B * dx) / (dx.transpose() * dx) * dx.transpose();
        }
    };

    template<int N_DIMS, typename UpdaterT>
    class QuasiNewtonDeltaStrategy {
    public:
        static constexpr int N = N_DIMS;

        QuasiNewtonDeltaStrategy()
        {
            mB.setIdentity();
        }

        QuasiNewtonDeltaStrategy(matrix<N, N> hess) : mB(move(hess))
        {
            out.precision(15);
        }

        matrix<N, N> makeGood(matrix<N, N> const& m)
        {
            matrix<N, N> A = linearization(m);
            matrix<N, N> L = A.transpose() * m * A;
            for (size_t i = 0; i < N; i++)
//                L(i, i) = max(abs(L(i, i)), .5);
                L(i, i) = max(abs(L(i, i)), .1);
            auto Ai = A.inverse();

            return Ai.transpose() * L * Ai;
        };

        vect<N> operator()(size_t iter, vect<N> const &p, double value, vect<N> const &grad) {
            if (iter)
                updater.template update<N>(mB, mLastDelta, grad - mLastGrad);

//            int const N = 3;

//            if ((int) iter > N)
//            linearization(mB);
            mLastDelta = -makeGood(mB).inverse() * grad;
//            mLastDelta = -mSpeed * mB.inverse() * grad;
//            mLastDelta = -mSpeed * mB * grad;
//            else
//                mLastDelta = -mSpeed * grad;
            mLastGrad = grad;

            out << p(0) << ' ' << p(1) << endl;

            return mLastDelta;
        }

        void initializeHessian(matrix<N, N> hess)
        {
            mB = move(hess);
        }

    private:
        matrix<N, N> mB;
//        matrix<N, N> mH;
        vect<N> mLastDelta;
        vect<N> mLastGrad;
        UpdaterT updater;
    };
}