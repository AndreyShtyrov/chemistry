#pragma once

#include "helper.h"

#include <boost/type_index.hpp>
#include <typeinfo>

namespace optimization
{
    template<typename DeltaStrategyT, typename StopStrategyT>
    class GradientDescent
    {
    public:
        static constexpr int N = DeltaStrategyT::N;

        GradientDescent(DeltaStrategyT deltaStrategy, StopStrategyT stopStrategy) : mDeltaStrategy(move(deltaStrategy)),
                                                                                    mStopStrategy(move(stopStrategy))
        {}

        template<typename FuncT>
        vector<vect<N>> operator()(FuncT& func, vect<N> p0)
        {
            vector<vect<N>> path;

            for (size_t iter = 0;; iter++) {
                path.push_back(p0);

                auto val = func(p0);
                auto grad = func.grad(p0);

                auto delta = mDeltaStrategy(iter, p0, val, grad);
                if (mStopStrategy(iter, p0, grad, delta))
                    break;

                p0 += delta;
            }

            return path;
        }

    private:
        DeltaStrategyT mDeltaStrategy;
        StopStrategyT mStopStrategy;
    };

    template<typename DeltaStrategyT, typename StopStrategyT>
    auto make_gradient_descent(DeltaStrategyT&& deltaStrategyT, StopStrategyT&& stopStrategyT)
    {
        return GradientDescent<decay_t<DeltaStrategyT>, decay_t<StopStrategyT>>(forward<DeltaStrategyT>(deltaStrategyT),
                                                                                forward<StopStrategyT>(stopStrategyT));
    };

    template<int N_DIMS>
    class FollowGradientDeltaStrategy
    {
    public:
        static constexpr int N = N_DIMS;

        vect<N> operator()(size_t iter, vect<N> const& p, double value, vect<N> const& grad)
        {
            return -grad;
        }
    };


    template<int N_DIMS>
    class GradientLengthStopStrategy
    {
    public:
        static constexpr int N = N_DIMS;

        GradientLengthStopStrategy(double eps) : mEps(eps)
        {}

        bool operator()(size_t iter, vect<N> const& p, vect<N> const& grad, vect<N> const& delta)
        {
            return grad.norm() < mEps;
        }

    private:
        double mEps;
    };

    template<int N_DIMS>
    class DeltaNormStopStrategy
    {
    public:
        static constexpr int N = N_DIMS;

        DeltaNormStopStrategy(double eps) : mEps(eps)
        {}

        bool operator()(size_t iter, vect<N> const& p, vect<N> const& grad, vect<N> const& delta)
        {
            return delta.norm() < mEps;
        }

    private:
        double mEps;
    };

    template<int N_DIMS, typename... FuncsT>
    class AtomicStopStrategy
    {
    public:
        static constexpr int N = N_DIMS;

        AtomicStopStrategy(double maxAtomDelta, double rmsAtomDelta, FuncsT... funcs) : mMaxAtomDelta(maxAtomDelta),
                                                                                        mRmsAtomDelta(rmsAtomDelta),
                                                                                        mFuncs(move(funcs)...)
        {}

        bool operator()(size_t iter, vect<N> const& p, vect<N> const& grad, vect<N> const& delta)
        {
            auto x0 = back_transformation<0>(p);
            auto x1 = back_transformation<0>(vect<N>(p + delta));
            auto atomDeltas = x1 - x0;

            double atomDeltaSum = 0;
            double atomDeltaMax = 0;
            size_t atomCnt = (size_t) atomDeltas.rows() / 3;

            for (size_t i = 0; i < atomCnt; i++) {
                double atomDelta = atomDeltas.block(i * 3, 0, 3, 1).norm();
                if (atomDelta >= mMaxAtomDelta)
                    return false;

                atomDeltaSum += sqr(atomDelta);
                atomDeltaMax = max(atomDeltaMax, atomDelta);
            }

            cerr << atomDeltaSum << ' ' << atomDeltaMax << endl;

            return sqrt(atomDeltaSum / atomCnt) < mRmsAtomDelta;
        }

    public:
        template<int I, int CUR_N>
        auto back_transformation(vect<CUR_N> const& v, enable_if_t<I < sizeof...(FuncsT), int> _ = 0)
        {
            return back_transformation<I + 1>(get<I>(mFuncs).transform(v));
        }

        template<int I, int CUR_N>
        auto back_transformation(vect<CUR_N> const& v, enable_if_t<I == sizeof...(FuncsT), int> _ = 0)
        {
            return v;
        }

        double mMaxAtomDelta;
        double mRmsAtomDelta;
        tuple<FuncsT...> mFuncs;
    };

    template<typename... FuncT>
    auto make_atomic_stop_strategy(double maxAtomDelta, double rmsAtomDelta, FuncT&& ... funcs)
    {
        return AtomicStopStrategy<first_type_t<decay_t<FuncT>...>::N, decay_t<FuncT>...>(maxAtomDelta, rmsAtomDelta,
                                                                                         forward<FuncT>(funcs)...);
    }
}