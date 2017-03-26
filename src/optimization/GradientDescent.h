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

                auto grad = func.grad(p0);
                auto val = func(p0);

                auto delta = mDeltaStrategy(iter, p0, val, grad);
                if (mStopStrategy(iter, p0, grad, delta))
                    break;

                p0 += delta;
            }

            return path;
        }

        DeltaStrategyT const& getDeltaStrategy() const
        {
            return mDeltaStrategy;
        }

        StopStrategyT const& getStopStrategy() const
        {
            return mStopStrategy;
        }

    private:
        DeltaStrategyT mDeltaStrategy;
        StopStrategyT mStopStrategy;
    };

    template<typename DeltaStrategyT, typename StopStrategyT>
    auto makeGradientDescent(DeltaStrategyT&& deltaStrategyT, StopStrategyT&& stopStrategyT)
    {
        return GradientDescent<decay_t<DeltaStrategyT>, decay_t<StopStrategyT>>(forward<DeltaStrategyT>(deltaStrategyT),
                                                                                forward<StopStrategyT>(stopStrategyT));
    };
}