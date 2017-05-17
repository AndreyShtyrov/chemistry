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
        GradientDescent(DeltaStrategyT deltaStrategy, StopStrategyT stopStrategy) : mDeltaStrategy(move(deltaStrategy)),
                                                                                    mStopStrategy(move(stopStrategy))
        {}

        template<typename FuncT>
        vector<vect> operator()(FuncT& func, vect p)
        {
            vector<vect> path;

            for (size_t iter = 0;; iter++) {
                path.push_back(p);

                vect grad;
                double val = 0;

                try {
                    grad = func.grad(p);
                    val = func(p);
                }
                catch (GaussianException const& exc) {
                    LOG_ERROR("Gaussian Exception: {}", exc.what());
                    grad.setZero();
                    val = 0;
                }

                auto delta = mDeltaStrategy(iter, p, val, grad);
                if (mStopStrategy(iter, p, val, grad, delta))
                    break;

                p += delta;
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