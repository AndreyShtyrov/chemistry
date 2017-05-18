#pragma once

#include "helper.h"

#include <boost/type_index.hpp>
#include <typeinfo>

namespace optimization
{
    template<typename DeltaStrategyT, typename StopStrategyT>
    class SecondOrderGradientDescent
    {
    public:
        static constexpr int N = DeltaStrategyT::N;

        SecondOrderGradientDescent(DeltaStrategyT deltaStrategy, StopStrategyT stopStrategy) : mDeltaStrategy(
           move(deltaStrategy)), mStopStrategy(move(stopStrategy))
        {}

        template<typename FuncT>
        vector<vect> operator()(FuncT& func, vect p)
        {
            vector<vect> path;

            for (size_t iter = 0;; iter++) {
                path.push_back(p);

                matrix hess;
                vect grad;
                double val;

                try {
                    hess = func.hess(p);
                    grad = func.grad(p);
                    val = func(p);
                }
                catch (GaussianException const& exc) {
                    LOG_ERROR("Gaussian Exception: {}", exc.what());
                    val = 0;
                    grad.setZero();
                    hess.setZero();
                }

                auto delta = mDeltaStrategy(iter, p, val, grad, hess);
                if (mStopStrategy(iter, p, val, grad, hess, delta))
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
    auto makeSecondGradientDescent(DeltaStrategyT&& deltaStrategyT, StopStrategyT&& stopStrategyT)
    {
        return SecondOrderGradientDescent<decay_t<DeltaStrategyT>, decay_t<StopStrategyT>>(
           forward<DeltaStrategyT>(deltaStrategyT), forward<StopStrategyT>(stopStrategyT));
    };
}