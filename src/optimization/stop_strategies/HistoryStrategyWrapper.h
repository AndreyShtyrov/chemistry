#pragma once

#include "helper.h"

namespace optimization
{
    template<typename DeltaStrategyT>
    class HistoryStrategyWrapper
    {
    public:
        static constexpr int N = DeltaStrategyT::N;

        HistoryStrategyWrapper(DeltaStrategyT deltaStrategy) : mDeltaStrategy(move(deltaStrategy))
        {}

        vect operator()(size_t iter, vect const& p, double value, vect const& grad)
        {
            LOG_INFO("Delta strategy iteration:\n\tpoint: {}\n\tgrad: {}\n", p.transpose(), grad.transpose());
            mValues.push_back(value);
            return mDeltaStrategy(iter, p, value, grad);
        }

        vect operator()(size_t iter, vect const& p, double value, vect const& grad, matrix const& hess)
        {
            LOG_INFO("Delta strategy iteration:\n\tpoint: {}\n\tgrad: {}\n\thess values: {}", p.transpose(),
                     grad.transpose(), Eigen::JacobiSVD<matrix>(hess).singularValues().transpose());
            mValues.push_back(value);
            return mDeltaStrategy(iter, p, value, grad, hess);
        }

        vector<double> const& getValues() const
        {
            return mValues;
        }

    private:
        DeltaStrategyT mDeltaStrategy;
        vector<double> mValues;
    };

    template<typename DeltaStrategyT>
    auto makeHistoryStrategy(DeltaStrategyT&& deltaStrategy)
    {
        return HistoryStrategyWrapper<decay_t<DeltaStrategyT>>(forward<DeltaStrategyT>(deltaStrategy));
    }
}
