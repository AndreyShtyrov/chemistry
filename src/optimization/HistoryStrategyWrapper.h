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

        vect<N> operator()(size_t iter, vect<N> const& p, double value, vect<N> const& grad)
        {
            cerr << p.transpose() << endl;
            mValues.push_back(value);
            return mDeltaStrategy(iter, p, value, grad);
        }

        vect<N> operator()(size_t iter, vect<N> const& p, double value, vect<N> const& grad, matrix<N, N> const& hess)
        {
            cerr << value << endl;
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
    auto make_history_strategy(DeltaStrategyT&& deltaStrategy)
    {
        return HistoryStrategyWrapper<decay_t<DeltaStrategyT>>(forward<DeltaStrategyT>(deltaStrategy));
    }
}
