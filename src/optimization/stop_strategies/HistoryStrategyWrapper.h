#pragma once

#include "helper.h"

namespace optimization
{
    template<typename StopStrategyT>
    class HistoryStrategyWrapper
    {
    public:
        static constexpr int N = StopStrategyT::N;

        HistoryStrategyWrapper(StopStrategyT stopStrategy) : mStopStrategy(move(stopStrategy))
        {}

        bool operator()(size_t iter, vect const& p, double value, vect const& grad, vect const& delta)
        {
            LOG_DEBUG("Delta strategy iteration:\n\titeration: {}\n\tvalue: {}\n\tpoint: {}\n\tgrad: {} [{}]\n\tdelta: {} [{}]\n",
                     iter, value, p.transpose(), grad.norm(), grad.transpose(), delta.norm(), delta.transpose());
            mGrads.push_back(grad);

            return mStopStrategy(iter, p, value, grad, delta);
        }

        bool
        operator()(size_t iter, vect const& p, double value, vect const& grad, matrix const& hess, vect const& delta)
        {
            LOG_DEBUG(
               "Delta strategy iteration:\n\titeration: {}\n\tvalue: {:.13f}\n\tpoint: {}\n\tgrad: {} [{}]\n\tdelta: {} [{}]\n\thess values: {}\n",
               iter, value, p.transpose(), grad.norm(), grad.transpose(), delta.norm(), delta.transpose(),
               singularValues(hess));

            mValues.push_back(value);
            mGrads.push_back(grad);

            return mStopStrategy(iter, p, value, grad, hess, delta);
        }

//        vect operator()(size_t iter, vect const& p, double value, vect const& grad, matrix const& hess)
//        {
//            LOG_INFO("Delta strategy iteration:\n\titeration: {}\n\tpoint: {}\n\tgrad: {}\n\thess values: {}", iter,
//                     p.transpose(), grad.transpose(), Eigen::JacobiSVD<matrix>(hess).singularValues().transpose());
//            mValues.push_back(value);you
//            mGrads.push_back(grad);
//
//            return mStopStrategy(iter, p, value, grad, hess);
//        }
//
//        vect operator()(size_t iter, vect const& p, double value, vect const& grad)
//        {
//            LOG_INFO("Delta strategy iteration:\n\titeration: {}\n\tpoint: {}\n\tgrad: {}\n", iter, p.transpose(),
//                     grad.transpose());
//            mValues.push_back(value);
//            mGrads.push_back(grad);
//
//            return mStopStrategy(iter, p, value, grad);
//        }

        vector<double> const& getValues() const
        {
            return mValues;
        }

        vector<vect> const& getGrads() const
        {
            return mGrads;
        }

    private:
        StopStrategyT mStopStrategy;
        vector<double> mValues;
        vector<vect> mGrads;
    };

    template<typename StopStrategyT>
    auto makeHistoryStrategy(StopStrategyT&& deltaStrategy)
    {
        return HistoryStrategyWrapper<decay_t<StopStrategyT>>(forward<StopStrategyT>(deltaStrategy));
    }
}
