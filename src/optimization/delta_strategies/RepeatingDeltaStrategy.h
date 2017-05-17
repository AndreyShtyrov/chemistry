#pragma once

#include "helper.h"

namespace optimization
{
    template<typename DeltaStrategyT>
    class RepeatingDeltaStrategy
    {
    public:
        RepeatingDeltaStrategy(DeltaStrategyT deltaStrategy) : mDeltaStrategy(move(deltaStrategy))
        { }

        template<typename... ArgsT>
        vect operator()(size_t iter, vect const& p, double value, ArgsT const&... args)
        {
            if (!iter || value < mLastValue) {
                mLastP = p;
                mLastValue = value;
                mLastDelta = mDeltaStrategy(iter, p, value, args...);
                return mLastDelta;
            }
            else {
                LOG_INFO("bad value change (from {} to {}). Trying next delta ({} delta norm)", mLastValue, value, mLastDelta.norm());
                mLastDelta = mLastP - p + 0.5 * mLastDelta;
                return mLastDelta;
            }
        }

//        vect operator()(size_t iter, vect const& p, double value, vect const& grad)
//        {
//            if (!iter || value < mLastValue) {
//                mLastP = p;
//                mLastValue = value;
//                mLastDelta = mDeltaStrategy(iter, p, value, grad);
//                return mLastDelta;
//            }
//            else {
//                LOG_INFO("bad value change (from {} to {}). Trying next delta ({} delta norm)", mLastValue, value, mLastDelta.norm());
//                mLastDelta = mLastP - p + 0.5 * mLastDelta;
//                return mLastDelta;
//            }
//        }

    private:
        vect mLastP;
        vect mLastDelta;
        double mLastValue;
        DeltaStrategyT mDeltaStrategy;
    };

    template<typename DeltaStrategyT>
    auto makeRepeatDeltaStrategy(DeltaStrategyT&& deltaStrategy)
    {
        return RepeatingDeltaStrategy<decay_t<DeltaStrategyT>>(forward<DeltaStrategyT>(deltaStrategy));
    }
}