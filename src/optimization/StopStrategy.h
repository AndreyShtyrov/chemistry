#pragma once

#include "helper.h"

namespace optimization
{
    class StopStrategy
    {
    public:
        StopStrategy(double gradThreshold, double deltaThreshold) : mGradThreshold(gradThreshold), mDeltaThreshold(deltaThreshold)
        { }

        template<int N>
        bool operator()(size_t iter, vect<N> const &p, vect<N> const &grad, vect<N> const &delta) {
            return /*iter == 1 || */(grad.norm() < mGradThreshold && delta.norm() < mDeltaThreshold);
        }

    private:
        double mGradThreshold;
        double mDeltaThreshold;
    };
}