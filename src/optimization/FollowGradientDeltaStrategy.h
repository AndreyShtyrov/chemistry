#pragma once

#include "helper.h"

namespace optimization
{
    template<int N_DIMS>
    class FollowGradientDeltaStrategy
    {
    public:
        static constexpr int N = N_DIMS;

        FollowGradientDeltaStrategy(double speed = 1.) : mSpeed(speed)
        {}

        vect<N> operator()(size_t iter, vect<N> const& p, double value, vect<N> const& grad)
        {
            return -mSpeed * grad;
        }

    private:
        double mSpeed;
    };
}