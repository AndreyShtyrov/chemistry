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

        vect operator()(size_t iter, vect const& p, double value, vect const& grad)
        {
            return -mSpeed * grad;
        }

    private:
        double mSpeed;
    };
}