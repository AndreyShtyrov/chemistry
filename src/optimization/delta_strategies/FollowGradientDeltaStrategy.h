#pragma once

#include "helper.h"

namespace optimization
{
    class FollowGradientDeltaStrategy
    {
    public:
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