#pragma once

#include "helper.h"

namespace optimization
{
    class FollowGradDeltaStrategy
    {
    public:
        FollowGradDeltaStrategy(double speed = 1.) : mSpeed(speed)
        {}

        vect operator()(size_t iter, vect const& p, double value, vect const& grad)
        {
            return -mSpeed * grad;
        }

    private:
        double mSpeed;
    };
}