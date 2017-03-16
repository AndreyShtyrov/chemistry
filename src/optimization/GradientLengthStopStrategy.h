#pragma once

#include "helper.h"

namespace optimization
{
    template<int N_DIMS>
    class FollowGradientDeltaStrategy
    {
    public:
        static constexpr int N = N_DIMS;

        vect<N> operator()(size_t iter, vect<N> const& p, double value, vect<N> const& grad)
        {
            return -grad;
        }
    };
}