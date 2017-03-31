#pragma once

#include "helper.h"

namespace optimization
{
    template<int N_DIMS>
    class HessianDeltaStrategy
    {
    public:
        static constexpr int N = N_DIMS;

        vect operator()(size_t iter, vect const& p, double value, vect const& grad, matrix const& hess)
        {
            return -hess.inverse() * grad;
        }
    };
}