#pragma once

#include "helper.h"

namespace optimization
{
    template<int N_DIMS>
    class HessianDeltaStrategy
    {
    public:
        static constexpr int N = N_DIMS;

        vect<N> operator()(size_t iter, vect<N> const& p, double value, vect<N> const& grad, matrix<N, N> const& hess)
        {
            return -hess.inverse() * grad;
        }
    };
}