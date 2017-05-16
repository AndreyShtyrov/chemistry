#pragma once

#include "helper.h"

namespace optimization
{
    template<int N_DIMS>
    class GradientLengthStopStrategy
    {
    public:
        static constexpr int N = N_DIMS;

        GradientLengthStopStrategy(double eps) : mEps(eps)
        {}

        bool operator()(size_t iter, vect const& p, vect const& grad, vect const& delta)
        {
            return grad.norm() < mEps;
        }

    private:
        double mEps;
    };
}