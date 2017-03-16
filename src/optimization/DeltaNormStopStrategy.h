#pragma once

#include "helper.h"

namespace optimization
{
    template<int N_DIMS>
    class DeltaNormStopStrategy
    {
    public:
        static constexpr int N = N_DIMS;

        DeltaNormStopStrategy(double eps) : mEps(eps)
        {}

        bool operator()(size_t iter, vect<N> const& p, vect<N> const& grad, vect<N> const& delta)
        {
            return delta.norm() < mEps;
        }

    private:
        double mEps;
    };
}