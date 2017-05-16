#pragma once

#include "helper.h"

namespace optimization
{
    class HessianDeltaStrategy
    {
    public:
        vect operator()(size_t iter, vect const& p, double value, vect const& grad, matrix const& hess)
        {
            return -hess.inverse() * grad;
        }
    };
}