#pragma once

#include "helper.h"

namespace optimization
{
    class StopStrategy
    {
    public:
        StopStrategy(double gradThreshold, double deltaThreshold) : mGradThreshold(gradThreshold), mDeltaThreshold(deltaThreshold)
        { }

        bool operator()(size_t iter, vect const &p, double value, vect const &grad, vect const &delta) {
            return /*iter == 1 || */(grad.norm() < mGradThreshold && delta.norm() < mDeltaThreshold);
        }

        bool operator()(size_t iter, vect const &p, double value, vect const &grad, matrix const& hess, vect const &delta) {
            return iter == 20 || (grad.norm() < mGradThreshold && delta.norm() < mDeltaThreshold);
        }

    private:
        double mGradThreshold;
        double mDeltaThreshold;
    };
}