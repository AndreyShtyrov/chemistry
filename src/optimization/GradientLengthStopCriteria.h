#pragma once

#include "helper.h"

#include "History.h"

namespace optimization
{
    class GradientLengthStopCriteria
    {
    public:
        GradientLengthStopCriteria(double threshold) : mThreshold(threshold)
        { }

        bool operator()(vect const& pos, vect const& delta)
        {
            mDeltas.push_back(delta);
            return delta.norm() < mThreshold;
        }

    private:
        double mThreshold;
        vector<vect> mDeltas;
    };
}