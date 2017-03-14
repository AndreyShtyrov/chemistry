#pragma once

#include "helper.h"

#include "History.h"

namespace optimization
{
    template<int N_DIMS>
    class GradientLengthStopCriteria : public History<N_DIMS>
    {
    public:
        using History<N_DIMS>::N;

        GradientLengthStopCriteria(double threshold) : mThreshold(threshold)
        { }

        bool operator()(vect<N> const& pos, vect<N> const& delta)
        {
            mDeltas.push_back(delta);
            return delta.norm() < mThreshold;
        }

    private:
        double mThreshold;
        vector<vect<N>> mDeltas;
    };
}