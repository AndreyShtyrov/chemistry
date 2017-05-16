#pragma once

#include "History.h"
#include "GradientLengthStopCriteria.h"

#include "GradientDescent.h"
#include "SecondOrderGradientDescent.h"

#include "FollowGradientDeltaStrategy.h"
#include "HessianDeltaStrategy.h"
#include "QuasiNewtonDeltaStrategy.h"

#include "AtomicStopStrategy.h"
#include "DeltaNormStopStrategy.h"
#include "GradientNormStopStrategy.h"
#include "HistoryStrategyWrapper.h"
#include "StopStrategy.h"

namespace optimization
{
    template<template<int, typename> typename OptimizerT, typename CriteriaT>
    OptimizerT<CriteriaT::N, CriteriaT> make_optimizer_with_criteria(CriteriaT const &criteria)
    {
        return OptimizerT<CriteriaT::N, CriteriaT>(criteria);
    }
}