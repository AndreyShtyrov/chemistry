#pragma once

#include "GradientDescent.h"
#include "SecondOrderGradientDescent.h"

#include "delta_strategies/FollowGradDeltaStrategy.h"
#include "delta_strategies/HessianDeltaStrategy.h"
#include "delta_strategies/QuasiNewtonDeltaStrategy.h"
#include "delta_strategies/RepeatingDeltaStrategy.h"

#include "stop_strategies/AtomicStopStrategy.h"
#include "stop_strategies/DeltaNormStopStrategy.h"
#include "stop_strategies/GradientNormStopStrategy.h"
#include "stop_strategies/StopStrategy.h"
#include "stop_strategies/HistoryStrategyWrapper.h"

namespace optimization
{
    template<template<int, typename> typename OptimizerT, typename CriteriaT>
    OptimizerT<CriteriaT::N, CriteriaT> make_optimizer_with_criteria(CriteriaT const &criteria)
    {
        return OptimizerT<CriteriaT::N, CriteriaT>(criteria);
    }
}