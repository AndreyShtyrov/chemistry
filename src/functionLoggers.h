#pragma once

#include "helper.h"

#include "producers/InPolar.h"
#include "linearAlgebraUtils.h"

template<typename FuncT>
void logFunctionInfo(FuncT& func, vect const& p, string const& title = "")
{
    auto valueGradHess = func.valueGradHess(p);
    auto value = get<0>(valueGradHess);
    auto grad = get<1>(valueGradHess);
    auto hess = get<2>(valueGradHess);

    LOG_INFO("{}\n\tposition: {}\n\tenergy: {}\n\tgradient: {} [{}]\n\thessian: {}\n\n", title, p.transpose(), value,
             grad.norm(), grad.transpose(), singularValues(hess));
}

template<typename FuncT>
void logFunctionPolarInfo(FuncT&& func, vect const& p, double r, string const& title = "")
{
    auto polar = makePolarWithDirection(func, r, p);

    auto valueGradHess = polar.valueGradHess(makeConstantVect(polar.nDims, M_PI / 2));
    auto value = get<0>(valueGradHess);
    auto grad = get<1>(valueGradHess);
    auto hess = get<2>(valueGradHess);

    LOG_INFO("{}\n\tposition: {}\n\tenergy: {}\n\tgradient: {} [{}]\n\thessian: {}\n\n", title, p.transpose(), value,
             grad.norm(), grad.transpose(), singularValues(hess));
}

