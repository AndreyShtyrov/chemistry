#pragma once

#include "helper.h"

#include "linearAlgebraUtils.h"

template<typename FuncT>
void logFunctionInfo(string const& title, FuncT& func, vect const& p)
{
    auto hess = func.hess(p);
    auto grad = func.grad(p);
    auto value = func(p);

    LOG_INFO("{}\n\tposition: {}\n\tenergy: {}\n\tgradient: {} [{}]\n\thessian: {}\n\n",
             title, p.transpose(), value, grad.norm(), grad.transpose(), singularValues(hess));
}

void shs();
void testTrajectory();
void testDegreesDelition();
void coordinateSystemChanges();
