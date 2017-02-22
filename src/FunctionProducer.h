#pragma once

#include "helper.h"

template<int N_DIMS>
class FunctionProducer
{
public:
    static constexpr int N = N_DIMS;

    virtual ~FunctionProducer() = default;

    virtual double       operator()(vect<N> const& x) = 0;
    virtual vect<N>      grad(vect<N> const& x) = 0;
    virtual matrix<N, N> hess(vect<N> const& x) = 0;
};
