#pragma once

#include <iostream>
#include <fstream>
#include <cstdio>
#include <cassert>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <boost/format.hpp>
#include <boost/type_index.hpp>

using namespace std;

using matrix = Eigen::MatrixXd;

using vect = Eigen::VectorXd;

template<typename T>
T sqr(T const& a)
{
    return a * a;
}

template<int>
void assign(vect& v)
{ }

template<int P, typename ArgT, typename... ArgsT>
void assign(vect& v, ArgT&& arg, ArgsT&&... args)
{
    v(P) = forward<ArgT>(arg);
    assign<P + 1>(v, forward<ArgsT>(args)...);
}

template<typename... Args>
vect makeVect(Args&& ... args)
{
    vect v(sizeof...(Args));
    assign<0>(v, forward<Args>(args)...);
    return v;
}

inline vector<double> arange(size_t size)
{
    vector<double> range(size);
    for (size_t i = 0; i < size; i++)
        range[i] = i;
    return range;
}

inline vector<double> linspace(double from, double to, size_t iters) {
    assert(iters > 1);

    vector<double> result;
    for (size_t i = 0; i < iters; i++)
        result.push_back(from + (to - from) / (iters - 1) * i);
    return result;
}

inline matrix makeRandomMatrix(int rows, int cols)
{
    matrix matr(rows, cols);
    matr.setRandom();
    return matr;
};

inline vect makeRandomVect(int n)
{
    vect v(n);
    v.setRandom();
    return v;
};

inline vect makeRandomVect(vect const& lowerBound, vect const& upperBound)
{
    return lowerBound + (0.5 * (1 + makeRandomVect(lowerBound.rows()).array()) * (upperBound - lowerBound).array()).matrix();
}

inline vect makeConstantVect(int n, double constant)
{
    vect v(n);
    v.setConstant(constant);
    return v;
}

inline vect eye(int n, size_t i)
{
    vect result(n);
    result.setZero();
    result(i) = 1.;

    return result;
};

inline matrix makeConstantMatrix(int rows, int cols, double constant)
{
    matrix m(rows, cols);
    m.setConstant(constant);
    return m;
}

inline matrix identity(int rows, int cols)
{
    matrix result(rows, cols);
    result.setIdentity();
    return result;
};

template<typename T>
vect readVect(int rows, T&& stream)
{
    vect v(rows);
    for (size_t i = 0; i < rows; i++)
        stream >> v(i);
    return v;
}