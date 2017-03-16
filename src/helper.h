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

template<int Rows, int Cols>
using matrix = Eigen::Matrix<double, Rows, Cols>;

template<size_t Rows>
using vect = Eigen::Matrix<double, Rows, 1>;

template<typename T>
T sqr(T const& a)
{
    return a * a;
}

template<int N, int>
void assign(vect<N>& v)
{ }

template<int N, int P, typename ArgT, typename... ArgsT>
void assign(vect<N>& v, ArgT&& arg, ArgsT&&... args)
{
    v(P) = forward<ArgT>(arg);
    assign<N, P + 1>(v, forward<ArgsT>(args)...);
}

template<typename... Args>
vect<sizeof...(Args)> make_vect(Args&&... args)
{
    vect<sizeof...(Args)> v;
    assign<sizeof...(Args), 0>(v, forward<Args>(args)...);
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

template<int N, int M>
matrix<N, M> make_random_matrix()
{
    matrix<N, N> matr;
    matr.setRandom();
    return matr;
};

template<int N>
vect<N> make_random_vect()
{
    vect<N> v;
    v.setRandom();
    return v;
};

template<int N>
vect<N> make_constant_vect(double constant)
{
    vect<N> v;
    v.setConstant(constant);
    return v;
}

template<int N>
vect<N> eye(size_t i)
{
    vect<N> result;
    result.setZero();
    result(i) = 1.;

    return result;
};

template<typename T, typename... Ts>
struct first_type
{
    using type = T;
};
template<typename... Ts> using first_type_t = typename first_type<Ts...>::type;

template<typename T, typename... Ts>
struct last_type
{
    using type = typename last_type<Ts...>::type;
};

template<typename T>
struct last_type<T>
{
    using type = T;
};

template<typename... Ts>
using last_type_t = typename last_type<Ts...>::type;

