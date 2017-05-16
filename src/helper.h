#pragma once

#include <iostream>
#include <fstream>
#include <cstdio>
#include <cassert>
#include <vector>
#include <memory>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <boost/type_index.hpp>
#include <boost/format.hpp>
#include <spdlog/spdlog.h>
#include <spdlog/fmt/ostr.h>

#define SPDLOG_STR_H(x) #x
#define SPDLOG_STR_HELPER(x) SPDLOG_STR_H(x)

#define LOG_INFO(...) logger->info("[" __FILE__ ":" SPDLOG_STR_HELPER(__LINE__) "] " __VA_ARGS__)
#define LOG_WARN(...) logger->warn("[" __FILE__ ":" SPDLOG_STR_HELPER(__LINE__) "] " __VA_ARGS__)
#define LOG_ERROR(...) logger->error("[" __FILE__ ":" SPDLOG_STR_HELPER(__LINE__) "] " __VA_ARGS__)
#define LOG_CRITICAL(...) logger->critical("[" __FILE__ ":" SPDLOG_STR_HELPER(__LINE__) "] " __VA_ARGS__)

using namespace std;

extern shared_ptr<spdlog::logger> logger;
void initializeLogger();

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
vect readVect(size_t rows, T&& stream)
{
    vect v(rows);
    for (size_t i = 0; i < rows; i++)
        stream >> v(i);
    return v;
}

inline string to_chemcraft_coords(vector<size_t> const& charges, vect p)
{
    stringstream result;
    for (size_t i = 0; i < charges.size(); i++)
        result << boost::format("%1%\t%2%\t%3%\t%4%") % charges[i] % p(i * 3 + 0) % p(i * 3 + 1) % p(i * 3 + 2) << endl;
    return result.str();
}
