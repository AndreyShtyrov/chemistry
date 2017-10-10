#pragma once

#include <iostream>
#include <fstream>
#include <cstdio>
#include <cassert>
#include <vector>
#include <memory>
#include <cmath>
#include <random>
#include <type_traits>
#include <queue>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <boost/type_index.hpp>
#include <boost/optional.hpp>
#include <spdlog/spdlog.h>
#include <spdlog/fmt/ostr.h>

#define SPDLOG_STR_H(x) #x
#define SPDLOG_STR_HELPER(x) SPDLOG_STR_H(x)

#define LOG_DEBUG(...) logger->debug("[" __FILE__ ":" SPDLOG_STR_HELPER(__LINE__) "] " __VA_ARGS__)
#define LOG_INFO(...) logger->info("[" __FILE__ ":" SPDLOG_STR_HELPER(__LINE__) "] " __VA_ARGS__)
#define LOG_WARN(...) logger->warn("[" __FILE__ ":" SPDLOG_STR_HELPER(__LINE__) "] " __VA_ARGS__)
#define LOG_ERROR(...) logger->error("[" __FILE__ ":" SPDLOG_STR_HELPER(__LINE__) "] " __VA_ARGS__)
#define LOG_CRITICAL(...) logger->critical("[" __FILE__ ":" SPDLOG_STR_HELPER(__LINE__) "] " __VA_ARGS__)

using namespace std;

using boost::optional;
using boost::make_optional;

using fmt::format;

extern shared_ptr<spdlog::logger> logger;
void initializeLogger();

extern thread_local mt19937 randomGen;

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
