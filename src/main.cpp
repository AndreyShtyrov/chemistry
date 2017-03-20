#include "helper.h"

ofstream out;

#include <boost/format.hpp>
#include <boost/lexical_cast.hpp>
#include <Eigen/LU>
#include <optimization/GradientLengthStopCriteria.h>
#include <optimization/GradientOptimization.h>

#include "linearization.h"
#include "FunctionProducers.h"
#include "PythongraphicsFramework.h"

#include "optimization/optimizations.h"

using namespace std;
using namespace optimization;

constexpr double EPS = 1e-7;

constexpr double calculate_delta(double min, double max, int n)
{
    return (max - min) / (n - 1);
}

constexpr double MAX_VAL = 1;
constexpr double MIN_X = -MAX_VAL;
constexpr double MAX_X = MAX_VAL;
constexpr double MIN_Y = -MAX_VAL;
constexpr double MAX_Y = MAX_VAL;

constexpr size_t N = 250;
constexpr double DX = calculate_delta(MIN_X, MAX_X, N);
constexpr double DY = calculate_delta(MIN_Y, MAX_Y, N);

constexpr size_t PHI = 1000;

template<typename FuncT, typename Optimizer1T, typename Optimizer2T>
vector<vect<FuncT::N>>
shs(FuncT& func, Optimizer1T optimizer1, Optimizer2T optimizer2, double deltaR, vect<FuncT::N> pos)
{
    vector<vect<FuncT::N>> path;
    path.push_back(pos);

    double lastValue = func(pos);

    vect<FuncT::N - 1> lastPolar;
    lastPolar.setRandom();
    cout << lastPolar << endl << endl;

    for (int i = 0; i < 20; ++i) {
//        cerr << "!" << endl;

        auto polars = makePolar(func, deltaR * (i + 1));
        lastPolar = optimizer1(polars, lastPolar).back();

        pos = polars.transform(lastPolar);

        double curValue = func(pos);
        if (curValue < lastValue) {
            break;
        }
        lastValue = curValue;
        path.push_back(pos);
    }


    auto saddle = pos;
    for (int i = 0; i < 3; i++) {
        path.push_back(saddle);
        saddle = optimize(pos, func.grad(saddle), func.hess(saddle));
    }
    path.push_back(saddle);

//    auto pathToMinimum = optimizer2(func, pos);
//    path.insert(path.end(), pathToMinimum.begin(), pathToMinimum.end());

    return path;
}

double get_linear_comb(double from, double to, double t)
{
    return from + (to - from) * t;
}

template<typename FuncT>
string drawPlot(FuncT&& func, double from, double to, size_t iters)
{
    vector<double> xs, ys;
    for (size_t i = 0; i < iters; i++) {
        auto x = get_linear_comb(from, to, (double) i / (iters - 1));
        xs.push_back(x);
        ys.push_back(func(make_vect(x)));
    }

    auto axis = framework.newPlot();
    framework.plot(axis, xs, ys);

    return axis;
}

template<typename FuncT>
string draw3dPlot(FuncT&& func, vect<2> from, vect<2> to, size_t iters)
{
    vector<double> xs, ys, zs;
    for (size_t i = 0; i < iters; i++)
        for (size_t j = 0; j < iters; j++) {
            double x = get_linear_comb(from(0), to(0), (double) i / (iters - 1));
            double y = get_linear_comb(from(1), to(1), (double) j / (iters - 1));
            xs.push_back(x);
            ys.push_back(y);
            zs.push_back(func(make_vect(x, y)));
        }

    auto axis = framework.newPlot();
    framework.contour(axis, reshape(xs, iters), reshape(ys, iters), reshape(zs, iters), 250);

    return axis;
}

template<int N>
vect<N> getRandomPoint(vect<N> const& lowerBound, vect<N> const& upperBound)
{
    auto p = make_random_vect<N>();
    return lowerBound.array() + p.array() * (upperBound.array() - lowerBound.array());
}

template<int N>
string to_chemcraft_coords(vector<size_t> const& sizes, vect<N> p, double globalDx=0)
{
    assert(N == sizes.size() * 3);

    stringstream result;
    for (size_t i = 0; i < sizes.size(); i++)
        result << boost::format("%1%\t%2%\t%3%\t%4%") % sizes[i] % (p(i * 3 + 0) + globalDx) % p(i * 3 + 1) % p(i * 3 + 2) << endl;
    return result.str();
}

template<template<typename, typename> typename OptimizerT, typename DeltaStrategyT, typename StopStrategyT, typename FuncT, int N>
tuple<vector<double>, vector<double>>
testOptimizer(DeltaStrategyT&& deltaStrategy, StopStrategyT&& stopStrategy, FuncT& func, vect<N> const& p)
{
    auto optimizer = OptimizerT<HistoryStrategyWrapper<decay_t<DeltaStrategyT>>, decay_t<StopStrategyT>>(
       make_history_strategy(forward<DeltaStrategyT>(deltaStrategy)), forward<StopStrategyT>(stopStrategy));
    auto path = optimizer(func, p);
    auto vals = optimizer.getDeltaStrategy().getValues();

    return make_tuple(arange(vals.size()), vals);
};

void standardOptimizationTest()
{
    auto v = make_vect(1.04218, 0.31040, 1.00456);
//    auto v = make_vect(1.00335, -0.140611, 0.993603);

    vector<size_t> weights = {8, 1, 1};
    auto atomicFunc = GaussianProducer<9>(weights);
    auto func = fix_atom_symmetry(atomicFunc);

    auto axis = framework.newPlot();
    vector<double> path, vals;

    tie(path, vals) = testOptimizer<GradientDescent>(QuasiNewtonDeltaStrategy<func.N, DFP>(),
                                                     makeAtomicStopStrategy(0.00045, 0.0003, 0.018, 0.012, func), func,
                                                     v);
    framework.plot(axis, arange(vals.size()), vals);

    tie(path, vals) = testOptimizer<SecondOrderGradientDescent>(HessianDeltaStrategy<func.N>(),
                                                                makeAtomicStopStrategy(0.00045, 0.0003, 0.018, 0.012,
                                                                                       func), func, v);
    framework.plot(axis, arange(vals.size()), vals);

    tie(path, vals) = testOptimizer<GradientDescent>(FollowGradientDeltaStrategy<func.N>(),
                                                     makeAtomicStopStrategy(0.00045, 0.0003, 0.018, 0.012, func), func,
                                                     v);
    framework.plot(axis, arange(vals.size()), vals);
};

template<typename T>
auto optimize(T& func, vect<T::N> const& x)
{
    return makeGradientDescent(QuasiNewtonDeltaStrategy<T::N, BFGS>(), makeStandardAtomicStopStrategy(func))(func, x);
//    return makeGradientDescent(FollowGradientDeltaStrategy<T::N>(), makeStandardAtomicStopStrategy(func))(func, x).back();
};


int main()
{
    auto start_point = make_vect(1.04218, 0.31040, 1.00456);

    vector<size_t> weights = {8, 1, 1};
    auto atomicFunc = GaussianProducer<9>(weights);
    auto func = fix_atom_symmetry(atomicFunc);

    auto local_minima = optimize(func, start_point).back();

    cout << boost::format("local minima:\n%1%\ngradient: %2%\nhessian:\n%3%\n\n") %
            to_chemcraft_coords(weights, func.transform(local_minima)) % func.grad(local_minima).transpose() %
            func.hess(local_minima);


    auto linear_hessian = prepareForPolar(func, local_minima);
    auto zero = makeConstantVect<linear_hessian.N>(0.);
    cout << linear_hessian.grad(zero).transpose() << endl << endl << linear_hessian.hess(zero) << endl;

    out = ofstream("a.out");

    double firstR = 0.3;
    double deltaR = 0.1;
    auto polar = makePolar(linear_hessian, firstR);

    for (size_t iter = 0; iter < 10; iter++) {
        auto deltaStrategy = QuasiNewtonDeltaStrategy<polar.N, BFGS>();
        auto stopStrategy = StopStrategy(0.0001, 0.001);
//        auto polar_minima = makeGradientDescent(deltaStrategy, stopStrategy)(polar, make_vect(5.24, 0.96)).back();
//        auto polar_minima = makeGradientDescent(deltaStrategy, stopStrategy)(polar, make_vect(0.73, 0.17)).back();
        auto polar_minima = makeGradientDescent(deltaStrategy, stopStrategy)(polar, make_vect(5.5, -1.1)).back();
//        auto polar_minima = makeGradientDescent(deltaStrategy, stopStrategy)(polar, make_random_vect<polar.N>()).back();

        cerr << boost::format("polar minima:\n\t%1%\n") % polar_minima;
        out << endl;

        vector<vect<linear_hessian.N>> path;
        path.push_back(polar.transform(polar_minima));

        double lastValue = polar(polar_minima);
        vect<polar.N> lastPoint = polar_minima;

        vector<double> polarValues;
        vector<vect<polar.N>> polarPath;
        polarPath.push_back(lastPoint);

        for (size_t i = 1; i < 20; ++i) {
            auto polar2 = makePolar(linear_hessian, firstR + deltaR * i);

            auto deltaStrategy = QuasiNewtonDeltaStrategy<polar2.N, BFGS>(0.5);
            auto stopStrategy = StopStrategy(0.0001, 0.001);
            auto nextPoint = makeGradientDescent(deltaStrategy, stopStrategy)(polar2, lastPoint).back();

            double curValue = polar2(nextPoint);

            path.push_back(polar2.transform(nextPoint));
            polarPath.push_back(nextPoint);
            polarValues.push_back(curValue);

            {
                vector<double> xs, ys;
                for (auto polarPoint : polarPath)
                    xs.push_back(polarPoint(0)), ys.push_back(polarPoint(1));
                framework.scatter(framework.newPlot(), xs, ys);
            }
            framework.plot(framework.newPlot(), arange(polarValues.size()), polarValues);

            cerr << "new value: " << curValue << endl;
            auto hess = linear_hessian.hess(path.back());
            auto A = linearization(hess);
            cerr << A.transpose() * hess * A << endl << endl;
            cerr << "Chemcraft coords:" << endl;
            cerr << to_chemcraft_coords(weights, func.transform(linear_hessian.transform(path.back()))) << endl;

            if (curValue < lastValue) {
//                break;
            }

            lastValue = curValue;
            lastPoint = nextPoint;
        }

        {
            vector<double> xs, ys;
            for (auto polarPoint : polarPath)
                xs.push_back(polarPoint(0)), ys.push_back(polarPoint(1));
            framework.scatter(framework.newPlot(), xs, ys);
        }

        auto pathToMin = optimize(linear_hessian, path.back());
        path.insert(path.end(), pathToMin.begin(), pathToMin.end());

        vector<double> vals;
        for (auto point : path) {
            vals.push_back(linear_hessian(point));
        }
        framework.plot(framework.newPlot(), arange(vals.size()), vals);


        cerr << endl << endl << endl;
        for (auto point : path)
        {
            static int i = 0;
            cout << to_chemcraft_coords(weights, func.transform(linear_hessian.transform(point)), 10 * i) << endl;
            ++i;
        }


        break;
    }


}
