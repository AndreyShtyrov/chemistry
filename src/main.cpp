#include "helper.h"

ofstream out;

#include <boost/format.hpp>
#include <boost/lexical_cast.hpp>
#include <Eigen/LU>
#include <optimization/GradientLengthStopCriteria.h>

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
vector<vect> shs(FuncT& func, Optimizer1T optimizer1, Optimizer2T optimizer2, double deltaR, vect pos)
{
    vector<vect> path;
    path.push_back(pos);

    double lastValue = func(pos);

    vect lastPolar;
    lastPolar.setRandom();
    cout << lastPolar << endl << endl;

    for (int i = 0; i < 20; ++i) {
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
        ys.push_back(func(makeVect(x)));
    }

    auto axis = framework.newPlot();
    framework.plot(axis, xs, ys);

    return axis;
}

template<typename FuncT>
string draw3dPlot(FuncT&& func, vect from, vect to, size_t iters)
{
    vector<double> xs, ys, zs;
    for (size_t i = 0; i < iters; i++)
        for (size_t j = 0; j < iters; j++) {
            double x = get_linear_comb(from(0), to(0), (double) i / (iters - 1));
            double y = get_linear_comb(from(1), to(1), (double) j / (iters - 1));
            xs.push_back(x);
            ys.push_back(y);
            zs.push_back(func(makeVect(x, y)));
        }

    auto axis = framework.newPlot();
    framework.contour(axis, reshape(xs, iters), reshape(ys, iters), reshape(zs, iters), 250);

    return axis;
}

vect getRandomPoint(vect const& lowerBound, vect const& upperBound)
{
    auto p = makeRandomVect(lowerBound.rows());
    return lowerBound.array() + p.array() * (upperBound.array() - lowerBound.array());
}

template<template<typename, typename> typename OptimizerT, typename DeltaStrategyT, typename StopStrategyT, typename FuncT>
tuple<vector<double>, vector<double>>
testOptimizer(DeltaStrategyT&& deltaStrategy, StopStrategyT&& stopStrategy, FuncT& func, vect const& p)
{
    auto optimizer = OptimizerT<HistoryStrategyWrapper<decay_t<DeltaStrategyT>>, decay_t<StopStrategyT>>(
       makeHistoryDeltaStrategy(forward<DeltaStrategyT>(deltaStrategy)), forward<StopStrategyT>(stopStrategy));
    auto path = optimizer(func, p);
    auto vals = optimizer.getDeltaStrategy().getValues();

    return make_tuple(arange(vals.size()), vals);
};

void standardOptimizationTest()
{
    auto v = makeVect(1.04218, 0.31040, 1.00456);
//    auto v = make_vect(1.00335, -0.140611, 0.993603);

    vector<size_t> weights = {8, 1, 1};
    auto atomicFunc = GaussianProducer(weights);
    auto func = fixAtomSymmetry(atomicFunc);

    auto axis = framework.newPlot();
    vector<double> path, vals;

    tie(path, vals) = testOptimizer<GradientDescent>(QuasiNewtonDeltaStrategy<DFP>(),
                                                     makeAtomicStopStrategy(0.00045, 0.0003, 0.018, 0.012, func), func,
                                                     v);
    framework.plot(axis, arange(vals.size()), vals);

    tie(path, vals) = testOptimizer<SecondOrderGradientDescent>(HessianDeltaStrategy(),
                                                                makeAtomicStopStrategy(0.00045, 0.0003, 0.018, 0.012,
                                                                                       func), func, v);
    framework.plot(axis, arange(vals.size()), vals);

    tie(path, vals) = testOptimizer<GradientDescent>(FollowGradientDeltaStrategy(),
                                                     makeAtomicStopStrategy(0.00045, 0.0003, 0.018, 0.012, func), func,
                                                     v);
    framework.plot(axis, arange(vals.size()), vals);
};

template<typename T>
auto optimize(T& func, vect const& x)
{
    return makeGradientDescent(QuasiNewtonDeltaStrategy<BFGS>(func.hess(x)), makeStandardAtomicStopStrategy(func))(func, x);
};


void fullShs()
{
    auto start_point = makeVect(1.04218, 0.31040, 1.00456);

    vector<size_t> weights = {8, 1, 1};
    auto atomicFunc = GaussianProducer(weights);
    auto func = fixAtomSymmetry(atomicFunc);

    auto localMinima = optimize(func, start_point).back();
//    auto localMinima = makeVect(0.996622544216218, -0.240032763088067, 0.967285186815903);

    cout << boost::format("local minima:\n%1%\ngradient: %2%\nhessian:\n%3%\n\n") %
            to_chemcraft_coords(weights, func.transform(localMinima)) % func.grad(localMinima).transpose() %
            func.hess(localMinima);

    auto linear_hessian = prepareForPolar(func, localMinima);

    out = ofstream("a.out");

    double firstR = 0.3;
    double deltaR = 0.1;
    auto polar = makePolar(linear_hessian, firstR);

    for (size_t iter = 0; iter < 10; iter++) {
//        auto initialPoint = makeRandomVect<polar.N>(polarVectLowerBound<polar.N>(), polarVectUpperBound<polar.N>());
        auto initialPoint = makeVect(5.4,
                                     1.0);//makeRandomVect<polar.N>(polarVectLowerBound<polar.N>(), polarVectUpperBound<polar.N>());
        auto deltaStrategy = QuasiNewtonDeltaStrategy<BFGS>(polar.hess(initialPoint));
        auto stopStrategy = StopStrategy(0.0001, 0.001);
        auto polarMinima = makeGradientDescent(deltaStrategy, stopStrategy)(polar, initialPoint).back();

        cout << boost::format("First polar minima: %1%\n") % polarMinima.transpose();
        out << endl;

        vector<vect> path;
        path.push_back(polar.transform(polarMinima));

        double lastValue = polar(polarMinima);
        vect lastPoint = polarMinima;

        vector<double> polarValues;
        vector<vect> polarPath;
        polarPath.push_back(lastPoint);

        ofstream(str(boost::format("./data/%1%/%2%.xyz") % iter % 0))
           << to_chemcraft_coords(weights, func.transform(localMinima)) << endl;

        for (size_t i = 1; i < 20; ++i) {
            cout << boost::format("iteration %1%:") % i << endl;
            ofstream(str(boost::format("./data/%1%/%2%.xyz") % iter % i))
               << to_chemcraft_coords(weights, func.transform(linear_hessian.transform(path.back()))) << endl;

            auto polar2 = makePolar(linear_hessian, firstR + deltaR * i);

            auto deltaStrategy = QuasiNewtonDeltaStrategy<BFGS>();
            deltaStrategy.initializeHessian(polar2.hess(lastPoint));
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

            cout << "\tnew value: " << curValue << endl;
            auto hess = linear_hessian.hess(path.back());
            auto A = linearization(hess);
            cout << A.transpose() * hess * A << endl << endl;

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
        for (auto point : path) {
            static int i = 0;
            cout << to_chemcraft_coords(weights, func.transform(linear_hessian.transform(point))) << endl;
            ++i;
        }

        break;
    }
}

void buildPolarPicture()
{
    auto start_point = makeVect(1.04218, 0.31040, 1.00456);

    vector<size_t> weights = {8, 1, 1};
    auto atomicFunc = GaussianProducer(weights);
    auto func = fixAtomSymmetry(atomicFunc);

    auto localMinima = optimize(func, start_point).back();

    cout.precision(15);
    cout << fixed << localMinima.transpose() << endl;
    cout << boost::format("local minima:\n%1%\ngradient: %2%\nhessian:\n%3%\n\n") %
            to_chemcraft_coords(weights, func.transform(localMinima)) % func.grad(localMinima).transpose() %
            func.hess(localMinima);
    return;
//    auto linearHess = prepareForPolar(func, localMinima);
//    auto zero = makeConstantVect < linearHess.N > (0.);
//    cout << linearHess.grad(zero).transpose() << endl << endl << linearHess.hess(zero) << endl;
//
//    out = ofstream("a.out");
//
//    double firstR = 0.3;
//    auto polar = makePolar(linearHess, firstR);
//
//    size_t const N = 100;
//    vector<double> xs, ys, zs;
//
//    auto lowerBound = polarVectLowerBound < polar.N > ();
//    auto upperBound = polarVectUpperBound < polar.N > ();
//    for (size_t i = 0; i < N; i++)
//        for (size_t j = 0; j < N; j++) {
//            xs.push_back(lowerBound(0) + (upperBound(0) - lowerBound(0)) * i / (N - 1));
//            ys.push_back(lowerBound(1) + (upperBound(1) - lowerBound(1)) * j / (N - 1));
//            zs.push_back(polar(makeVect(xs.back(), ys.back())));
//        }
//    framework.contour(framework.newPlot(), reshape(xs, N), reshape(ys, N), reshape(zs, N), 100);
}

void firstRadiusPolarPicture()
{
//    auto start_point = makeVect(1.04218, 0.31040, 1.00456);

    vector<size_t> weights = {8, 1, 1};
    auto atomicFunc = GaussianProducer(weights);
    auto func = fixAtomSymmetry(atomicFunc);

//    auto localMinima = optimize(func, start_point).back();
    auto localMinima = makeVect(0.996622544216218, -0.240032763088067, 0.967285186815903);

    cout << boost::format("local minima:\n%1%\ngradient: %2%\nhessian:\n%3%\n\n") %
            to_chemcraft_coords(weights, func.transform(localMinima)) % func.grad(localMinima).transpose() %
            func.hess(localMinima);

    auto linearHess = prepareForPolar(func, localMinima);
    auto zero = makeConstantVect(linearHess.nDims, 0.);
    cout << linearHess.grad(zero).transpose() << endl << endl << linearHess.hess(zero) << endl;

    out = ofstream("a.out");

    double firstR = 0.3;
    auto polar = makePolar(linearHess, firstR);

    for (size_t iter = 0; iter < 10; iter++) {
        vect startingPoint = makeRandomVect(polarVectLowerBound(polar.nDims), polarVectUpperBound(polar.nDims));

        auto deltaStrategy = QuasiNewtonDeltaStrategy<BFGS>();
        deltaStrategy.initializeHessian(polar.hess(startingPoint));
        auto stopStrategy = StopStrategy(0.000001, 0.00001);
        auto polarMinima = makeGradientDescent(deltaStrategy, stopStrategy)(polar, startingPoint).back();

        cout << boost::format("polar minima: %1%\n") % polarMinima.transpose();
        out << endl;
    }
}

//void c2h4() {
//    vector<size_t> weights = {6, 6, 1, 1, 1, 1};
////    auto startPoint = makeVect(0.664007917, 1.726194372, -1.239100083, -1.022740312, -0.120487628, -1.239100083,
////                               -1.022740312, 1.726194372, 1.236957917, -1.022740312, -0.120487628, 1.236957917);
//    auto startPoint = readVect<18>(ifstream("./data/cur.in"));
//    startPoint /= (double) GaussianProducer<18>::MAGIC_CONSTANT;
//
//    auto atomicFunc = GaussianProducer<18>(weights);
//    auto func = fixAtomSymmetry(atomicFunc, startPoint);
//
////    auto localMinima = optimize(func, startPoint).back();
////    auto localMinima = makeVect(0.996622544216218, -0.240032763088067, 0.967285186815903);
//    vect<func.N> localMinima;
////    Eigen::MatrixXf localMinima;
////    cout << startPoint << endl;
//    localMinima << startPoint(5), startPoint.template block<func.N - 1, 1>(7, 0);
////    cout << localMinima;
//
//    cout << boost::format("local minima:\n%1%\ngradient: %2%\nhessian:\n%3%\n\n") %
//            to_chemcraft_coords(weights, func.transform(localMinima)) % func.grad(localMinima).transpose() %
//            func.hess(localMinima);
//
//    auto linear_hessian = prepareForPolar(func, localMinima);
//
//    out = ofstream("a.out");
//
//    double firstR = 0.3;
//    double deltaR = 0.1;
//    auto polar = makePolar(linear_hessian, firstR);
//
//    for (size_t iter = 0; iter < 10; iter++) {
////        auto initialPoint = makeVect(5.4, 1.0);
//        auto initialPoint = makeRandomVect<polar.N>(polarVectLowerBound<polar.N>(), polarVectUpperBound<polar.N>());
//        auto deltaStrategy = QuasiNewtonDeltaStrategy<polar.N, BFGS>(polar.hess(initialPoint));
//        auto stopStrategy = StopStrategy(0.0001, 0.001);
//        auto polarMinima = makeGradientDescent(deltaStrategy, stopStrategy)(polar, initialPoint).back();
//
//        cout << boost::format("First polar minima: %1%\n") % polarMinima.transpose();
//        out << endl;
//
//        vector<vect<linear_hessian.N>> path;
//        path.push_back(polar.transform(polarMinima));
//
//        double lastValue = polar(polarMinima);
//        vect<polar.N> lastPoint = polarMinima;
//
//        vector<double> polarValues;
//        vector<vect<polar.N>> polarPath;
//        polarPath.push_back(lastPoint);
//
//        ofstream(str(boost::format("./data/%1%/%2%.xyz") % iter % 0))
//                << to_chemcraft_coords(weights, func.transform(localMinima)) << endl;
//
//        for (size_t i = 1; i < 20; ++i) {
//            cout << boost::format("iteration %1%:") % i << endl;
//            ofstream(str(boost::format("./data/%1%/%2%.xyz") % iter % i))
//                    << to_chemcraft_coords(weights, func.transform(linear_hessian.transform(path.back()))) << endl;
//
//            auto polar2 = makePolar(linear_hessian, firstR + deltaR * i);
//
//            auto deltaStrategy = QuasiNewtonDeltaStrategy<polar2.N, BFGS>();
//            deltaStrategy.initializeHessian(polar2.hess(lastPoint));
//            auto stopStrategy = StopStrategy(0.0001, 0.001);
//            auto nextPoint = makeGradientDescent(deltaStrategy, stopStrategy)(polar2, lastPoint).back();
//
//            double curValue = polar2(nextPoint);
//
//            path.push_back(polar2.transform(nextPoint));
//            polarPath.push_back(nextPoint);
//            polarValues.push_back(curValue);
//
//            {
//                vector<double> xs, ys;
//                for (auto polarPoint : polarPath)
//                    xs.push_back(polarPoint(0)), ys.push_back(polarPoint(1));
//                framework.scatter(framework.newPlot(), xs, ys);
//            }
//            framework.plot(framework.newPlot(), arange(polarValues.size()), polarValues);
//
//            cout << "\tnew value: " << curValue << endl;
//            auto hess = linear_hessian.hess(path.back());
//            auto A = linearization(hess);
//            cout << A.transpose() * hess * A << endl << endl;
//
//            if (curValue < lastValue) {
////                break;
//            }
//
//            lastValue = curValue;
//            lastPoint = nextPoint;
//        }
//
//        {
//            vector<double> xs, ys;
//            for (auto polarPoint : polarPath)
//                xs.push_back(polarPoint(0)), ys.push_back(polarPoint(1));
//            framework.scatter(framework.newPlot(), xs, ys);
//        }
//
//        auto pathToMin = optimize(linear_hessian, path.back());
//        path.insert(path.end(), pathToMin.begin(), pathToMin.end());
//
//        vector<double> vals;
//        for (auto point : path) {
//            vals.push_back(linear_hessian(point));
//        }
//        framework.plot(framework.newPlot(), arange(vals.size()), vals);
//
//
//        cerr << endl << endl << endl;
//        for (auto point : path) {
//            static int i = 0;
//            cout << to_chemcraft_coords(weights, func.transform(linear_hessian.transform(point)), 10 * i) << endl;
//            ++i;
//        }
//
//        break;
//    }
//}


void optimizeStructure()
{
    ifstream input("H2O");

    auto molecule = readMolecule(input);
    auto initState = readVect(molecule.nDims, input);
    auto prepared = fixAtomSymmetry(makeAffineTransfomation(molecule, initState));

    LOG_INFO("nDims = {}", molecule.nDims);
    LOG_INFO("gradient: {}", molecule.grad(initState).transpose());
    LOG_INFO("gradient: {}", prepared.grad(makeConstantVect(prepared.nDims, 0)).transpose());

    auto optimized = optimize(prepared, makeConstantVect(prepared.nDims, 0)).back();

    cout << to_chemcraft_coords(molecule.getCharges(), prepared.transform(optimized)) << endl;
}

int main()
{
    initializeLogger();

//    buildPolarPicture();
//    firstRadiusPolarPicture();
//    fullShs();

    optimizeStructure();

    return 0;
}