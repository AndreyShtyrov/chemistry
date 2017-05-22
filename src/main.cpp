#include "helper.h"

ofstream out;

#include <boost/format.hpp>
#include <boost/lexical_cast.hpp>
#include <Eigen/LU>

#include "linearization.h"
#include "FunctionProducers.h"
#include "PythongraphicsFramework.h"

#include "optimization/optimizations.h"
#include "InputOutputUtils.h"

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

template<typename T>
auto optimize(T& func, vect const& x, bool deltaHistory = false)
{
    if (deltaHistory) {
        auto optimizer = makeSecondGradientDescent(makeRepeatDeltaStrategy(HessianDeltaStrategy()),
                                                   makeHistoryStrategy(makeStandardAtomicStopStrategy(func)));
        auto path = optimizer(func, x);
        return path;
    } else {
        return makeSecondGradientDescent(makeRepeatDeltaStrategy(HessianDeltaStrategy()),
                                         makeStandardAtomicStopStrategy(func))(func, x);
    }
};

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
            toChemcraftCoords(weights, func.transform(localMinima)) % func.grad(localMinima).transpose() %
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
            toChemcraftCoords(weights, func.transform(localMinima)) % func.grad(localMinima).transpose() %
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
//    ifstream input("H2O");
    ifstream input("C2H4");

    vector<size_t> charges;
    vect initState;
    tie(charges, initState) = readMolecule(input);

    initState = rotateToFix(initState);
    auto molecule = GaussianProducer(charges);
    auto prepared = fixAtomSymmetry(makeAffineTransfomation(molecule, initState));

    LOG_INFO("nDims = {}", molecule.nDims);

    auto optimized = optimize(prepared, makeConstantVect(prepared.nDims, 0), true).back();

    cout << toChemcraftCoords(molecule.getCharges(), prepared.transform(optimized)) << endl;
}

void fullShs()
{
    vector<size_t> charges;
    vect initState;
    tie(charges, initState) = readMolecule(ifstream("C2H4"));

    initState = rotateToFix(initState);
    auto molecule = GaussianProducer(charges);
    auto fixed = fixAtomSymmetry(makeAffineTransfomation(molecule, initState));

//    auto equilStruct = optimize(fixed, makeConstantVect(fixed.nDims, 0)).back();
    auto equilStruct = makeVect(-0.495722, 0.120477, -0.874622, 0.283053, 0.784344, -0.00621205, -0.787401, -0.193879,
                                -0.301919, -0.553383, 0.552153, 0.529974);

    LOG_INFO("local minima: {}", equilStruct.transpose());
    LOG_INFO("chemcraft coords:\n{}", toChemcraftCoords(charges, fixed.fullTransform(equilStruct)));
    LOG_INFO("gradient: {}", fixed.grad(equilStruct).transpose());
    LOG_INFO("hessian values: {}", Eigen::JacobiSVD<matrix>(fixed.hess(equilStruct)).singularValues().transpose());

    auto linearHessian = prepareForPolar(fixed, equilStruct);

    double firstR = 0.2;
    double deltaR = 0.05;

//    auto polarDirection = makeRandomVect(polar.nDims);
//    auto polarDirection = makeConstantVect(linearHessian.nDims - 1, M_PI / 2);
//    auto polarDirection = makeVect(1.25211,2.10604,1.30287,2.18491,0.827295,1.74907,1.37185,1.53325,1.49286,1.64736,1.59163);
    auto polarDirection = readVect(ifstream("C2H4_polarDiractions"));

    for (size_t iter = 0;; iter++) {
        auto polar = makePolar(linearHessian, firstR + iter * deltaR);
        auto deltaStrategy = makeRepeatDeltaStrategy(HessianDeltaStrategy());
        auto stopStrategy = makeHistoryStrategy(StopStrategy(1e-3, 1e-3));
        polarDirection = makeSecondGradientDescent(deltaStrategy, stopStrategy)(polar, polarDirection).back();

        LOG_INFO("Path chemcraft coords: {}\n", polarDirection.transpose(),
                 toChemcraftCoords(charges, polar.fullTransform(polarDirection)));
        ofstream output("./second/" + to_string(iter) + ".xyz");
        output << toChemcraftCoords(charges, polar.fullTransform(polarDirection));
    }

//
////        auto initialPoint = makeRandomVect<polar.N>(polarVectLowerBound<polar.N>(), polarVectUpperBound<polar.N>());
//    auto initialPoint = makeVect(5.4,
//                                 1.0);//makeRandomVect<polar.N>(polarVectLowerBound<polar.N>(), polarVectUpperBound<polar.N>());
//    auto deltaStrategy = QuasiNewtonDeltaStrategy<BFGS>(polar.hess(initialPoint));
//    auto stopStrategy = StopStrategy(0.0001, 0.001);
//    auto polarMinima = makeGradientDescent(deltaStrategy, stopStrategy)(polar, initialPoint).back();
//
//
//    cout << boost::format("First polar minima: %1%\n") % polarMinima.transpose();
//    out << endl;
//
//    vector<vect> path;
//    path.push_back(polar.transform(polarMinima));
//
//    double lastValue = polar(polarMinima);
//    vect lastPoint = polarMinima;
//
//    vector<double> polarValues;
//    vector<vect> polarPath;
//    polarPath.push_back(lastPoint);

//    for (size_t i = 1; i < 20; ++i) {
//        cout << boost::format("iteration %1%:") % i << endl;
//        ofstream(str(boost::format("./data/%1%/%2%.xyz") % iter % i))
//           << to_chemcraft_coords(weights, func.transform(linear_hessian.transform(path.back()))) << endl;
//
//        auto polar2 = makePolar(linear_hessian, firstR + deltaR * i);
//
//        auto deltaStrategy = QuasiNewtonDeltaStrategy<BFGS>();
//        deltaStrategy.initializeHessian(polar2.hess(lastPoint));
//        auto stopStrategy = StopStrategy(0.0001, 0.001);
//        auto nextPoint = makeGradientDescent(deltaStrategy, stopStrategy)(polar2, lastPoint).back();
//
//        double curValue = polar2(nextPoint);
//
//        path.push_back(polar2.transform(nextPoint));
//        polarPath.push_back(nextPoint);
//        polarValues.push_back(curValue);
//
//        {
//            vector<double> xs, ys;
//            for (auto polarPoint : polarPath)
//                xs.push_back(polarPoint(0)), ys.push_back(polarPoint(1));
//            framework.scatter(framework.newPlot(), xs, ys);
//        }
//        framework.plot(framework.newPlot(), arange(polarValues.size()), polarValues);
//
//        cout << "\tnew value: " << curValue << endl;
//        auto hess = linear_hessian.hess(path.back());
//        auto A = linearization(hess);
//        cout << A.transpose() * hess * A << endl << endl;
//
//        if (curValue < lastValue) {
////                break;
//        }
//
//        lastValue = curValue;
//        lastPoint = nextPoint;
//    }
//
//    {
//        vector<double> xs, ys;
//        for (auto polarPoint : polarPath)
//            xs.push_back(polarPoint(0)), ys.push_back(polarPoint(1));
//        framework.scatter(framework.newPlot(), xs, ys);
//    }
//
//    auto pathToMin = optimize(linear_hessian, path.back());
//    path.insert(path.end(), pathToMin.begin(), pathToMin.end());
//
//    vector<double> vals;
//    for (auto point : path) {
//        vals.push_back(linear_hessian(point));
//    }
//    framework.plot(framework.newPlot(), arange(vals.size()), vals);
//
//
//    cerr << endl << endl << endl;
//    for (auto point : path) {
//        static int i = 0;
//        cout << to_chemcraft_coords(weights, func.transform(linear_hessian.transform(point))) << endl;
//        ++i;
//    }
}

int main()
{
    initializeLogger();
//    buildPolarPicture();
//    firstRadiusPolarPicture();

//    optimizeStructure();
    fullShs();

    return 0;
}