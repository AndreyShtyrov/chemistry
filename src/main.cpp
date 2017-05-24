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

vector<vect> parseLogForStartingPoints()
{
    vector<vect> result;

    ifstream log("loggg");
    string s;
    while (getline(log, s))
        if (s.find("initial polar Direction") != string::npos) {
            stringstream ss(s);
            for (int i = 0; i < 5; i++)
                ss >> s;

            vect v(11);
            for (int i = 0; i < v.rows(); i++)
                ss >> v(i);
            result.push_back(v);

//            stringstream ss;
//
//            getline(log, s);
//            for (int i = 0; i < 6; i++) {
//                getline(log, s);
//                ss << s << endl;
//            }
//
//            vector<size_t> charges;
//            vect initState;
//            tie(charges, initState) = readMolecule(ss);
//
//            result.push_back(initState);
        }

    return result;
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

    double firstR = 0.3;
    double deltaR = 0.05;

//    auto polarDirection = makeRandomVect(polar.nDims);
//    auto polarDirection = makeConstantVect(linearHessian.nDims - 1, M_PI / 2);
//    auto polarDirection = makeVect(1.25211,2.10604,1.30287,2.18491,0.827295,1.74907,1.37185,1.53325,1.49286,1.64736,1.59163);
//    auto polarDirection = readVect(ifstream("C2H4_polarDiractions"));

    ifstream input("C2H4_polarDirections");
    size_t cnt;
    input >> cnt;

    for (size_t i = 0; i < cnt; i++) {
        auto polarDirection = readVect(input);

        system(str(boost::format("mkdir %1%") % i).c_str());

        LOG_INFO("Starting new {} direction [{}]", i, polarDirection.transpose());

        for (size_t iter = 0; iter < 20; iter++) {
            auto polar = makePolar(linearHessian, firstR + iter * deltaR);
            auto deltaStrategy = makeRepeatDeltaStrategy(HessianDeltaStrategy());
            auto stopStrategy = makeHistoryStrategy(StopStrategy(1e-3, 1e-3));
            polarDirection = makeSecondGradientDescent(deltaStrategy, stopStrategy)(polar, polarDirection).back();

            LOG_INFO("Path chemcraft coords {}:\n {}\n", iter, polarDirection.transpose(),
                     toChemcraftCoords(charges, polar.fullTransform(polarDirection)));
            ofstream output(str(boost::format("./%1%/%2%.xyz") % i % iter));
            output << toChemcraftCoords(charges, polar.fullTransform(polarDirection));
        }
    }
}


int main()
{
    initializeLogger();

//    optimizeStructure();
    fullShs();

    return 0;
}