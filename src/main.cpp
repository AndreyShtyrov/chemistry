#include "helper.h"

ofstream out;

#include <boost/format.hpp>
#include <boost/lexical_cast.hpp>
#include <Eigen/LU>

#include "linearAlgebraUtils.h"
#include "producers/producers.h"
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

vector<vect> parseLogForStartingPoints(string const& file, size_t nDims)
{
    vector<vect> result;

    ifstream log(file);
    string s;
    while (getline(log, s))
        if (s.find("initial polar Direction") != string::npos) {
            stringstream ss(s);
            for (int i = 0; i < 5; i++)
                ss >> s;

            vect v(nDims);
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

template<typename FuncT, typename StopStrategy>
vector<vect> optimizeOnSphere(StopStrategy stopStrategy, FuncT& func, vect p, double r, size_t preHessIters)
{
    try {
        assert(abs(r - p.norm()) < 1e-7);

        auto e = eye(func.nDims, func.nDims - 1);
        auto theta = makeConstantVect(func.nDims - 1, M_PI / 2);

        vector<vect> path;

        for (size_t iter = 0;; iter++) {
            auto rotation = rotationMatrix(e, p);
            auto rotated = makeAffineTransfomation(func, rotation);
            auto polar = makePolar(rotated, r);

            double value;
            vect grad;
            matrix hess(func.nDims, func.nDims);

            vect lastP = p;
            if (iter < preHessIters) {
                hess.setZero();
                grad = polar.grad(theta);
                value = polar(theta);

                p = rotated.transform(polar.transform(theta - 2 * grad));
            } else {
                hess = polar.hess(theta);
                grad = polar.grad(theta);
                value = polar(theta);

                p = rotated.transform(polar.transform(theta - hess.inverse() * grad));
            }

            path.push_back(p);

            if (stopStrategy(iter, p, value, grad, hess, p - lastP))
                break;
        }

        return path;
    } catch (GaussianException const& exc) {
        return {};
    }
}

template<typename FuncT>
void findInitialPolarDirections(FuncT& func, double r)
{
    auto axis = framework.newPlot();
    auto projMatrix = makeRandomMatrix(2, func.nDims + 6);

    while (true) {
        try {
            vector<double> xs, ys;

            vect pos = randomVectOnSphere(func.nDims, r);

            auto path = optimizeOnSphere(makeHistoryStrategy(StopStrategy(1e-4, 1e-4)), func, pos, r, 100);
            if (path.empty())
                continue;

            for (auto const& p : path) {
                vect t = func.fullTransform(p);
                vect proj = projMatrix * t;
                xs.push_back(proj(0));
                ys.push_back(proj(1));
            }


            framework.plot(axis, xs, ys);
            framework.scatter(axis, xs, ys);

            LOG_INFO("initial polar Direction: {}", path.back().transpose());

            vect p = path.back();
            vect grad = func.grad(p);
            vect dir = p / p.norm();
            vect first = dir * grad.dot(dir);
            vect second = grad - first;
            LOG_INFO("read grad norm is {}\nfirst norm = {}\nsecond norm = {}", grad.norm(), first.norm(),
                     second.norm());
        } catch (GaussianException const& exc) {
            LOG_ERROR("exception: {}", exc.what());
        }
    }
}

vector<vector<double>> calcPairwiseDists(vect v)
{
    assert(v.rows() % 3 == 0);

    size_t n = (size_t) v.rows() / 3;
    vector<vector<double>> dists(n);
    for (size_t i = 0; i < n; i++)
        for (size_t j = 0; j < n; j++)
            dists[i].push_back((v.block(i * 3, 0, 3, 1) - v.block(j * 3, 0, 3, 1)).norm());

    return dists;
}

double calcDist(vect v1, vect v2)
{
    auto d1 = calcPairwiseDists(v1);
    auto d2 = calcPairwiseDists(v2);

    vector<size_t> permut;
    for (size_t i = 0; i < d1.size(); i++)
        permut.push_back(i);

    double minMse = -1;

    do {
        double mse = 0;
        for (size_t i = 0; i < d1.size(); i++)
            for (size_t j = 0; j < d1.size(); j++)
                mse += sqr(d1[i][j] - d2[permut[i]][permut[j]]);

        if (minMse < 0 || mse < minMse)
            minMse = mse;
    } while (next_permutation(permut.begin(), permut.end()));

    return minMse;
}

void filterPolarDirectionsLogFile()
{
    ifstream input("C2H4");
    auto charges = readCharges(input);
    auto equilStruct = readVect(input);

    auto molecule = GaussianProducer(charges);
    auto fixedSym = fixAtomSymmetry(molecule);
    equilStruct = fixedSym.backTransform(equilStruct);

    auto linearHessian = prepareForPolar(fixedSym, equilStruct);

    auto ps = parseLogForStartingPoints("./logs/log_2017-05-31_13-38", 12);
    cout << ps.size() << endl;

    vector<bool> used(ps.size());
    for (size_t i = 0; i < ps.size(); i++)
        for (size_t j = i + 1; j < ps.size(); j++)
            if (calcDist(linearHessian.fullTransform(ps[i]), linearHessian.fullTransform(ps[j])) < 1e-1)
                used[j] = true;

    size_t cnt = 0;
    for (size_t i = 0; i < ps.size(); i++)
        if (!used[i])
            cnt++;
    cout << cnt << endl;

    for (auto val : used)
        cout << val;
    cout << endl << endl;

    cout.precision(10);
    for (size_t i = 0; i < ps.size(); i++)
        if (!used[i]) {
            cout << ps[i].rows() << endl;
            for (int j = 0; j < ps[i].rows(); j++)
                cout << fixed << ps[i](j) << ' ';
            cout << endl;
        }
    cout << endl << endl << endl;

    cout << ps.size() << endl;
    for (auto const& p : ps)
        cout << p.transpose() << endl;
    cout << endl << endl;

    cout.precision(2);
    for (size_t i = 0; i < ps.size(); i++) {
        auto const& p1 = ps[i];
        if (used[i])
            continue;

        vector<double> xs;

        for (auto const& p2 : ps) {
//            auto dist = (p1 - p2).norm();
            auto dist = calcDist(linearHessian.fullTransform(p1), linearHessian.fullTransform(p2));

            cout << fixed << dist << ' ';
            xs.push_back(dist);
        }
        cout << endl;

        sort(xs.begin(), xs.end());
        framework.plot(framework.newPlot(), xs);
    }
}

#include <boost/filesystem.hpp>

using namespace boost::filesystem;

void shs()
{
    ifstream input("./C2H4");
    auto charges = readCharges(input);
    auto equilStruct = readVect(input);

    auto molecule = GaussianProducer(charges);
    auto fixedSym = fixAtomSymmetry(molecule);
    equilStruct = fixedSym.backTransform(equilStruct);

    LOG_INFO("local minima: {}", equilStruct.transpose());
    LOG_INFO("chemcraft coords:\n{}", toChemcraftCoords(charges, fixedSym.fullTransform(equilStruct)));
    LOG_INFO("energy: {:.13f}", fixedSym(equilStruct));
    LOG_INFO("gradient: {}", fixedSym.grad(equilStruct).transpose());
    LOG_INFO("hessian values: {}", Eigen::JacobiSVD<matrix>(fixedSym.hess(equilStruct)).singularValues().transpose());

    auto linearHessian = prepareForPolar(fixedSym, equilStruct);

    size_t cnt;
    input >> cnt;

    double const firstR = 0.3;
    double const deltaR = 0.01;

    auto projMatrix = makeRandomMatrix(2, linearHessian.nDims);

    for (size_t i = 0; i < cnt; i++) {
        auto direction = readVect(input);
        LOG_INFO("Path #{}. Initial direction: {}", i, direction.transpose());

        system(str(boost::format("mkdir %1%") % i).c_str());
        double value = linearHessian(direction);

        vector<double> xs, ys;

        for (size_t j = 0; j < 200; j++) {
            vect proj = projMatrix * direction / direction.norm();
            xs.push_back(proj(0)), ys.push_back(proj(1));

            double r = firstR + deltaR * j;

            vect prev = direction;
            direction = direction / direction.norm() * r;
            auto path = optimizeOnSphere(makeHistoryStrategy(StopStrategy(5e-4, 5e-4)), linearHessian, direction, r, 0);
            if (path.empty()) {
                LOG_ERROR("empty path");
                break;
            }
            direction = path.back();

            double newValue = linearHessian(direction);
            LOG_INFO("New {} point in path:\n\tvalue = {:.13f}\n\tdelta norm = {:.13f}\n\t{}\nchemcraft coords:\n{}", j,
                     newValue, (direction / direction.norm() - prev / prev.norm()).norm(), direction.transpose(),
                     toChemcraftCoords(charges, linearHessian.fullTransform(direction)));

            ofstream output(str(boost::format("./%1%/%2%.xyz") % i % j));
            output << toChemcraftCoords(charges, linearHessian.fullTransform(direction)) << endl;

            if (newValue < value) {
                LOG_ERROR("newValue < value [{:.13f} < {:.13f}]. Stopping", newValue, value);
//                break;
            }

            value = newValue;
        }

        framework.plot(framework.newPlot(), xs, ys);
        return;
    }
}

int main()
{
    initializeLogger();
//    filterPolarDirectionsLogFile();
    shs();

//    findInitialPolarDirections(linearHessian, .3);
}