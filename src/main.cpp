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
        auto optimizer = makeSecondGradientDescent(HessianDeltaStrategy(),
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
    tie(charges, initState) = readChemcraft(input);

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
        throw exc;
    }
}

template<typename FuncT>
void findInitialPolarDirections(FuncT& func, double r)
{
    auto axis = framework.newPlot();
    auto projMatrix = makeRandomMatrix(2, func.nDims + 5);

    while (true) {
        try {
            vector<double> xs, ys;

            vect pos = randomVectOnSphere(func.nDims, r);

            LOG_INFO("\n{}\n{}\n{}", pos.transpose(), func.fullTransform(pos).transpose(),
                     toChemcraftCoords({6, 6, 1, 1, 1, 1}, func.fullTransform(pos).transpose()));

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
            throw exc;
//            LOG_ERROR("exception: {}", exc.what());
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

        if (i < 3)
            continue;

        system(str(boost::format("mkdir %1%") % i).c_str());
        double value = linearHessian(direction);

        vector<double> xs, ys;

        for (size_t j = 0; j < 600; j++) {
            vect proj = projMatrix * direction / direction.norm();
            xs.push_back(proj(0)), ys.push_back(proj(1));

            double r = firstR + deltaR * j;

            vect prev = direction;
            direction = direction / direction.norm() * r;
            try {
                auto path = optimizeOnSphere(makeHistoryStrategy(StopStrategy(5e-4, 5e-4)), linearHessian, direction, r,
                                             0);

                if (path.empty()) {
                    LOG_ERROR("empty path");
                    break;
                }
                direction = path.back();

                double newValue = linearHessian(direction);
                LOG_INFO(
                   "New {} point in path:\n\tvalue = {:.13f}\n\tdelta norm = {:.13f}\n\t{}\nchemcraft coords:\n{}", j,
                   newValue, (direction / direction.norm() - prev / prev.norm()).norm(), direction.transpose(),
                   toChemcraftCoords(charges, linearHessian.fullTransform(direction)));

                ofstream output(str(boost::format("./%1%/%2%.xyz") % i % j));
                output << toChemcraftCoords(charges, linearHessian.fullTransform(direction)) << endl;

                if (newValue < value) {
                    LOG_ERROR("newValue < value [{:.13f} < {:.13f}]. Stopping", newValue, value);
                    //                break;
                }

                value = newValue;
            } catch (GaussianException const& exc) {
                break;
            }
        }

        framework.plot(framework.newPlot(), xs, ys);
//        return;
    }
}

vector<vect> fromCartesianToPositions(vect v)
{
    assert(v.rows() % 3 == 0);

    vector<vect> rs;
    for (size_t i = 0; i < (size_t) v.rows(); i += 3)
        rs.push_back(v.block(i, 0, 3, 1));
    return rs;
}

vect centerOfMass(vector<vect> const& rs)
{
    auto p = makeConstantVect(3, 0);
    for (auto r : rs)
        p += r;
    return p;
}

matrix tensorOfInertia(vector<vect> const& rs)
{
    auto J = makeConstantMatrix(3, 3, 0);
    for (auto r : rs)
        J += identity(3) * r.dot(r) + r * r.transpose();
    return J;
}

vect solveEquations(vect v)
{
    auto withZeros = makeConstantVect((size_t) v.rows() + 6, 0);
    for (size_t i = 0, j = 0; i < (size_t) withZeros.rows(); i++)
        if (i < 7 && i != 5)
            withZeros(i) = 0;
        else
            withZeros(i) = v(j++);
    LOG_INFO("\nv = {}\nw = {}", v.transpose(), withZeros.transpose());
    v = withZeros;

    auto rs = fromCartesianToPositions(v);
    auto p = centerOfMass(rs);
    auto J = tensorOfInertia(rs);

    LOG_INFO("before:\np: {}\nJ:\n{}", p.transpose(), J);

    double A = -p(0);
    double B = -p(1);
    double C = -p(2);
    double D = -J(0, 1);
    double E = -J(0, 2);
    double F = -J(1, 2);

    LOG_INFO("values: {} {} {} {} {} {}", A, B, C, D, E, F);

    {
        LOG_INFO("Linear Equasions #2: x: {}", C);
        v(2) = C;
    }

    {
        Eigen::Matrix2d m;
        m << 1, 1, v(2), v(5);
        Eigen::Vector2d b;
        b << B, F;
        Eigen::Vector2d x = m.colPivHouseholderQr().solve(b);
        LOG_INFO("Linear Equasions #2:\nm:\n{}\nb: {}\nx: {}", m, b.transpose(), x.transpose());
        v(1) = x(0);
        v(4) = x(1);
    }

    {
        Eigen::Matrix3d m;
        m << 1, 1, 1, v(1), v(4), v(7), v(2), v(5), v(8);
        Eigen::Vector3d b;
        b << A, D, E;
        Eigen::Vector3d x = m.colPivHouseholderQr().solve(b);
        LOG_INFO("Linear Equasions #3:\nm:\n{}\nb: {}\nx: {}", m, b.transpose(), x.transpose());
        v(0) = x(0);
        v(3) = x(1);
        v(6) = x(2);
    }

    rs = fromCartesianToPositions(v);
    p = centerOfMass(rs);
    J = tensorOfInertia(rs);
    LOG_INFO("after:\np: {}\nJ:\n{}", p.transpose(), J);
    LOG_INFO("result: {}", v.transpose());

    return v;
}

void analizeFolder()
{
    vector<double> energies;
    vector<vect> states;
    for (size_t i = 0; i < 200; i++) {
        vector<size_t> charges;
        vect state;
        tie(charges, state) = readChemcraft(ifstream(str(boost::format("./2/%1%.xyz") % i)));

        GaussianProducer molecule(charges);
        auto fixed = fixAtomSymmetry(molecule);
        state = fixed.backTransform(state);

        auto hess = fixed.hess(state);
        auto grad = fixed.grad(state);
        auto energy = fixed(state);

        energies.push_back(energy);
        states.push_back(state);

        LOG_INFO("State #{}: {}\n\tenergy = {}\n\tgradient = {} [{}]\n\thess values = {}\nchemcraft coords:\n{}", i,
                 state.transpose(), energy, grad.norm(), grad.transpose(), singularValues(hess),
                 toChemcraftCoords(charges, fixed.fullTransform(state)));
    }

    framework.plot(framework.newPlot(), energies);

    for (size_t i = 0; i < energies.size(); i++) {
        LOG_INFO("#{}: {}, {}", i, i == 0 || energies[i - 1] < energies[i] ? '+' : '-',
                 i + 1 == energies.size() || energies[i] < energies[i + 1] ? '+' : '-');
    }
}

int main()
{
    initializeLogger();
//    shs();
//    return 0;

    ifstream input("./C2H4");
    auto charges = readCharges(input);
    auto equilStruct = readVect(input);

    auto molecule = GaussianProducer(charges);

    LOG_INFO("Initial structure:\n\tlocal minima: {}\nchemcraft coords:\n\t{}\tenergy: {}\n\tgradient: {}\n\thessian: {}\n\n",
             equilStruct.transpose(), toChemcraftCoords(charges, equilStruct), molecule(equilStruct),
             molecule.grad(equilStruct).transpose(), singularValues(molecule.hess(equilStruct)));

    auto fixedSym = fixAtomSymmetry(molecule);
    equilStruct = fixedSym.backTransform(equilStruct);

    LOG_INFO("For fixed coordinates:\n\tlocal minima: {}\nchemcraft coords:\n\t{}\tenergy: {}\n\tgradient: {}\n\thessian: {}\n\n",
             equilStruct.transpose(), toChemcraftCoords(charges, fixedSym.fullTransform(equilStruct)), fixedSym(equilStruct),
             fixedSym.grad(equilStruct).transpose(), singularValues(fixedSym.hess(equilStruct)));

//    auto v = makeRandomVect(9);
//    auto rs = fromCartesianToPositions(v);
//
//    LOG_INFO("\n{}", v.transpose());
//    LOG_INFO("\n{}", centerOfMass(rs).transpose());
//    LOG_INFO("\n{}", inertyTensor(rs));

//    ifstream input("./C2H4");
//    vector<size_t> charges = readCharges(input);
//    vect state = readVect(input);
//
//    GaussianProducer molecule(charges);
//    auto fixed = fixAtomTranslations(molecule);
//
//    LOG_INFO("{}", state.transpose());
//    state = fixed.backTransform(moveToFix(state));
//    LOG_INFO("{}", state.transpose());
//    auto linearHessian = prepareForPolar(fixed, state);
//
//    findInitialPolarDirections(linearHessian, 0.01);

//    vector<vect> states;
//    for (size_t i = 0; i < 200; i++) {
//        vector<size_t> charges;
//        vect state;
//        tie(charges, state) = readChemcraft(ifstream(str(boost::format("./0/%1%.xyz") % i)));
//
//        states.push_back(state);
//    }
//
//    vect d1 = states[165] - states[164];
//    vect d2 = states[166] - states[165];
//
//    cout << d1.transpose() << endl << d2.transpose() << endl << d1.dot(d2) / d1.norm() / d2.norm() << endl;
//    cout << d1.norm() << ' ' << d2.norm() << endl;

//    filterPolarDirectionsLogFile();


////    findInitialPolarDirections(linearHessian, .3);
}