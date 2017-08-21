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
using namespace boost;
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

        for (size_t iter = 0; iter < preHessIters; iter++) {
            auto rotation = rotationMatrix(e, p);
            auto rotated = makeAffineTransfomation(func, rotation);
            auto polar = makePolar(rotated, r);

            double value;
            vect grad;
            matrix hess(func.nDims, func.nDims);

            vect lastP = p;
            if (iter < preHessIters) {
                auto valueGrad = polar.valueGrad(theta);

                value = get<0>(valueGrad);
                grad = get<1>(valueGrad);
                hess.setZero();

                p = rotated.transform(polar.transform(theta - grad));
            } else {
                auto valueGradHess = polar.valueGradHess(theta);

                value = get<0>(valueGradHess);
                grad = get<1>(valueGradHess);
                hess = get<2>(valueGradHess);

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

template<typename FuncT, typename StopStrategy>
vector<vect> optimizeOnSphere2(StopStrategy stopStrategy, FuncT& func, vect p, double r, size_t preHessIters)
{
    try {
        assert(abs(r - p.norm()) < 1e-7);

        auto e = eye(func.nDims, func.nDims - 1);
        auto theta = makeConstantVect(func.nDims - 1, M_PI / 2);

        vector<vect> path;

        vect momentum;
        double const gamma = .9;
        double const alpha = 1;

        for (size_t iter = 0; iter < preHessIters; iter++) {
            auto rotation = rotationMatrix(e, p);
            auto rotated = makeAffineTransfomation(func, rotation);
            auto polar = makePolar(rotated, r);

            vect grad = 0.1 * polar.grad(theta);
            auto value = polar(theta);
            matrix hess(func.nDims, func.nDims);
            hess.setZero();

            if (iter)
                momentum = gamma * momentum + alpha * grad;
            else
                momentum = grad;

            auto lastP = p;
            p = rotated.transform(polar.transform(theta - momentum));
            path.push_back(p);

            if (stopStrategy(iter, p, value, grad, hess, p - lastP))
                break;
        }

        return path;
    } catch (GaussianException const& exc) {
        throw exc;
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

    auto linearHessian = normalizeForPolar(fixedSym, equilStruct);

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



#include "testing/tests.h"

void drawTrajectories()
{
    RandomProjection proj(15);
    vector<size_t> quantities = {335, 318, 218, 43};
    auto axis = framework.newPlot();
    for (size_t i = 0; i < quantities.size(); i++) {
        vector<double> xs, ys;
        for (size_t j = 0; j < quantities[i]; j++) {
            vector<size_t> charges;
            vect state;
            tie(charges, state) = readChemcraft(ifstream(str(format("./%1%/%2%.xyz") % i % j)));
            auto cur = proj(toDistanceSpace(state, false));
            xs.push_back(cur(0));
            ys.push_back(cur(1));
        }

        framework.plot(axis, xs, ys);
    }
}

template<typename FuncT, typename StopStrategy>
bool tryToConverge(StopStrategy stopStrategy, FuncT& func, vect p, double r, vector<vect>& path, size_t globalIter)
{
    auto const theta = makeConstantVect(func.nDims - 1, M_PI / 2);
    bool converged = false;

    vector<vect> newPath;
    try {
        for (size_t i = 0; i < 5; i++) {
            auto polar = makePolarWithDirection(func, r, p);

            auto valueGradHess = polar.valueGradHess(theta);
            auto value = get<0>(valueGradHess);
            auto grad = get<1>(valueGradHess);
            auto hess = get<2>(valueGradHess);

            auto lastP = p;
            p = polar.getInnerFunction().transform(polar.transform(theta - hess.inverse() * grad));
            newPath.push_back(p);

            if (stopStrategy(globalIter + i, p, value, grad, p - lastP)) {
                converged = true;
                break;
            }
        }
    } catch (GaussianException const& exc) {
        LOG_ERROR("Did not converged");
    }

    if (converged) {
        path.insert(path.end(), newPath.begin(), newPath.end());
        return true;
    }

    return false;
};

template<typename FuncT, typename StopStrategy>
vector<vect> optimizeOnSphere3(StopStrategy stopStrategy, FuncT& func, vect p, double r, size_t preHessIters)
{
    try {
        assert(abs(r - p.norm()) < 1e-7);

        auto const theta = makeConstantVect(func.nDims - 1, M_PI / 2);

        vector<vect> path;
        vect momentum;

        for (size_t iter = 0; ; iter++) {
            if (iter % preHessIters == 0 && tryToConverge(stopStrategy, func, p, r, path, iter)) {
                LOG_ERROR("breaked here");
                break;
            }

            auto polar = makePolarWithDirection(func, r, p);

            auto valueGrad = polar.valueGrad(theta);
            auto value = get<0>(valueGrad);
            auto grad = get<1>(valueGrad);

            if (iter)
                momentum = 0.5 * (1 + momentum.dot(grad) / (grad.norm() * momentum.norm())) * momentum + grad;
            else
                momentum = grad;

            auto lastP = p;
            p = polar.getInnerFunction().transform(polar.transform(theta - momentum));
            path.push_back(p);

            if (stopStrategy(iter, p, value, grad, p - lastP)) {
                LOG_ERROR("breaked here");
                break;
            }
        }

        return path;
    } catch (GaussianException const& exc) {
        throw exc;
    }
}

template<typename FuncT>
vector<vect> findInitialPolarDirections(FuncT& func, double r)
{
    auto axis = framework.newPlot();
    RandomProjection projection(func.getFullInnerFunction().nDims);

    vector<vect> result;

#pragma omp parallel for
    for (size_t i = 0; i < 4; i++) {
        try {
            vector<double> xs, ys;

            vect pos = randomVectOnSphere(func.nDims, r);

            LOG_INFO("\n{}\n{}\n{}", pos.transpose(), func.fullTransform(pos).transpose(),
                     toChemcraftCoords({6, 6, 1, 1, 1, 1}, func.fullTransform(pos).transpose()));

            auto path = optimizeOnSphere3(makeHistoryStrategy(StopStrategy(1e-4, 1e-4)), func, pos, r, 50);
            if (path.empty())
                continue;

            for (auto const& p : path) {
                vect t = func.fullTransform(p);
                vect proj = projection(t);
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
            LOG_INFO("read grad norm is {}\nfirst norm = {}\nsecond norm = {}", grad.norm(), first.norm(), second.norm());

            result.push_back(path.back());
        } catch (GaussianException const& exc) {
            throw exc;
//            LOG_ERROR("exception: {}", exc.what());
        }
    }

    return result;
}

template<typename FuncT>
void logFunctionInfo(string const& title, FuncT& func, vect const& p)
{
    auto hess = func.hess(p);
    auto grad = func.grad(p);
    auto value = func(p);

    LOG_INFO("{}\n\tposition: {}\n\tenergy: {}\n\tgradient: {} [{}]\n\thessian: {}\n\n",
             title, p.transpose(), value, grad.norm(), grad.transpose(), singularValues(hess));
}

double getTimeFromNow(chrono::time_point<chrono::system_clock> const& timePoint)
{
    return chrono::duration<double>(chrono::system_clock::now() - timePoint).count();
}

void benchmark(string const &method, size_t nProc, size_t mem, size_t iters)
{
    static string const pattern = "%%chk=chk\n"
            "%%nproc=%1%\n"
            "%%mem=%2%mb\n"
            "# B3lyp/3-21g nosym %3%\n"
            "\n"
            "\n"
            "0 1\n"
            "6\t0.000000000000000000000000000000\t0.000000000000000000000000000000\t0.000000000000000000000000000000\n"
            "6\t1.329407574910000056078729357978\t0.000000000000000000000000000000\t0.000000000000000000000000000000\n"
            "1\t-0.573933341279999953421508962492\t0.921778012560000026276441076334\t0.000000000000000000000000000000\n"
            "1\t-0.573933775450000016604690245003\t-0.921778567219999955817399950320\t-0.000004707069999999999810257837\n"
            "1\t1.903341015899999932869945951097\t0.921778658150000040905069909059\t0.000004367929999999999699671939\n"
            "1\t1.903341021419999945507584016013\t-0.921778294680000054306390211423\t-0.000003291940000000000028635132\n"
            "\n";

    auto globalStartTime = chrono::system_clock::now();

#pragma omp parallel for
    for (size_t i = 0; i < iters; i++) {
        string filemask = boost::str(boost::format("./tmp/tmp%1%") % std::hash<std::thread::id>()(this_thread::get_id()));
        auto localStartTime = chrono::system_clock::now();

        ofstream inputFile(filemask + ".in");
        inputFile << boost::format(pattern) % nProc % mem % method;
        inputFile.close();

        system(str(boost::format("mg09D %1%.in %1%.out > /dev/null") % filemask).c_str());
    }
    double duration = getTimeFromNow(globalStartTime);
    LOG_INFO("{}.{}.{} all iters : {}, {} per iteration", method, nProc, mem, duration, duration / iters);
}

void runBenchmarks()
{
    for (string const& method : {"scf", "force", "freq"})
        for (size_t nProc : {1, 2, 3, 4})
            for (size_t mem : {250, 500, 750, 1000, 1250})
                benchmark(method, nProc, mem, 100);
}



int main()
{
    initializeLogger();

    ifstream input("./C2H4");
    auto charges = readCharges(input);
    auto equilStruct = readVect(input);

    auto molecule = fixAtomSymmetry(GaussianProducer(charges));
    logFunctionInfo("", molecule, molecule.backTransform(equilStruct));
    equilStruct = molecule.backTransform(equilStruct);
    auto normalized = normalizeForPolar(molecule, equilStruct);

    auto startTime = chrono::system_clock::now();
    auto result = findInitialPolarDirections(normalized, 0.1);
    LOG_INFO("time passed: {}s", chrono::duration<double>(chrono::system_clock::now() - startTime).count());

    ofstream output("./mins_on_sphere");
    for (auto& v : result)
        output << v.rows() << endl << v.transpose() << endl;

    ofstream output2("./mins_on_sphere2");
    for (auto v : result) {
        v = normalized.fullTransform(v);
        output2 << v.rows() << endl << v.transpose() << endl;
    }

    return 0;
}