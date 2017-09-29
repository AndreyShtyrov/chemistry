#include "helper.h"

#include <boost/format.hpp>
#include <boost/lexical_cast.hpp>
#include <Eigen/LU>

#include "linearAlgebraUtils.h"
#include "producers/producers.h"
#include "PythongraphicsFramework.h"

#include "optimization/optimizations.h"
#include "optimization/optimizeOnSphere.h"
#include "InputOutputUtils.h"
#include "constants.h"

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

vector<vect> fromCartesianToPositions(vect v)
{
    assert(v.rows() % 3 == 0);

    vector<vect> rs;
    for (size_t i = 0; i < (size_t) v.rows(); i += 3)
        rs.push_back(v.block(i, 0, 3, 1));
    return rs;
}

vect centerOfMass(vector<size_t> charges, vector<vect> const& rs)
{
    double massSum = 0;
    auto p = makeConstantVect(3, 0);

    for (size_t i = 0; i < charges.size(); i++) {
        p += rs[i] * MASSES[charges[i]];
        massSum += MASSES[charges[i]];
    }

    return p / massSum;
}

matrix tensorOfInertia(vector<size_t> charges, vector<vect> const& rs)
{
    auto J = makeConstantMatrix(3, 3, 0);
    for (size_t i = 0; i < charges.size(); i++)
        J += MASSES[charges[i]] * (identity(3) * rs[i].dot(rs[i]) + rs[i] * rs[i].transpose());
    return J;
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

template<typename FuncT>
void logFunctionInfo(FuncT& func, vect const& p, string const& title = "")
{
    auto valueGradHess = func.valueGradHess(p);
    auto value = get<0>(valueGradHess);
    auto grad = get<1>(valueGradHess);
    auto hess = get<2>(valueGradHess);

    LOG_INFO("{}\n\tposition: {}\n\tenergy: {}\n\tgradient: {} [{}]\n\thessian: {}\n\n", title, p.transpose(), value,
             grad.norm(), grad.transpose(), singularValues(hess));
}

template<typename FuncT>
void logFunctionPolarInfo(FuncT&& func, vect const& p, double r, string const& title = "")
{
    auto polar = makePolarWithDirection(func, r, p);

    auto valueGradHess = polar.valueGradHess(makeConstantVect(polar.nDims, M_PI / 2));
    auto value = get<0>(valueGradHess);
    auto grad = get<1>(valueGradHess);
    auto hess = get<2>(valueGradHess);

    LOG_INFO("{}\n\tposition: {}\n\tenergy: {}\n\tgradient: {} [{}]\n\thessian: {}\n\n", title, p.transpose(), value,
             grad.norm(), grad.transpose(), singularValues(hess));
}

vector<vect> filterByDistance(vector<vect> const& vs, double r)
{
    vector<vect> result;

    for (size_t i = 0; i < vs.size(); i++) {
        bool flag = true;
        for (size_t j = 0; j < i; j++)
            if ((vs[i] - vs[j]).norm() < r)
                flag = false;
        if (flag)
            result.push_back(vs[i]);
    }
    return result;
}

template<typename FuncT>
vector<vect> filterBySingularValues(vector<vect> const& vs, FuncT& func)
{
    vector<vect> result;

    for (auto const& v : vs) {
        auto polar = makePolarWithDirection(func, 0.1, v);
        auto sValues = singularValues(polar.hess(makeConstantVect(polar.nDims, M_PI / 2)));

        bool flag = true;
        for (size_t i = 0; flag && i < sValues.size(); i++)
            if (sValues(i) < 0)
                flag = false;

        if (flag)
            result.push_back(v);
    }

    return result;
}


void analizeMinsOnSphere()
{

    ifstream input("./C2H4");
    auto charges = readCharges(input);
    auto equilStruct = readVect(input);

    auto molecule = fixAtomSymmetry(GaussianProducer(charges, 1));
    equilStruct = molecule.backTransform(equilStruct);
    auto normalized = normalizeForPolar(molecule, equilStruct);

    logFunctionInfo(normalized, makeConstantVect(normalized.nDims, 0), "normalized energy for equil structure");

    ifstream mins("./mins_on_sphere");
    size_t cnt;
    mins >> cnt;
    vector<vect> vs;
    for (size_t i = 0; i < cnt; i++)
        vs.push_back(readVect(mins));

    vs = filterByDistance(vs, .0001);
    LOG_INFO("{} remained after filtering", vs.size());

    RandomProjection proj((size_t) vs.back().size());
    vector<double> xs, ys;
    for (auto const& v : vs) {
        auto projection = proj(v);
        xs.push_back(projection(0));
        ys.push_back(projection(1));
    }
    framework.scatter(framework.newPlot(), xs, ys);

    ofstream mins2("./mins_on_sphere_filtered");
    mins2.precision(30);
    mins2 << vs.size() << endl;
    for (auto const& v : vs)
        mins2 << v.size() << endl << fixed << v.transpose() << endl;

    for (auto const& v : vs) {
        auto polar = makePolarWithDirection(normalized, .1, v);
        logFunctionInfo(polar, makeConstantVect(polar.nDims, M_PI / 2), "");
    }
}

template<typename FuncT>
void findInitialPolarDirections(FuncT& func, double r)
{
    auto const axis = framework.newPlot();
    RandomProjection const projection(func.nDims);
    StopStrategy const stopStrategy(1e-7, 1e-7);

    ofstream output("./C2H4/mins_on_sphere");
    output.precision(30);

#pragma omp parallel for
    for (size_t i = 0; i < 2 * func.nDims; i++) {
        vect pos = r * eye(func.nDims, i / 2);
        if (i % 2)
            pos *= -1;

        auto path = optimizeOnSphere(stopStrategy, func, pos, r, 50);
        if (path.empty())
            continue;

#pragma omp critical
        {
            vect p = path.back();
            output << p.size() << endl << fixed << p << endl;

            vector<double> xs, ys;
            for (auto const& p : path) {
                vect proj = projection(p);
                xs.push_back(proj(0));
                ys.push_back(proj(1));
            }

            framework.plot(axis, xs, ys);
            framework.scatter(axis, xs, ys);

            auto polar = makePolarWithDirection(func, r, path.back());
            logFunctionInfo(polar, makeConstantVect(polar.nDims, M_PI / 2),
                            str(format("new direction (%1%)") % path.back().transpose()));
        }
    }
}

template<typename FuncT>
void minimaElimination(FuncT&& func)
{
    func.getFullInnerFunction().setGaussianNProc(3);
    auto zeroEnergy = func(makeConstantVect(func.nDims, 0));

    double const r = .05;
    vector<double> values;
    vector<vect> directions;

    auto const axis = framework.newPlot();
    RandomProjection const projection(func.nDims);
    auto stopStrategy = makeHistoryStrategy(StopStrategy(1e-4 * r, 1e-4 * r));

    bool firstEpochFinished = false;
    auto lastSuccessIter = (size_t) -1;

    while (!firstEpochFinished) {
        for (size_t iter = 0; iter < (func.nDims - 2) * 2; iter++) {
            if (iter == lastSuccessIter) {
                firstEpochFinished = true;
                break;
            }

//            Cosine3OnSPhereInterpolation supplement(normalized.nDims, values, directions);
//            LargestCosine3OnSphere supplement(normalized.nDims, values, directions);
//            Cosine3OnSphereInterpolation supplement(normalized.nDims, values, directions);
            CleverCosine3OnSphereInterpolation supplement(func.nDims, values, directions);
            auto withSupplement = func + supplement;

            int sign = 2 * ((int) iter % 2) - 1;
            vect startingDirection = r * sign * eye(func.nDims, iter / 2);
            auto path = optimizeOnSphere(stopStrategy, withSupplement, startingDirection, r, 50, 5);
            auto direction = path.back();

            logFunctionPolarInfo(withSupplement, direction, r, "func in new direction");
            logFunctionPolarInfo(func, direction, r, "normalized in new direction");

            stringstream distances;
            double minAngle = 0;
            for (auto const& prevDir : directions) {
                minAngle = max(minAngle, angleCosine(direction, prevDir));
                distances
                   << boost::format("[%1%, %2%]") % distance(direction, prevDir) % angleCosine(direction, prevDir);
            }
            LOG_ERROR("Distances from previous {} directons [dist, cos(angle)]:\n{}\nmin angle = {}", directions.size(),
                      distances.str(), minAngle);

            bool needToAssert = false;

            vector<vect> supplePath;
            if (tryToConverge(stopStrategy, func, direction, r, supplePath, 10)) {
                LOG_ERROR("second optimization converged for {} steps", supplePath.size());
            } else {
                LOG_ERROR("second optimization did not converged with hessian update. Tryin standard optimization");
                supplePath = optimizeOnSphere(stopStrategy, func, direction, r, 50, 5);
                needToAssert = true;
            }

            path.insert(path.end(), supplePath.begin(), supplePath.end());
            auto oldDirection = direction;
            direction = path.back();
            LOG_ERROR("cos(oldDirection, direction) = {} after second optimization",
                      angleCosine(oldDirection, direction));

            logFunctionPolarInfo(func, direction, r, "normalized after additional optimization");

            distances = stringstream();
            minAngle = 0;
            for (auto const& prevDir : directions) {
                minAngle = max(minAngle, angleCosine(direction, prevDir));
                distances
                   << boost::format("[%1%, %2%]") % distance(direction, prevDir) % angleCosine(direction, prevDir);
            }
            LOG_ERROR("Distances from previous {} directons [dist, cos(angle)]:\n{}\nmin angle = {}", directions.size(),
                      distances.str(), minAngle);

            if (minAngle < .975) {
                values.push_back(sqr(r) / 2 - (func(direction) - zeroEnergy));

                directions.push_back(direction);

                ofstream mins("./mins_on_sphere");
                mins.precision(21);

                mins << directions.size() << endl;
                for (auto const& dir : directions) {
                    mins << dir.size() << endl << fixed << dir.transpose() << endl;
                }

                lastSuccessIter = iter;

                assert(!needToAssert);
            } else {
                LOG_ERROR("min angle is too large: {}", minAngle);
            }
        }
    }

    while (true) {
//        Cosine3OnSphereInterpolation supplement(normalized.nDims, values, directions);
        CleverCosine3OnSphereInterpolation supplement(func.nDims, values, directions);
        auto withSupplement = func + supplement;

        auto path = optimizeOnSphere(stopStrategy, withSupplement, r * randomVectOnSphere(func.nDims), r, 50, 5);
        auto direction = path.back();

        logFunctionPolarInfo(withSupplement, direction, r, "func in new direction");
        logFunctionPolarInfo(func, direction, r, "normalized in new direction");

        stringstream distances;
        double minAngle = 0;
        for (auto const& prevDir : directions) {
            minAngle = max(minAngle, angleCosine(direction, prevDir));
            distances << boost::format("[%1%, %2%]") % distance(direction, prevDir) % angleCosine(direction, prevDir);
        }
        LOG_ERROR("Distances from previous {} directons [dist, cos(angle)]:\n{}\nmin angle = {}", directions.size(),
                  distances.str(), minAngle);

        size_t const N = 15;
        auto directionMem = direction;

        for (size_t i = 0; i < N; i++) {
            double alpha = (double) (i + 1) / N;
            auto linearComb = alpha * func + (1 - alpha) * withSupplement;

            auto supplePath = optimizeOnSphere(stopStrategy, linearComb, direction, r, 50, 10);
            LOG_ERROR("experimental iteration {}: converged for {} steps", i + 1, supplePath.size());

            path.insert(path.end(), supplePath.begin(), supplePath.end());
            direction = path.back();
        }

        LOG_ERROR("Experimental convergence result:cos(angle) = {}", angleCosine(direction, directionMem));

        logFunctionPolarInfo(func, direction, r, "normalized after additional optimization");

        distances = stringstream();
        minAngle = 0;
        for (auto const& prevDir : directions) {
            minAngle = max(minAngle, angleCosine(direction, prevDir));
            distances << boost::format("[%1%, %2%]") % distance(direction, prevDir) % angleCosine(direction, prevDir);
        }
        LOG_ERROR("Distances from previous {} directons [dist, cos(angle)]:\n{}\nmin angle = {}", directions.size(),
                  distances.str(), minAngle);

        if (minAngle < .975) {
            values.push_back(sqr(r) / 2 - (func(direction) - zeroEnergy));
            directions.push_back(direction);

            ofstream mins("./mins_on_sphere");
            mins.precision(21);

            mins << directions.size() << endl;
            for (auto const& dir : directions) {
                mins << dir.size() << endl << fixed << dir.transpose() << endl;
            }

        } else {
            LOG_ERROR("min angle is too large: {}", minAngle);
        }
    }
}


template<typename FuncT>
void minimaBruteForce(FuncT&& func)
{
    func.getFullInnerFunction().setGaussianNProc(1);
    auto zeroEnergy = func(makeConstantVect(func.nDims, 0));

    double const r = .05;
    vector<double> values;
    vector<vect> directions;

    auto const axis = framework.newPlot();
    RandomProjection const projection(func.nDims);
    auto stopStrategy = makeHistoryStrategy(StopStrategy(1e-4 * r, 1e-4 * r));

    ofstream allMins("./all_mins_on_sphere");
    allMins.precision(21);

    #pragma omp parallel
    while (true) {
        auto path = optimizeOnSphere(stopStrategy, func, randomVectOnSphere(func.nDims, r), r, 50, 5);
        auto direction = path.back();

        stringstream distances;
        double minAngle = 0;
        for (auto const& prevDir : directions) {
            minAngle = max(minAngle, angleCosine(direction, prevDir));
            distances
               << boost::format("[%1%, %2%]") % distance(direction, prevDir) % angleCosine(direction, prevDir);
        }
        LOG_ERROR("Distances from previous {} directons [dist, cos(angle)]:\n{}\nmin angle = {}", directions.size(),
                  distances.str(), minAngle);

        #pragma omp critical
        {
            allMins << direction.size() << endl << fixed << direction.transpose() << endl;

            if (minAngle < .975) {
                directions.push_back(direction);

                ofstream mins("./mins_on_sphere");
                mins.precision(21);

                mins << directions.size() << endl;
                for (auto const& dir : directions) {
                    mins << dir.size() << endl << fixed << dir.transpose() << endl;
                }

            } else {
                LOG_ERROR("min angle is too large: {}", minAngle);
            }
        }
    }
}


template<typename FuncT>
void researchPaths(FuncT&& normalized)
{
    normalized.getFullInnerFunction().setGaussianNProc(1);

    RandomProjection projection(15);
    auto axis1 = framework.newPlot("true distance space");
    auto axis2 = framework.newPlot("false distance space");

//    for (size_t i = 0; i < 11; i++) {
    for (size_t i = 9; i <= 9; i++) {
        vector<vector<size_t>> charges;
        vector<vect> structures;

        tie(charges, structures) = readWholeChemcraft(ifstream(str(format("./results/%1%.xyz") % i)));

        {
            vector<double> xs, ys;
            for (auto structure : structures) {
                auto proj = projection(toDistanceSpace(structure, true));
                xs.push_back(proj(0));
                ys.push_back(proj(1));
            }

            framework.plot(axis1, xs, ys, to_string(i));
        }

        {
            vector<double> xs, ys;
            for (auto structure : structures) {
                auto proj = projection(toDistanceSpace(structure, false));
                xs.push_back(proj(0));
                ys.push_back(proj(1));
            }

            framework.plot(axis2, xs, ys, to_string(i));
        }
    }
    framework.legend(axis1);
    framework.legend(axis2);

    LOG_INFO("paths were built");

//    for (size_t i = 0; i < 11; i++) {
    for (size_t i = 9; i <= 9; i++) {
        vector<vector<size_t>> charges;
        vector<vect> structures;

        tie(charges, structures) = readWholeChemcraft(ifstream(str(format("./results/%1%.xyz") % i)));

        vector<double> values(charges.size());
        vector<vect> grads(charges.size());
        vector<matrix> hess(charges.size());

        vect prev_structure = structures[0];

        #pragma omp parallel for
        for (size_t j = 0; j < charges.size(); j++) {
            auto structure = structures[j];
            auto curCharges = charges[j];

            GaussianProducer molecule(curCharges, 1);

            auto valueGradHess = molecule.valueGradHess(structure);
            values[j] = get<0>(valueGradHess);
            grads[j] = get<1>(valueGradHess);
            hess[j] = get<2>(valueGradHess);
        }

        vector<double> gradNorms(charges.size());
        vector<double> dists(charges.size());
        vector<double> angles(charges.size());

        for (size_t j = 0; j < charges.size(); j++) {
            gradNorms[j] = grads[j].norm();
            dists[j] = j ? distance(structures[j - 1], structures[j]) : 0;
            angles[j] = j ? angleCosine(structures[j - 1], structures[j]) : 0;

            LOG_INFO("point #{}:\n\tpos = {}\n\tgrad = {} [{}]\n\thess = {}\n",
                     j, structures[j].transpose(), grads[j].norm(), grads[j].transpose(), singularValues(hess[j]));
        }

        dists[0] = dists[1];
        angles[0] = angles[1];

        framework.plot(framework.newPlot("values" + to_string(i)), values);
        framework.plot(framework.newPlot("grads" + to_string(i)), gradNorms);
        framework.plot(framework.newPlot("dists" + to_string(i)), dists);
        framework.plot(framework.newPlot("angles " + to_string(i)), angles);
    }
}

template<typename FuncT>
auto remove6LesserHessValuesOld(FuncT&& func, vect structure)
{
    auto hess = func.hess(structure);
    auto A = linearizationNormalization(hess, 6);

    size_t nDims = func.nDims;
    vector<size_t> poss = {nDims - 6, nDims - 5, nDims - 4, nDims - 3, nDims - 2, nDims - 1};
    vector<double> vals(poss.size(), 0.);
    return fix(makeAffineTransfomation(std::forward<FuncT>(func), structure, A), poss, vals);
}

template<typename FuncT>
auto remove6LesserHessValues(FuncT&& func, vect structure)
{
    size_t n = func.nDims / 3;
    vector<vect> vs;
    for (size_t i = 0; i < 3; i++) {
        vect v(structure.size());

        for (size_t j = 0; j < n; j++)
            v.block(j * 3, 0, 3, 1) = eye(3, i);
        vs.push_back(normalized(v));
    }

    for (size_t i = 0; i < 3; i++) {
        vect v(structure.size());

        for (size_t j = 0; j < n; j++) {
            vect block = structure.block(j * 3, 0, 3, 1);

            block(i) = 0;
            if (i == 0)
                swap(block(1), block(2)), block(1) *= -1;
            else if (i == 1)
                swap(block(0), block(2)), block(0) *= -1;
            else
                swap(block(0), block(1)), block(0) *= -1;

            v.block(j * 3, 0, 3, 1) = block;
        }

        vs.push_back(normalized(v));
    }

    matrix basis(func.nDims, vs.size());
    for (size_t i = 0; i < vs.size(); i++)
        basis.block(0, i, func.nDims, 1) = vs[i];

    for (size_t i = vs.size(); i < func.nDims; i++) {
        auto v = makeRandomVect(func.nDims);
        vect x = basis.colPivHouseholderQr().solve(v);

        v = v - basis * x;
        basis = horizontalStack(basis, normalized(v));
    }

    auto transformed = makeAffineTransfomation(func, structure,
                                               basis.block(0, vs.size(), basis.rows(), basis.cols() - vs.size()));
    auto hess = transformed.hess(makeConstantVect(transformed.nDims, 0.));
    auto A = linearizationNormalization(hess);

    return makeAffineTransfomation(transformed, A);
}

template<typename FuncT>
auto remove6LesserHessValues2(FuncT&& func, vect structure)
{
    size_t n = func.nDims / 3;
    vector<vect> vs;
    for (size_t i = 0; i < 3; i++) {
        vect v(structure.size());

        for (size_t j = 0; j < n; j++)
            v.block(j * 3, 0, 3, 1) = eye(3, i);
        vs.push_back(normalized(v));
    }

    for (size_t i = 0; i < 3; i++) {
        vect v(structure.size());

        for (size_t j = 0; j < n; j++) {
            vect block = structure.block(j * 3, 0, 3, 1);

            block(i) = 0;
            if (i == 0)
                swap(block(1), block(2)), block(1) *= -1;
            else if (i == 1)
                swap(block(0), block(2)), block(0) *= -1;
            else
                swap(block(0), block(1)), block(0) *= -1;

            v.block(j * 3, 0, 3, 1) = block;
        }

        vs.push_back(normalized(v));
    }

    matrix basis(func.nDims, vs.size());
    for (size_t i = 0; i < vs.size(); i++)
        basis.block(0, i, func.nDims, 1) = vs[i];

    for (size_t i = vs.size(); i < func.nDims; i++) {
        auto v = makeRandomVect(func.nDims);
        vect x = basis.colPivHouseholderQr().solve(v);

        v = v - basis * x;
        basis = horizontalStack(basis, normalized(v));
    }

    return makeAffineTransfomation(func, structure, basis.block(0, vs.size(), basis.rows(), basis.cols() - vs.size()));
}

tuple<bool, vect> tryToOptimizeTS(GaussianProducer& molecule, vect structure, size_t iters = 10)
{
    try {
        for (size_t j = 0; j < iters; j++) {
            //todo: double hessian calculation
            auto transformed = remove6LesserHessValues2(molecule, structure);
            auto valueGradHess = transformed.valueGradHess(makeConstantVect(transformed.nDims, 0));
            auto grad = get<1>(valueGradHess);
            auto hess = get<2>(valueGradHess);

            auto sValues = singularValues(hess);
            bool hasNegative;
            for (size_t i = 0; i < sValues.size(); i++)
                if (sValues(i) < 0)
                    hasNegative = true;

            if (!hasNegative) {
                LOG_INFO("TS structure condidat has no negative singular values");
                return make_tuple(false, vect());
            }

            structure = transformed.fullTransform(-get<2>(valueGradHess).inverse() * get<1>(valueGradHess));
        }

        auto transformed = remove6LesserHessValues2(molecule, structure);
        auto sValues = singularValues(transformed.hess(makeConstantVect(transformed.nDims, 0)));

        logFunctionInfo(molecule, structure, "final structure");
        LOG_INFO("final TS result xyz\n{}\n", toChemcraftCoords(molecule.getCharges(), structure));

        return make_tuple(true, structure);
    }
    catch (GaussianException const& exc) {
        LOG_INFO("ts optimization try breaked with exception");
        return make_tuple(false, vect());
    }
}

void optimizeInterestingTSs()
{
//    vector<size_t> interesting = {127, 139, 159};
    vector<size_t> interesting = {27};
//    vector<size_t> interesting = {139};
    vector<vector<size_t>> charges;
    vector<vect> structures;

    tie(charges, structures) = readWholeChemcraft(ifstream("./results/9.xyz"));

    for (size_t i : interesting) {
        auto structure = structures[i];

        GaussianProducer molecule(charges[i], 3);

        logFunctionInfo(molecule, structures[i], "");
        LOG_INFO("\n{}", toChemcraftCoords(charges[i], structures[i]));

        tryToOptimizeTS(molecule, structures[i]);
    }
}

matrix experimentalInverse(matrix const& m) {
    auto A = linearization(m);
    matrix diag = A.transpose() * m * A;

    for (size_t i = 0; i < diag.rows(); i++)
        if (diag(i, i) < 0 && false)
            diag(i, i) = 1;
        else
            diag(i, i) = 1 / abs(diag(i, i));

    return A * diag * A.transpose();
}

template<typename FuncT, typename StopStrategy>
bool experimentalTryToConverge(StopStrategy stopStrategy, FuncT& func, vect p, double r, vector<vect>& path, size_t iterLimit=5, size_t globalIter=0, bool needSingularTest=true)
{
    auto const theta = makeConstantVect(func.nDims - 1, M_PI / 2);
    bool converged = false;

    vector<vect> newPath;
    try {
        for (size_t i = 0; i < iterLimit; i++) {
            auto polar = makePolarWithDirection(func, r, p);

            auto valueGradHess = polar.valueGradHess(theta);
            auto value = get<0>(valueGradHess);
            auto grad = get<1>(valueGradHess);
            auto hess = get<2>(valueGradHess);

            if (needSingularTest) {
                auto sValues = singularValues(hess);
                for (size_t j = 0; j < sValues.size(); j++) {
                    if (sValues(j) < 0) {
                        LOG_INFO("singular values converge break, stop strategy with zero delta: {}",
                                 stopStrategy(globalIter + i, p, value, grad, hess, p - p));
                        return false;
                    }
                }
            }

            stringstream point;
            point.precision(13);
            for (size_t i = 0; i < p.size(); i++)
                point << fixed << p(i) << ", ";
            LOG_INFO("{}", point.str());

            auto lastP = p;
            p = polar.getInnerFunction().transform(polar.transform(theta - experimentalInverse(hess) * grad));
            newPath.push_back(p);

            if (stopStrategy(globalIter + i, p, value, grad, hess, p - lastP)) {
                converged = true;
                break;
            }
        }
    } catch (GaussianException const& exc) {
        LOG_ERROR("GaussianException converge break");
        return false;
    }

    if (converged) {
        path.insert(path.end(), newPath.begin(), newPath.end());
        return true;
    }

    return false;
};

bool shsTSTryRoutine(GaussianProducer& molecule, vect structure, ostream& output)
{
    vect ts;
    bool optimized;
    tie(optimized, ts) = tryToOptimizeTS(molecule, structure);

    if (optimized) {
        auto valueGradHess = molecule.valueGradHess(ts);
        auto grad = get<1>(valueGradHess);
        auto hess = get<2>(valueGradHess);

        LOG_CRITICAL("TS FOUND.\nTS gradient: {} [{}]\nsingularr hess values: {}\n{}", grad.norm(), print(grad),
                     singularValues(hess), toChemcraftCoords(molecule.getCharges(), ts), grad.norm());

        output << toChemcraftCoords(molecule.getCharges(), ts, "final TS");
        output.flush();


        return true;
    }

    return false;
}

template<typename FuncT>
void shs(FuncT&& func)
{
    auto& molecule = func.getFullInnerFunction();
    molecule.setGaussianNProc(1);
    logFunctionInfo(func, makeConstantVect(func.nDims, 0), "normalized energy for equil structure");

    ifstream minsOnSphere("./mins_on_sphere");
    size_t cnt;
    minsOnSphere >> cnt;
    vector<vect> vs;
    for (size_t i = 0; i < cnt; i++)
        vs.push_back(readVect(minsOnSphere));

    double const firstR = 0.05;
    double const deltaR = 0.04;
    size_t const CONV_ITER_LIMIT = 10;

    auto const stopStrategy = makeHistoryStrategy(StopStrategy(1e-8, 1e-5));

#pragma omp parallel for
    for (size_t i = 0; i < cnt; i++) {
//    for (size_t i = 9; i <= 9; i++) {
        vector<vect> trajectory;
        ofstream output(str(format("./results/%1%.xyz") % i));

        auto direction = vs[i];
        LOG_INFO("Path #{}. Initial direction: {}", i, direction.transpose());

        vect lastPoint = func.fullTransform(makeConstantVect(func.nDims, 0));
        double value = func(direction);
        double r = firstR;

        for (size_t step = 0; step < 600; step++) {
            if (!step) {
                auto polar = makePolarWithDirection(func, .1, direction);
                logFunctionInfo(polar, makeConstantVect(polar.nDims, M_PI / 2),
                                str(boost::format("Paht %1% initial direction info") % i));
            }

            if (step && step % 7 == 0 && shsTSTryRoutine(molecule, lastPoint, output)) {
                LOG_INFO("Path #{} TS found. Break", i);
                break;
            }

            vect prev = direction;
            direction = direction / direction.norm() * (r + deltaR);

            bool converged = false;
            bool tsFound = false;
            double currentDr = deltaR;

            for (size_t convIter = 0; convIter < CONV_ITER_LIMIT; convIter++, currentDr *= 0.5) {
                double nextR = r + currentDr;
                vector<vect> path;
                if (experimentalTryToConverge(stopStrategy, func, direction, nextR, path, 30, 0, false)) {
                    if (angleCosine(direction, path.back()) < .9) {
                        LOG_ERROR("Path {} did not converge (too large angle: {})", i, angleCosine(direction, path.back()));
                        continue;
                    }

                    LOG_ERROR("CONVERGED with dr = {}\nnew direction = {}\nangle = {}", currentDr, print(path.back(), 17), angleCosine(direction, path.back()));
                    LOG_INFO("Path #{} converged with delta r {}", i, deltaR);

                    r += currentDr;
                    direction = path.back();

                    converged = true;
                    break;
                }
                else if (shsTSTryRoutine(molecule, lastPoint, output)) {
                    tsFound = true;
                    converged = true;
                    break;
                }
            }

            if (tsFound) {
                LOG_INFO("Path #{} TS found. Break", i);
                break;
            }

            if (!converged) {
                LOG_ERROR("Path #{} exceeded converge iteration limit ({}). Break", i, CONV_ITER_LIMIT);
                break;
            }

            double newValue = func(direction);
            LOG_INFO("New {} point in path {}:\n\tvalue = {:.13f}\n\tdelta angle cosine = {:.13f}\n\tdirection: {}", step,
                     i, newValue, angleCosine(direction, prev), direction.transpose());

            lastPoint = func.fullTransform(direction);
            output << toChemcraftCoords(molecule.getCharges(), lastPoint, to_string(step));
            output.flush();

            if (newValue < value) {
                LOG_ERROR("newValue < value [{:.13f} < {:.13f}]. Stopping", newValue, value);
            }
            value = newValue;
        }
    }
}

void goDown(GaussianProducer& molecule, vect structure, string const& filename) {
    ofstream output(filename);

    vector<double> values;
    vector<double> gradNorms;

    for (size_t step = 0; step < 200; step++){
        auto fixed = remove6LesserHessValues2(molecule, structure);
        auto valueGrad = fixed.valueGrad(makeConstantVect(fixed.nDims, 0.));
        auto value = get<0>(valueGrad);
        auto grad = get<1>(valueGrad);

        values.push_back(value);
        gradNorms.push_back(grad.norm());

        LOG_INFO("step #{}\nvalue = {}\ngrad = {} [{}]", step, value, grad.norm(), print(grad));

        structure = fixed.fullTransform(-grad * .3);
        output << toChemcraftCoords(molecule.getCharges(), structure, to_string(step));
        output.flush();
    }

    framework.plot(framework.newPlot("values"), values);
    framework.plot(framework.newPlot("gradient norms"), gradNorms);
}

void twoWayTS()
{
    vector<vect> structures;
    vector<vector<size_t>> _charges;
    tie(_charges, structures) = readWholeChemcraft(ifstream("./results/6.xyz"));

    auto charges = _charges.back();
    auto structure = structures.back();

    GaussianProducer molecule(charges, 3);
    auto fixed = remove6LesserHessValues2(molecule, structure);

    auto hess = fixed.hess(makeConstantVect(fixed.nDims, 0));
    auto A = linearization(hess);

    double const FACTOR = .01;

    for (size_t i = 0; i < A.cols(); i++) {
        vect v = A.col(i);

        double value = v.transpose() * hess * v;
        if (value < 0) {
            auto first = fixed.fullTransform(-FACTOR * v);
            auto second = fixed.fullTransform(FACTOR * v);

            LOG_INFO("{} < 0:\n{}\n\n{}\n\n", value,
                     toChemcraftCoords(charges, first, "first"),
                     toChemcraftCoords(charges, second, "second"));
            ofstream output("current.xyz");
            output << toChemcraftCoords(charges, first, "first")
                   << toChemcraftCoords(charges, second, "second");

            goDown(molecule, first, "first.xyz");
            goDown(molecule, second, "second.xyz");
        }
    }
}

int main()
{
    initializeLogger();

    ifstream C2H4("./C2H4_2");
    vector<size_t> charges;
    vect equilStruct;
    tie(charges, equilStruct) = readChemcraft(ifstream("./C2H4_2"));

    auto center = centerOfMass(charges, fromCartesianToPositions(equilStruct));
    for (size_t i = 0; i < equilStruct.size(); i += 3)
        equilStruct.block(i, 0, 3, 1) -= center;
    auto const stopStrategy = makeHistoryStrategy(StopStrategy(5e-8, 5e-8));

    auto molecule = GaussianProducer(charges, 3);

    minimaBruteForce(remove6LesserHessValues(molecule, equilStruct));
//    shs(remove6LesserHessValues(molecule, equilStruct));
//    minimaElimination(remove6LesserHessValues(molecule, equilStruct));
//    researchPaths(remove6LesserHessValues(molecule, equilStruct));
//    optimizeInterestingTSs();
//    return 0;


    return 0;
}