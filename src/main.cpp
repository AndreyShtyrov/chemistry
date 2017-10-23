#include "helper.h"

#include <boost/format.hpp>
#include <boost/lexical_cast.hpp>
#include <Eigen/LU>

#include "linearAlgebraUtils.h"
#include "producers/producers.h"
#include "PythongraphicsFramework.h"

#include "optimization/optimizations.h"
#include "optimization/optimizeOnSphere.h"
#include "inputOutputUtils.h"
#include "constants.h"
#include "functionLoggers.h"
#include "normalCoordinates.h"
#include "shsWorkflow.h"

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
            tie(charges, state) = readChemcraft(ifstream(format("./{}/{}.xyz", i, j)));
            auto cur = proj(toDistanceSpace(state, false));
            xs.push_back(cur(0));
            ys.push_back(cur(1));
        }

        framework.plot(axis, xs, ys);
    }
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
void minimaBruteForce(FuncT&& func)
{
    func.getFullInnerFunction().setGaussianNProc(1);
    auto zeroEnergy = func(makeConstantVect(func.nDims, 0));

    double const r = .01;
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
    for (size_t i = 0; i <= 0; i++) {
        vector<vector<size_t>> charges;
        vector<vect> structures;

        tie(charges, structures) = readWholeChemcraft(ifstream(format("./results/{}.xyz", i)));

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
    for (size_t i = 0; i <= 0; i++) {
        vector<vector<size_t>> charges;
        vector<vect> structures;

        tie(charges, structures) = readWholeChemcraft(ifstream(format("./results/{}.xyz", i)));

        vector<double> values(charges.size());
        vector<vect> grads(charges.size());
        vector<vect> coordGrads(charges.size());
        vector<matrix> hess(charges.size());

        vect prev_structure = structures[0];

        #pragma omp parallel for
        for (size_t j = 0; j < charges.size(); j++) {
            auto structure = structures[j];
            auto curCharges = charges[j];

            GaussianProducer molecule(curCharges, 3);

            auto valueGradHess = molecule.valueGradHess(structure);
            values[j] = get<0>(valueGradHess);
            grads[j] = get<1>(valueGradHess);
            hess[j] = get<2>(valueGradHess);

            coordGrads[j] = normalized.getBasis().transpose() * normalized.getInnerFunction().getBasis().transpose() * get<1>(valueGradHess);
        }

        vector<double> gradNorms(charges.size());
        vector<double> coordGradNorms(charges.size());
        vector<double> dists(charges.size());
        vector<double> angles(charges.size());

        for (size_t j = 0; j < charges.size(); j++) {
            gradNorms[j] = grads[j].norm();
            coordGradNorms[j] = coordGrads[j].norm();
            dists[j] = j ? distance(structures[j - 1], structures[j]) : 0;
            angles[j] = j ? angleCosine(structures[j - 1], structures[j]) : 0;

            LOG_INFO("point #{}:\n\tpos = {}\n\tgrad = {} [{}]\n\thess = {}\n",
                     j, structures[j].transpose(), grads[j].norm(), grads[j].transpose(), singularValues(hess[j]));
        }

        dists[0] = dists[1];
        angles[0] = angles[1];

        framework.plot(framework.newPlot("values" + to_string(i)), values);
        framework.plot(framework.newPlot("grads" + to_string(i)), gradNorms);
        framework.plot(framework.newPlot("coord grads" + to_string(i)), coordGradNorms);
        framework.plot(framework.newPlot("dists" + to_string(i)), dists);
        framework.plot(framework.newPlot("angles " + to_string(i)), angles);
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

void explorPathTS(vector<size_t> numbers)
{
    vector<size_t> memCharges;
    vector<vect> equilStructures;

    for (size_t i : numbers) {
        vector<vect> structures;
        vector<vector<size_t>> _charges;
        tie(_charges, structures) = readWholeChemcraft(ifstream(format("./result_C2H4/{}.xyz", i)));

        auto charges = _charges.back();
        auto structure = structures.back();

        GaussianProducer molecule(charges, 3);

        vector<vect> path;
        optional<vect> startES, endES;

        tie(path, startES, endES) = twoWayTS(molecule, structure);
        if (startES)
            equilStructures.push_back(*startES);
        if (endES)
            equilStructures.push_back(*endES);

        ofstream output(format("./result_paths/{}.xyz", i));
        if (startES)
            output << toChemcraftCoords(charges, *startES, "start ES");
        for (size_t j = 0; j < path.size(); j++)
            output << toChemcraftCoords(charges, path[j], to_string(j));
        if (endES)
            output << toChemcraftCoords(charges, *endES, "end ES");

        memCharges = molecule.getCharges();
    }

    stringstream dists1;
    dists1.precision(5);
    for (auto const& struct1 : equilStructures) {
        for (auto const& struct2 : equilStructures)
            dists1 << fixed << distance(struct1, struct2) << ' ';
        dists1 << endl;
    }

    stringstream dists2;
    dists2.precision(5);
    for (auto const& struct1 : equilStructures) {
        for (auto const& struct2 : equilStructures)
            dists2 << fixed << distance(toDistanceSpace(struct1), toDistanceSpace(struct2)) << ' ';
        dists2 << endl;
    }

    LOG_INFO("\ndists1:\n{}\n\ndits2:\n{}\n", dists1.str(), dists2.str());

    for (auto const& equilStructure : equilStructures)
        LOG_INFO("\n{}", toChemcraftCoords(memCharges, equilStructure));
}

int main()
{
    initializeLogger();

//    {
//        vector<vect> structs;
//        vector<vector<size_t>> _charges;
//        tie(_charges, structs) = readWholeChemcraft(ifstream("./test.xyz"));
//        auto charges = _charges[0];
//        GaussianProducer molecule(charges, 3);
//
//        vector<double> values;
//        vector<double> gradNorms;
//
//        for (size_t i = 0; i + 1 < structs.size(); i++) {
//            auto from = structs[i];
//            auto to = structs[i + 1];
//
//            auto angle = i ? angleCosine(from - structs[i - 1], to - from) : 0;
//
//            auto dist1 = distance(from, to);
//
////            auto removed = remove6LesserHessValues(molecule, from);
////            auto hess = removed.hess(makeConstantVect(removed.nDims, 0.));
//
//            auto normalized = remove6LesserHessValues(molecule, from);
//            auto fromNorm = normalized.backTransform(from);
//            auto toNorm = normalized.backTransform(to);
//
//            auto valueGradHess = normalized.valueGradHess(makeConstantVect(normalized.nDims, 0.));
//            SecondOrderFunction supplement(get<0>(valueGradHess), get<1>(valueGradHess), get<2>(valueGradHess));
//            auto func = normalized - supplement;
//
////            auto onSphere = makePolarWithDirection(normalized, 0.05, toNorm - fromNorm);
//
//            auto dist2 = distance(from, to);
//
//            cerr << print(toNorm) << endl;
//
//            auto dir = ::normalized(toNorm - fromNorm);
//            auto grad = normalized.grad(toNorm);
//            double proj = grad.dot(dir);
//            double nProj = sqrt(grad.norm() * grad.norm() - proj * proj);
//
//            auto grad2 = func.grad(toNorm);
//            double proj2 = grad2.dot(dir);
//            double nProj2 = sqrt(grad2.norm() * grad2.norm() - proj2 * proj2);
//
//            LOG_INFO("{}: angle = {:.7f}, dist1 = {:.7f}, dist2 = {:.7f}\ngrad info: len = {} proj = {} n_proj = {}\ngrad info: len = {} proj = {} n_proj = {}\n",
//                     i, angle, dist1, dist2, grad.norm(), proj, nProj, grad2.norm(), proj2, nProj2);
//
//            logFunctionPolarInfo(func, toNorm - fromNorm, 0.05);
//        }
//
//        for (auto const& structure : structs) {
//            auto valueGrad = molecule.valueGrad(structure);
//
//            values.push_back(get<0>(valueGrad));
//            gradNorms.push_back(get<1>(valueGrad).norm());
//        }
//
//        framework.plot(framework.newPlot(), values);
//        framework.plot(framework.newPlot(), gradNorms);
//
//        return 0;
//    }


    ifstream C2H4("./C2H4");
    auto _charges = readCharges(C2H4);
    auto equilStruct = readVect(C2H4);

    auto molecule = GaussianProducer(_charges, 3);
    logFunctionInfo(molecule, equilStruct, "equil struct");
    auto normalized = remove6LesserHessValues(molecule, equilStruct);


    ifstream minsOnSphere("./mins_on_sphere");
    size_t cnt;
    minsOnSphere >> cnt;
    vector<vect> vs;
    for (size_t i = 0; i < cnt; i++) {
        vs.push_back(readVect(minsOnSphere));
        logFunctionPolarInfo(normalized, vs.back(), 0.01, "polar normalized in initial direction");
    }

//    double const firstR = 0.05;
    double const deltaR = 0.01;
//    double const R = .01;

    auto const stopStrategy = makeHistoryStrategy(StopStrategy(1e-5, 1e-5));

    auto direction = vs.back();
    vect pos = equilStruct;
    vector<vect> path;
    path.push_back(pos);

    for (size_t j = 0; j < 600; j++) {
        auto normalized2 = remove6LesserHessValues(molecule, pos);
        auto valueGradHess = normalized2.valueGradHess(makeConstantVect(normalized2.nDims, 0.));
        SecondOrderFunction supplement(get<0>(valueGradHess), get<1>(valueGradHess), get<2>(valueGradHess));
        auto func = normalized2 - supplement;

        logFunctionInfo(molecule, pos, "molecule in pos");

//        if (!j) {
            logFunctionPolarInfo(normalized2, direction, deltaR, "polar normalized2 in direction");
//            logFunctionPolarInfo(supplement, direction, deltaR, "supplement");
//            logFunctionPolarInfo(func, direction, deltaR, "func");
//        }

        if (j) {
            vect guess = path[path.size() - 1] * 2 - path[path.size() - 2];
            vect dir = normalized.backTransform(guess);
            vect normalizedDir = dir / dir.norm();
            LOG_INFO("\n{}", dir.transpose());

//            direction = optimizeOnSphere(stopStrategy, func, normalizedDir * deltaR, deltaR, 50, 10).back();

            bool converged = false;
            for (double curR = deltaR;; curR *= .75) {
//                vector<vect> optimizationPath;

                auto optimizationPath = optimizeOnSphere(stopStrategy, func, normalizedDir * curR, curR, 100, 5);
                if (!optimizationPath.empty()) {
//                if (experimentalTryToConverge(stopStrategy, func, normalizedDir * curR, curR, optimizationPath, 30, 0, false)) {
//                if (tryToConverge(stopStrategy, func, normalizedDir * curR, curR, optimizationPath, 10)) {
                    LOG_INFO("Converged with r = {}", curR);
                    direction = optimizationPath.back();

                    break;
                } else {
                    LOG_INFO("Did not converged with r = {}", curR);
                }
            }
        }

        LOG_INFO("direction norm = {}", direction.norm());
        pos = normalized2.fullTransform(direction);
        path.push_back(pos);

        LOG_INFO("delta dist = {}", distance(path[path.size() - 1], path[path.size() - 2]));

        {
            ofstream output("./test.xyz");
            for (size_t j = 0; j < path.size(); j++)
                output << toChemcraftCoords(normalized.getFullInnerFunction().getCharges(), path[j], to_string(j));
        }
    }

    return 0;
}