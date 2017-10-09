#pragma once

#include "helper.h"

#include <boost/format.hpp>

#include "linearAlgebraUtils.h"
#include "functionLoggers.h"
#include "producers/GaussianProducer.h"
#include "normalCoordinates.h"
#include "inputOutputUtils.h"
#include "optimization/optimizations.h"

using namespace optimization;

//todo: maybe remove inline
inline matrix experimentalInverse(matrix const& m) {
    auto A = linearization(m);
    matrix diag = A.transpose() * m * A;

    for (size_t i = 0; i < diag.rows(); i++)
        if (diag(i, i) < 0 && false)
            diag(i, i) = 1;
        else
            diag(i, i) = 1 / abs(diag(i, i));

    return A * diag * A.transpose();
}

template<typename StopStrategyT>
optional<vect> secondOrderStructureOptimization(StopStrategyT stopStrategy, GaussianProducer& molecule, vect structure, size_t iterLimit)
{
    for (size_t iter = 0; iter != iterLimit; ++iter) {
        auto fixed = remove6LesserHessValues2(molecule, structure);
        auto valueGradHess = fixed.valueGradHess(makeConstantVect(fixed.nDims, 0.));

        auto value = get<0>(valueGradHess);
        auto grad = get<1>(valueGradHess);
        auto hess = get<2>(valueGradHess);

        auto memStruct = structure;
        structure = fixed.fullTransform(-hess.inverse() * grad);

        if (stopStrategy(iter, structure, value, grad, hess, structure - memStruct))
            return make_optional(structure);
    }

    return boost::none;
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

optional<vect> tryToOptimizeTS(GaussianProducer& molecule, vect structure, size_t iters = 10)
{
    try {
        auto stopStrategy = makeHistoryStrategy(StopStrategy(1e-4, 1e-4));
        //todo: think about singular values check on each iteration of optimization

        auto optimized = secondOrderStructureOptimization(stopStrategy, molecule, structure, 10);
        if (!optimized)
            return boost::none;

        structure = *optimized;

        auto transformed = remove6LesserHessValues2(molecule, structure);
        auto sValues = singularValues(transformed.hess(makeConstantVect(transformed.nDims, 0)));
        bool hasNegative = false;
        for (size_t i = 0; i < sValues.size(); i++)
            if (sValues(i) < 0)
                hasNegative = true;

        if (!hasNegative) {
            LOG_INFO("no negative singular values after ts optimization");
            return boost::none;
        }

        logFunctionInfo(molecule, structure, "final TS structure info");
        LOG_INFO("final TS result xyz\n{}\n", toChemcraftCoords(molecule.getCharges(), structure));

        return make_optional(structure);
    }
    catch (GaussianException const& exc) {
        LOG_INFO("ts optimization try breaked with exception");
        return boost::none;
    }
}


bool shsTSTryRoutine(GaussianProducer& molecule, vect const& structure, ostream& output)
{
    if (auto ts = tryToOptimizeTS(molecule, structure)) {
        auto valueGradHess = molecule.valueGradHess(*ts);
        auto grad = get<1>(valueGradHess);
        auto hess = get<2>(valueGradHess);

        LOG_CRITICAL("TS FOUND.\nTS gradient: {} [{}]\nsingularr hess values: {}\n{}", grad.norm(), print(grad),
                     singularValues(hess), toChemcraftCoords(molecule.getCharges(), *ts), grad.norm());

        output << toChemcraftCoords(molecule.getCharges(), *ts, "final TS");
        output.flush();

        return true;
    }

    return false;
}


template<typename FuncT>
tuple<vector<vect>, optional<vect>> shsPath(FuncT&& func, vect direction, size_t pathNumber, double deltaR, size_t convIterLimit)
{
    auto& molecule = func.getFullInnerFunction();
    LOG_INFO("Path #{}. R0 = {}. Initial direction: {}", pathNumber, direction.norm(), direction.transpose());

    vect transitionState;
    vector<vect> trajectory;
    ofstream output(boost::str(boost::format("./results/%1%.xyz") % pathNumber));

    vect lastPoint = func.fullTransform(makeConstantVect(func.nDims, 0));
    trajectory.push_back(lastPoint);

    double value = func(direction);
    double r = direction.norm();

    auto const stopStrategy = makeHistoryStrategy(StopStrategy(1e-8, 1e-5));

    for (size_t step = 0; step < 600; step++) {
        if (!step) {
            logFunctionPolarInfo(func, direction, r, boost::str(boost::format("Paht %1% initial direction info") % pathNumber));
        }

        if (step && step % 7 == 0 && shsTSTryRoutine(molecule, lastPoint, output)) {
            LOG_INFO("Path #{} TS found. Break", pathNumber);
            break;
        }

        vect prev = direction;
        direction = direction / direction.norm() * (r + deltaR);

        bool converged = false;
        bool tsFound = false;
        double currentDr = min(direction.norm(), deltaR);

        for (size_t convIter = 0; convIter < convIterLimit; convIter++, currentDr *= 0.5) {
            double nextR = r + currentDr;
            vector<vect> path;
            if (experimentalTryToConverge(stopStrategy, func, direction, nextR, path, 30, 0, false)) {
                if (angleCosine(direction, path.back()) < .9) {
                    LOG_ERROR("Path {} did not converge (too large angle: {})", pathNumber, angleCosine(direction, path.back()));
                    continue;
                }

                LOG_ERROR("CONVERGED with dr = {}\nnew direction = {}\nangle = {}", currentDr, print(path.back(), 17), angleCosine(direction, path.back()));
                LOG_INFO("Path #{} converged with delta r {}", pathNumber, currentDr);

                r += currentDr;
                direction = path.back();

                converged = true;
                break;
            }
            else  {
                LOG_INFO("Path {} did not converged with", pathNumber);
                if (convIter == 0 && shsTSTryRoutine(molecule, lastPoint, output)) {
                    tsFound = true;
                    converged = true;
                    break;
                }
            }
        }

        if (tsFound) {
            LOG_INFO("Path #{} TS found. Break", pathNumber);
            break;
        }

        if (!converged) {
            LOG_ERROR("Path #{} exceeded converge iteration limit ({}). Break", pathNumber, convIterLimit);
            break;
        }

        double newValue = func(direction);
        LOG_INFO("New {} point in path {}:\n\tvalue = {:.13f}\n\tdelta angle cosine = {:.13f}\n\tdirection: {}", step,
                 pathNumber, newValue, angleCosine(direction, prev), direction.transpose());

        lastPoint = func.fullTransform(direction);
        output << toChemcraftCoords(molecule.getCharges(), lastPoint, to_string(step));
        output.flush();

        if (newValue < value) {
            LOG_ERROR("newValue < value [{:.13f} < {:.13f}]. Stopping", newValue, value);
        }
        value = newValue;
    }

    return make_tuple(trajectory, vect());
};

template<typename FuncT>
void shs(FuncT&& func)
{
    auto& molecule = func.getFullInnerFunction();
    molecule.setGaussianNProc(3);
    logFunctionInfo(func, makeConstantVect(func.nDims, 0), "normalized energy for equil structure");

    ifstream minsOnSphere("./mins_on_sphere");
    size_t cnt;
    minsOnSphere >> cnt;
    vector<vect> directions;
    for (size_t i = 0; i < cnt; i++)
        directions.push_back(readVect(minsOnSphere));

    double const DELTA_R = 0.04;
    size_t const CONV_ITER_LIMIT = 10;

#pragma omp parallel for
//    for (size_t i = 0; i < cnt; i++) {
    for (size_t i = 0; i <= 0; i++) {
        shsPath(func, directions[i], i, DELTA_R, CONV_ITER_LIMIT);
    }
}

template<typename FuncT>
vector<vect> minimaElimination(FuncT&& func)
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

    return directions;

    while (true) {
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

tuple<vector<vect>, optional<vect>> goDown(GaussianProducer& molecule, vect structure) {
    vector<vect> path;
    for (size_t step = 0; step < 100; step++){
        auto fixed = remove6LesserHessValues2(molecule, structure);
        auto valueGrad = fixed.valueGrad(makeConstantVect(fixed.nDims, 0.));
        auto value = get<0>(valueGrad);
        auto grad = get<1>(valueGrad);

        LOG_INFO("step #{}\nvalue = {}\ngrad = {} [{}]", step, value, grad.norm(), print(grad));

        structure = fixed.fullTransform(-grad * .3);
        path.push_back(structure);
    }

    bool converged = false;
    try {
        auto stopStrategy = makeHistoryStrategy(StopStrategy(1e-4, 1e-4));
        auto optimized = secondOrderStructureOptimization(stopStrategy, molecule, structure, 10);

        return make_tuple(path, optimized);
    }
    catch (GaussianException const& exc) {
        return make_tuple(path, vect());
    }
}


tuple<vector<vect>, optional<vect>, optional<vect>> twoWayTS(GaussianProducer& molecule, vect const& structure)
{
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

            optional<vect> firstES, secondES;
            vector<vect> firstPath, secondPath;

            tie(firstPath, firstES) = goDown(molecule, first);
            tie(secondPath, secondES) = goDown(molecule, second);

            reverse(firstPath.begin(), firstPath.end());
            firstPath.insert(firstPath.end(), secondPath.begin(), secondPath.end());

            return make_tuple(firstPath, firstES, secondES);
        }
    }
}


struct StructureSet
{
public:
    explicit StructureSet(double distSpaceEps) : mDistSpaceEps(distSpaceEps)
    {  }

    bool addStructure(vect const& structure)
    {
        auto distSpace = toDistanceSpace(structure);
        for (auto const& other : mStructs)
            if (distance(toDistanceSpace(other), distSpace) < mDistSpaceEps)
                return false;

        mStructs.push_back(structure);

        return true;
    }

private:
    double mDistSpaceEps;
    vector<vect> mStructs;
};

void addToSetAndQueu(StructureSet& set, queue<vect>& que, vect const& structure)
{
    if (set.addStructure(structure))
        que.push(structure);
}

void workflow(GaussianProducer& molecule, vect const& initialStruct, double deltaR, size_t iterLimit)
{
    system("mkdir -p info_logs");
    vector<spdlog::sink_ptr> sinks = {make_shared<spdlog::sinks::daily_file_sink_st>("info_logs/log", 0, 0)};
    auto infoLogger = make_shared<spdlog::logger>("info_logger", sinks.begin(), sinks.end());
    infoLogger->set_pattern("[%H:%M:%S %t] %v");
    infoLogger->set_error_handler([](string const& msg) { throw spdlog::spdlog_ex(msg); });
    infoLogger->set_level(spdlog::level::debug);


    StructureSet uniqueESs(1e-3);
    StructureSet uniqueTSs(1e-3);

    queue<vect> que;

    addToSetAndQueu(uniqueESs, que, initialStruct);

    size_t shsPathCounter = 0;
    while (!que.empty()) {
        auto equilStruct = que.front();
        que.pop();

        auto valueGradHess = molecule.valueGradHess(initialStruct);
        auto value = get<0>(valueGradHess);
        auto grad = get<1>(valueGradHess);
        auto hess = get<2>(valueGradHess);

        infoLogger->info("Initial equilibrium structure:\n\tvalue = {}\n\tgrad = {} [{}]\n\thess values = {}\nchemcraft coords:\n{}",
            value, grad.norm(), print(grad), singularValues(hess), toChemcraftCoords(molecule.getCharges(), initialStruct));
        infoLogger->flush();

        auto inNormalCoords = remove6LesserHessValues(molecule, equilStruct);

        auto minimaDirections = minimaElimination(inNormalCoords);

        stringstream minimas;
        for (auto const& direction : minimaDirections) {
            minimas << print(direction) << endl;
        }
        infoLogger->info("Found {} minima directions:\n{}", minimaDirections.size(), minimas.str());

        for (size_t i = 0; i < minimaDirections.size(); i++) {
            vector<vect> path;
            optional<vect> ts;
            tie(path, ts) = shsPath(inNormalCoords, minimaDirections[i], shsPathCounter + i, deltaR, iterLimit);

            if (ts) {
                uniqueTSs.addStructure(*ts);

                vector<vect> pathFromTS;
                optional<vect> firstEQ, secondEQ;
                tie(pathFromTS, firstEQ, secondEQ) = twoWayTS(molecule, *ts);

                if (firstEQ) {
                    addToSetAndQueu(uniqueESs, que, *firstES);
                    infoLogger->info("Found new ES:{}\nchemcraft:\n{}", print(*firstES), toChemcraftCoords(molecule.getCharges(), *firstES));


                }
                if (secondEQ)
                    addToSetAndQueu(uniqueESs, que, *secondES);
            }
        }
    }
}