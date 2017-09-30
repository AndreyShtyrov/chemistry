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
tuple<vector<vect>, vect> shsPath(FuncT&& func, vect direction, size_t pathNumber, double deltaR, size_t convIterLimit)
{
    auto& molecule = func.getFullInnerFunction();
    LOG_INFO("Path #{}. R0 = {}. Initial direction: {}", pathNumber, direction.norm(), direction.transpose());

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
                LOG_INFO("Path #{} converged with delta r {}", pathNumber, deltaR);

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
};

template<typename FuncT>
void shs(FuncT&& func)
{
    auto& molecule = func.getFullInnerFunction();
    molecule.setGaussianNProc(1);
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
    for (size_t i = 0; i < cnt; i++) {
//    for (size_t i = 9; i <= 9; i++) {
        shsPath(func, directions[i], i, DELTA_R, CONV_ITER_LIMIT);
    }
}