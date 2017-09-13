#pragma once

#include "helper.h"

#include "linearAlgebraUtils.h"
#include "producers/AffineTransformation.h"
#include "producers/GaussianProducer.h"
#include "producers/InPolar.h"

namespace optimization
{
    template<typename FuncT, typename StopStrategy>
    bool tryToConverge(StopStrategy stopStrategy, FuncT& func, vect p, double r, vector<vect>& path, size_t iterLimit=5, size_t globalIter=0)
    {
        auto const theta = makeConstantVect(func.nDims - 1, M_PI / 2);
        bool converged = false;

        vector<vect> newPath;
//        try {
            for (size_t i = 0; i < iterLimit; i++) {
                auto polar = makePolarWithDirection(func, r, p);

                auto valueGradHess = polar.valueGradHess(theta);
                auto value = get<0>(valueGradHess);
                auto grad = get<1>(valueGradHess);
                auto hess = get<2>(valueGradHess);

//                auto sValues = singularValues(hess);
//                for (size_t j = 0; j < sValues.size(); j++) {
//                    if (sValues(j) < 0) {
//                        LOG_INFO("singular values converge break, stop strategy with zero delta: {}",
//                                 stopStrategy(globalIter + i, p, value, grad, hess, p - p));
//                        return false;
//                    }
//                }

                auto lastP = p;
                p = polar.getInnerFunction().transform(polar.transform(theta - hess.inverse() * grad));
                newPath.push_back(p);

                if (stopStrategy(globalIter + i, p, value, grad, hess, p - lastP)) {
                    converged = true;
                    break;
                }
            }
//        } catch (GaussianException const& exc) {
//            LOG_ERROR("GaussianException converge break");
//        }

        if (converged) {
            path.insert(path.end(), newPath.begin(), newPath.end());
            return true;
        }

        return false;
    };

    template<typename FuncT, typename StopStrategy>
    vector<vect> optimizeOnSphere(StopStrategy stopStrategy, FuncT& func, vect p, double r, size_t preHessIters, size_t convergeIters)
    {
        LOG_INFO("{} {}", r, p.norm());
        assert(abs(r - p.norm()) < 1e-7);

        auto const theta = makeConstantVect(func.nDims - 1, M_PI / 2);

        vector<vect> path;
        vect momentum;

        vector<RandomProjection> projs;
        vector<vector<double>> xss, yss;

        for (size_t i = 0; i < 3; i++) {
            projs.push_back(RandomProjection(func.nDims));
        }

        for (size_t iter = 0; ; iter++) {
            if (iter % preHessIters == 0 && tryToConverge(stopStrategy, func, p, r, path, convergeIters, iter)) {
                break;
            }

            auto polar = makePolarWithDirection(func, r, p);

            auto valueGrad = polar.valueGrad(theta);
            auto value = get<0>(valueGrad);
            vect grad = get<1>(valueGrad);

            if (iter) {
                double was = momentum.norm();
                double factor = sqrt(max(0., angleCosine(momentum, grad)));
                momentum = factor * momentum + grad / r;
                LOG_INFO("was: {}, factor: {}, now: {} [delta: {}]", was, factor, momentum.norm(), grad.norm());
            }
            else
                momentum = grad / r;

            auto lastP = p;
            p = polar.getInnerFunction().transform(polar.transform(theta - momentum));
            path.push_back(p);

//            if (stopStrategy(iter, p, value, grad, momentum))
//                break;
            stopStrategy(iter, p, value, grad, momentum);

            if (iter % 25 == 0) {
                xss.clear();
                yss.clear();

                for (size_t i = 0; i < 3; i++) {
                    vector<double> xs, ys;
                    for (auto const& p : path) {
                        auto projected = projs[i](p);
                        xs.push_back(projected(0));
                        ys.push_back(projected(1));
                    }

                    xss.push_back(xs);
                    yss.push_back(ys);
                }

                framework = PythongraphicsFramework("func.out");

                auto axis = framework.newPlot();
                for (size_t i = 0; i < 3; i++)
                    framework.plot(axis, xss[i], yss[i]);

                axis = framework.newPlot();
                for (size_t i = 0; i < 3; i++) {
                    framework.plot(axis, xss[i], yss[i]);
                    framework.scatter(axis, xss[i], yss[i]);
                }
            }
        }

        LOG_INFO("converged for {} steps", path.size());

        return path;
    }

    template<typename FuncT, typename StopStrategy>
    vector<vect> optimizeOnSphere_old(StopStrategy stopStrategy, FuncT& func, vect p, double r, size_t preHessIters)
    {
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
                grad = 10 * get<1>(valueGrad);
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
    }


    template<typename FuncT, typename StopStrategy>
    vector<vect> optimizeOnSphere2(StopStrategy stopStrategy, FuncT& func, vect p, double r, size_t preHessIters)
    {
        assert(abs(r - p.norm()) < 1e-7);

        auto const theta = makeConstantVect(func.nDims - 1, M_PI / 2);

        vector<vect> path;
        vect momentum;

        for (size_t iter = 0; ; iter++) {
            if (iter % preHessIters == 0 && tryToConverge(stopStrategy, func, p, r, path, 5, iter)) {
                LOG_ERROR("breaked here");
                break;
            }

            auto polar = makePolarWithDirection(func, r, p);

            auto valueGrad = polar.valueGrad(theta);
            auto value = get<0>(valueGrad);
            vect grad = get<1>(valueGrad);

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
    }
}