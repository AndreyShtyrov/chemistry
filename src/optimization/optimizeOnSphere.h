#pragma once

#include "helper.h"

#include "linearAlgebraUtils.h"
#include "producers/AffineTransformation.h"
#include "producers/GaussianProducer.h"
#include "producers/InPolar.h"

namespace optimization
{
    template<typename FuncT, typename StopStrategy>
    vector<vect> optimizeOnSphere(StopStrategy stopStrategy, FuncT& func, vect p, double r, size_t preHessIters)
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
    bool tryToConverge(StopStrategy stopStrategy, FuncT& func, vect p, double r, vector<vect>& path, size_t globalIter=0)
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

                auto sValues = singularValues(hess);
                for (size_t j = 0; j < sValues.size(); j++)
                    if (sValues(j) < 0) {
                        return false;
                    }

                auto lastP = p;
                p = polar.getInnerFunction().transform(polar.transform(theta - hess.inverse() * grad));
                newPath.push_back(p);

                if (stopStrategy(globalIter + i, p, value, grad, hess, p - lastP)) {
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
    }

    template<typename FuncT, typename StopStrategy>
    vector<vect> optimizeOnSphere4(StopStrategy stopStrategy, FuncT& func, vect p, double r, size_t preHessIters)
    {
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
                momentum = max(0., momentum.dot(grad) / (grad.norm() * momentum.norm())) * momentum + grad;
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