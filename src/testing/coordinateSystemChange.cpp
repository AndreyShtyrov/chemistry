#include "helper.h"

#include "tests.h"

#include "InputOutputUtils.h"
#include "producers/InPolar.h"
#include "producers/FixValues.h"
#include "producers/GaussianProducer.h"
#include "producers/AffineTransformation.h"
#include "optimization/stop_strategies/HistoryStrategyWrapper.h"
#include "optimization/stop_strategies/StopStrategy.h"
#include "PythongraphicsFramework.h"

using namespace boost;
using namespace optimization;

template<typename FuncT, typename StopStrategy>
vector<vect> optimizeOnSphere(StopStrategy stopStrategy, FuncT &func, vect p, double r, size_t preHessIters) {
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
    } catch (GaussianException const &exc) {
        throw exc;
    }
}


void testTrajectory()
{
    ifstream input("./2/148.xyz");
    vect state;
    vector<size_t> charges;
    tie(charges, state) = readChemcraft(input);

    GaussianProducer func(charges);

    logFunctionInfo("", func, state);

    auto hess = func.hess(state);
    auto A = linearization(hess);
    cout << A.transpose() * hess * A << endl << endl;


    auto fixed = fixAtomSymmetry(func);
    state = fixed.backTransform(state);
    logFunctionInfo("", fixed, state);

    hess = fixed.hess(state);
    A = linearization(hess);
    cout << A.transpose() * hess * A << endl << endl;

    return;

    vector<double> vs;
    vector<vect> grads;
    vector<double> xs, ys;

    RandomProjection proj(15, 2);

    for (size_t i = 0; i < 217; i++) {
        ifstream input(str(format("./2/%1%.xyz") % i));

        vect state;
        vector<size_t> charges;
        tie(charges, state) = readChemcraft(input);

        GaussianProducer func(charges);

        vect grad = func.grad(state);
        double value = func(state);

        vs.push_back(value);
        grads.push_back(grad);

        auto cur = proj(toDistanceSpace(state));
        assert(cur.rows() == 2);
        xs.push_back(cur(0));
        ys.push_back(cur(1));
    }

    for (size_t i = 0; i < vs.size(); i++) {
        if (i == 0 || vs[i - 1] < vs[i])
            cout << '+';
        else
            cout << '-';

        if (i + 1 < vs.size() && vs[i] < vs[i + 1])
            cout << '+';
        else
            cout << '-';

        cout << i << ' ' << vs[i] << ' ' << grads[i].norm() << endl;
    }

    framework.plot(framework.newPlot(), vs);
    framework.plot(framework.newPlot(), xs, ys);
}

template<typename FuncT>
void runSHS(FuncT& func, vect equilStruct, vect direction)
{
    auto normalized = normalizeForPolar(func, equilStruct);
    direction = normalized.backTransform(direction);

    vect u = eye(func.nDims, func.nDims - 1);
    vect v = direction.normalized();
    auto M = rotationMatrix(u, v);
    auto rotated = makeAffineTransfomation(normalized, M);
    direction = rotated.backTransform(direction);

    auto polar = makePolar(rotated, direction.norm());
    auto phi = makeConstantVect(polar.nDims, M_PI / 2);

    LOG_INFO("{}", polar.grad(phi).norm());

//    equilStruct = prepared.backTransform(equilStruct);
//    direction = prepared.backTransform(direction);
//
//    LOG_INFO("\nnew equilStruct = {}\nnew diraction = {}\ndist = {}",
//             equilStruct.transpose(), direction.transpose(), direction.norm());
//
//    double r = direction.norm();
//    auto polar = makePolar(prepared, r);
}

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

    auto linearHessian = normalizeForPolar(fixedSym, equilStruct);

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
                LOG_INFO("New {} point in path:\n\tvalue = {:.13f}\n\tdelta norm = {:.13f}\n\t{}\nchemcraft coords:\n{}", j,
                         newValue, (direction / direction.norm() - prev / prev.norm()).norm(), direction.transpose(),
                         toChemcraftCoords(charges, linearHessian.fullTransform(direction)));

                ofstream output(str(boost::format("./%1%/%2%.xyz") % i % j));
                output << toChemcraftCoords(charges, linearHessian.fullTransform(direction)) << endl;

                if (newValue < value) {
                    LOG_ERROR("newValue < value [{:.13f} < {:.13f}]. Stopping", newValue, value);
                    //break;
                }

                value = newValue;
            } catch (GaussianException const &exc) {
                break;
            }
        }

        framework.plot(framework.newPlot(), xs, ys);
    }
}

void coordinateSystemChanges()
{
    ifstream input("./C2H4");
    auto charges = readCharges(input);
    auto equilStruct = readVect(input);

    vect diraction;
    tie(charges, diraction) = readChemcraft(ifstream("./0/0.xyz"));

    auto molecule = GaussianProducer(charges);
    auto fixedSym = fixAtomSymmetry(molecule);
    auto normalized = normalizeForPolar(fixedSym, fixedSym.backTransform(equilStruct)) ;

    int const N = 10;
    double const EPS = .1 / N;

    for (size_t i = 0; i < normalized.nDims; i++) {
        system(str(format("mkdir ./normalized/%1%") % i).c_str());
        for (int j = -N; j <= N; j++) {
            ofstream output(str(format("./normalized/%1%/%2%.xyz") % i % j));
            output << toChemcraftCoords(charges, normalized.fullTransform(eye(normalized.nDims, i) * EPS * j));
        }
    }

    return;

    {
        auto fixedSym = fixAtomSymmetry(molecule);
        runSHS(fixedSym, fixedSym.backTransform(equilStruct), fixedSym.backTransform(diraction));
    }

    {
        auto fixedSym = fixAtomSymmetry(molecule, 0, 2, 4);
        runSHS(fixedSym, fixedSym.backTransform(rotateToXYZ(equilStruct, 0, 2, 4)),fixedSym.backTransform(rotateToXYZ(diraction, 0, 2, 4)));
    }
}