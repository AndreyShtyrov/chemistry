#include "helper.h"

#include <gtest/gtest.h>


#include "linearAlgebraUtils.h"
#include "producers/GaussianProducer.h"
#include "producers/FixValues.h"
#include "producers/AffineTransformation.h"
#include "optimization/stop_strategies/StopStrategy.h"
#include "optimization/optimizeOnSphere.h"
#include "PythongraphicsFramework.h"

using namespace optimization;

template<typename FuncT>
void logFunctionInfo(string const& title, FuncT& func, vect const& p)
{
    auto valueGradHess = func.valueGradHess(p);
    auto value = get<0>(valueGradHess);
    auto grad = get<1>(valueGradHess);
    auto hess = get<2>(valueGradHess);

    LOG_INFO("{}\n\tposition: {}\n\tenergy: {}\n\tgradient: {} [{}]\n\thessian: {}\n\n",
             title, p.transpose(), value, grad.norm(), grad.transpose(), singularValues(hess));
}

template<typename FuncT>
void findInitialPolarDirections(FuncT& func, double r)
{
    auto const axis = framework.newPlot();
    RandomProjection const projection(func.nDims);
    StopStrategy const stopStrategy(1e-7, 1e-7);

    ofstream output("./mins_on_sphere");
    output.precision(30);

    #pragma omp parallel
    while (true) {
        vect pos = randomVectOnSphere(func.nDims, r);

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
            logFunctionInfo("new initial polar direction", polar, makeConstantVect(polar.nDims, M_PI / 2));
        }
    }
}

TEST(EntryPoint, InitialPolarDirectionsSearch)
{
    initializeLogger();

    ifstream input("./C2H4");
    auto charges = readCharges(input);
    auto equilStruct = readVect(input);

    auto molecule = fixAtomSymmetry(GaussianProducer(charges, 1));
    equilStruct = molecule.backTransform(equilStruct);
    auto normalized = normalizeForPolar(molecule, equilStruct);

    findInitialPolarDirections(normalized, .3);
}



