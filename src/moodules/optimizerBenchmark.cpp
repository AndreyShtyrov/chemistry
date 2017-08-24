#include "helper.h"

#include <gtest/gtest.h>


#include "linearAlgebraUtils.h"
#include "producers/GaussianProducer.h"
#include "producers/FixValues.h"
#include "producers/AffineTransformation.h"
#include "optimization/stop_strategies/StopStrategy.h"
#include "optimization/optimizeOnSphere.h"
#include "PythongraphicsFramework.h"

TEST(Benchmark, OptimizatorsOnSphere)
{
    ifstream input("./C2H4");
    auto charges = readCharges(input);
    auto equilStruct = readVect(input);

    auto molecule = fixAtomSymmetry(GaussianProducer(charges, 1));
    equilStruct = molecule.backTransform(equilStruct);
    auto normalized = normalizeForPolar(molecule, equilStruct);

    RandomProjection projection(normalized.nDims);
    auto const stopStrategy = optimization::StopStrategy(1e-4, 1e-4);

    double const r = .1;

#pragma omp parallel
    while (true) {
        vect pos = randomVectOnSphere(normalized.nDims, r);

        auto path1 = optimizeOnSphere3(stopStrategy, normalized, pos, r, 50000000);
        auto path2 = optimizeOnSphere4(stopStrategy, normalized, pos, r, 50000000);
        auto path3 = optimizeOnSphere(stopStrategy, normalized, pos, r, 500000000);

        auto axis = framework.newPlot();
        auto drawPath = [&](vector<vect> const& path) {
            vector<double> xs, ys;
            for (auto const& p : path) {
                auto proj = projection(p);
                xs.push_back(proj(0));
                ys.push_back(proj(1));
            }
            framework.plot(axis, xs, ys);
        };

#pragma omp critical
        {
            drawPath(path1);
            drawPath(path2);
            drawPath(path3);
            LOG_INFO("path lengths: {} vs {} vs {}", path1.size(), path2.size(), path3.size());
        }
    }

}
