#include "helper.h"

#include <gtest/gtest.h>

#include "producers/producers.h"

vect getRandomPoint(vect const& lowerBound, vect const& upperBound)
{
    auto p = makeRandomVect(lowerBound.rows());
    return lowerBound.array() + p.array() * (upperBound.array() - lowerBound.array());
}

template<typename FuncT>
void testGradient(FuncT& func, vect const& lowerBound, vect const& upperBound, size_t iters, double delta = 1e-5,
                  double eps = 1e-5)
{
    size_t N = (size_t) lowerBound.rows();

    for (size_t i = 0; i < iters; i++) {
        auto p = getRandomPoint(lowerBound, upperBound);
        auto grad = func.grad(p);

        for (size_t j = 0; j < N; j++) {
            vect e = delta * eye(N, j);
            double predicted = 0.5 * (func(p + e) - func(p - e)) / delta;

            cout << boost::format("gradient: grad(%1%) = %2% == %3%") % j % grad(j) % predicted << endl;
            ASSERT_LE(abs(grad(j) - predicted), eps)
                              << boost::format("value %1% != %2% (%3% error)") % grad(j) % predicted %
                                 abs(grad(j) - predicted);
        }
    }
};

template<typename FuncT>
void testHessian(FuncT& func, vect lowerBound, vect upperBound, size_t iters, double delta = 1e-4, double eps = 1e-5)
{
    size_t N = lowerBound.rows();

    for (size_t i = 0; i < iters; i++) {
        auto p = getRandomPoint(lowerBound, upperBound);
        auto hess = func.hess(p);
        for (size_t x1 = 0; x1 < N; x1++)
            for (size_t x2 = 0; x2 <= x1; x2++) {
                vect e1 = delta * eye(N, x1), e2 = delta * eye(N, x2);
                double predicted =
                   0.25 * (func(p + e1 + e2) - func(p - e1 + e2) - func(p + e1 - e2) + func(p - e1 - e2)) / sqr(delta);
                cout << boost::format("gradient: hess(%1%, %2%) = %3% == %4%") % x1 % x2 % hess(x1, x2) % predicted << endl;
                ASSERT_LE(abs(hess(x1, x2) - predicted), eps)
                                  << boost::format("value %1% != %2% (%3% error)") % hess(x1, x2) % predicted %
                                     abs(hess(x1, x2) - predicted);
            }
    }
};

template<typename FuncT>
void testCollected(FuncT func, vect lowerBound, vect upperBound, size_t iters, double eps)
{
    for (size_t i = 0; i < iters; i++) {
        auto p = getRandomPoint(lowerBound, upperBound);

        auto valueGradHess = func.valueGradHess(p);
        auto valueGrad = func.valueGrad(p);
        auto hess = func.hess(p);
        auto grad = func.grad(p);
        auto value = func(p);

        ASSERT_LE(abs(value - get<0>(valueGrad)), eps);
        ASSERT_LE(abs(value - get<0>(valueGradHess)), eps);
        ASSERT_LE((grad - get<1>(valueGrad)).norm(), eps);
        ASSERT_LE((grad - get<1>(valueGradHess)).norm(), eps);
        ASSERT_LE((hess - get<2>(valueGradHess)).norm(), eps);
    }
}

template<typename FuncT>
void testProducer(FuncT func, vect lowerBound, vect upperBound, size_t iters, double delta = 1e-5, double eps = 1e-3)
{
    testGradient(func, lowerBound, upperBound, iters, delta, eps);
    testHessian(func, lowerBound, upperBound, iters, delta, eps);
    testCollected(func, lowerBound, upperBound, iters, eps);
};


TEST(FunctionProducer, ModelFunction)
{
    auto lowerBound = makeVect(-1., -1.);
    auto upperBound = makeVect(1., 1.);

    testProducer(ModelFunction(), lowerBound, upperBound, 1000);
}

TEST(FunctionProducer, AffineTransformation)
{
    using type = ModelFunction;
//    using type = SqrNorm<5>;
    auto func = type();

    auto b = makeRandomVect(2);
//    auto hess = func.hess(b);
    cerr << func.hess(b) << endl;
    auto A = makeRandomMatrix(2, 2);
//    auto A = linearization(hess);

    cerr << endl << A << endl;

    auto lowerBound = makeConstantVect(2, -1.);
    auto upperBound = makeConstantVect(2, 1.);

    testProducer(makeAffineTransfomation(type(), b, A), lowerBound, upperBound, 1000);
    testProducer(normalizeForPolar(type(), b), lowerBound, upperBound, 1000);
}

TEST(FunctionProducer, InPolar)
{
    using FunctionType = ModelFunction;
//    using FunctionType = ModelMultidimensionalFunction<2>;
//    using FunctionType = ModelMultidimensionalZeroHessFunction<2>;

    auto lowerBound = makeConstantVect(1, 0.);
    auto upperBound = makeConstantVect(1, 2 * M_PI);

    testProducer(makePolar(FunctionType(), 1.313), lowerBound, upperBound, 100);
}

TEST(FunctionProducer, GaussianProducer)
{
    auto lowerBound = makeConstantVect(9, -1);
    auto upperBound = makeConstantVect(9, 1);

    testProducer(GaussianProducer({8, 1, 1}), lowerBound, upperBound, 1, 1e-4, 1e-3);
}

TEST(FunctionProducer, GaussianProducer_2)
{
    vector<size_t> weights = {6, 6, 1, 1, 1, 1};
    auto startPoint = makeVect(0.664007917, 1.726194372, -1.239100083, -1.022740312, -0.120487628, -1.239100083,
                               -1.022740312, 1.726194372, 1.236957917, -1.022740312, -0.120487628, 1.236957917);

    testProducer(fixAtomSymmetry(GaussianProducer(weights)), startPoint, startPoint, 1, 1e-4, 1e-3);
}


TEST(FunctionProducer, FixValues)
{
    auto lowerBound = makeConstantVect(3, .9);
    auto upperBound = makeConstantVect(3, 1);

    testProducer(fixAtomSymmetry(GaussianProducer({8, 1, 1})), lowerBound, upperBound, 5, 1e-4, 1e-3);
}

TEST(FunctionProducer, Stack)
{
    auto startPoint = makeVect(1.04218, 0.31040, 1.00456);

    vector<size_t> weights = {8, 1, 1};
    auto atomicFunc = GaussianProducer(weights);
    auto func = fixAtomSymmetry(atomicFunc);
    auto linear_hessian = normalizeForPolar(func, startPoint);
    auto polar = makePolar(linear_hessian, .3);

    auto lowerBound = makeConstantVect(polar.nDims, 0);
    auto upperBound = makeConstantVect(polar.nDims, 2 * M_PI);

    testProducer(polar, lowerBound, upperBound, 1, 1e-3, 1e-3);
}

TEST(FunctionProducer, Stack2)
{
    initializeLogger();

    vector<size_t> charges = {6, 6, 1, 1, 1, 1};
    vect initState = makeVect(0.000000000, 0.000000000, -0.665079000, 0.035065274, 0.058977884, 0.684472749,
                              0.150810779, 0.964915143, -1.225651234, -0.177203028, -0.964988711, -1.253913402,
                              -0.066854801, 0.720391837, 1.298331278, 0.301663799, -0.766483770, 1.184359568);

    initState = rotateToFix(initState);
    auto molecule = GaussianProducer(charges);
    auto prepared = fixAtomSymmetry(makeAffineTransfomation(molecule, initState));

    auto localMinima = makeVect(-0.495722, 0.120477, -0.874622, 0.283053, 0.784344, -0.00621205, -0.787401, -0.193879,
                                -0.301919, -0.553383, 0.552153, 0.529974);
    auto linearHessian = normalizeForPolar(prepared, localMinima);
    auto polar = makePolar(linearHessian, .3);


    auto lowerBound = makeConstantVect(polar.nDims, 0);
    auto upperBound = makeConstantVect(polar.nDims, 1);

    testHessian(polar, lowerBound, upperBound, 1, 1e-3, 1e-3);
    testProducer(polar, lowerBound, upperBound, 1, 1e-3, 1e-3);
}

TEST(FunctionProducer, Stack3)
{
    ifstream input("./C2H4");

    auto charges = readCharges(input);
    auto state = readVect(input);

//    state = rotateToFix(state);
    auto molecule = GaussianProducer(charges);
    auto prepared = fixAtomSymmetry(molecule);

    state = prepared.backTransform(state);
    auto linearHessian = normalizeForPolar(prepared, state);
    auto polar = makePolar(linearHessian, .3);

    auto lowerBound = makeConstantVect(polar.nDims, M_PI / 2 - 1);
    auto upperBound = makeConstantVect(polar.nDims, M_PI / 2 + 1);

    testProducer(polar, lowerBound, upperBound, 1, 1e-3, 1e-3);
}


TEST(FixValues, rotateToFix)
{
    initializeLogger();

    for (size_t i = 0; i < 100; i++) {
        vect v(9);
        for (size_t i = 0; i < (size_t) v.rows(); i++)
            v(i) = (double) rand() / RAND_MAX;
        v = rotateToFix(v);

        for (size_t i = 0; i < 9; i++)
            if (i != 3 && i != 6 && i != 7)
                ASSERT_LE(abs(v(0)), 1e-7) << "vector components error";
    }
}

TEST(FunctionProducer, OnSphereCosineSupplement)
{
    size_t const nDims = 2;

    vect direction = makeRandomVect(nDims);
    OnSphereCosineSupplement func(direction, 1.);

    testProducer(func, makeConstantVect(nDims, -1), makeConstantVect(nDims, 1), 1000);
}

TEST(FunctionProducer, OnSphereCosineSupplement2) {
    initializeLogger();

    ifstream mins("./mins_on_sphere_filtered");
    size_t cnt;
    mins >> cnt;

    auto direction = readVect(mins);
    ifstream input("./C2H4");
    auto charges = readCharges(input);
    auto equilStruct = readVect(input);

    auto molecule = fixAtomSymmetry(GaussianProducer(charges, 3));
    equilStruct = molecule.backTransform(equilStruct);
    auto normalized = normalizeForPolar(molecule, equilStruct);
    auto zeroEnergy = normalized(makeConstantVect(normalized.nDims, 0));

    double const r = .1;
    OnSphereCosineSupplement supplement(direction, (sqr(r) / 2 - (normalized(direction) - zeroEnergy)) / r / r / r);
    testProducer(supplement, makeConstantVect(supplement.nDims, -1), makeConstantVect(supplement.nDims, 1), 1000);
}

TEST(FunctionProducer, Cosine3OnSPhereInterpolation)
{
    size_t const nDims = 2;
    size_t const N = 5;

    vector<double> values;
    vector<vect> directions;

    uniform_real_distribution<double> random;
    for (size_t i = 0; i < N; i++) {
        values.push_back(random(randomGen));
        directions.push_back(randomVectOnSphere(nDims));
    }
    Cosine3OnSphereInterpolation func(nDims, values, directions);

    testProducer(func, makeConstantVect(nDims, -1), makeConstantVect(nDims, 1), 1000);
}


TEST(FunctionProducer, ClosestCosine3OnSphere)
{
    size_t const nDims = 25;
    size_t const N = 5;

    vector<double> values;
    vector<vect> directions;

    uniform_real_distribution<double> random;
    for (size_t i = 0; i < N; i++) {
        values.push_back(random(randomGen));
        directions.push_back(randomVectOnSphere(nDims));
    }
    ClosestCosine3OnSphere func(nDims, values, directions);

    testProducer(func, makeConstantVect(nDims, -1), makeConstantVect(nDims, 1), 1000);
}

TEST(FunctionProducer, LargestCosine3OnSphere)
{
    size_t const nDims = 25;
    size_t const N = 5;

    vector<double> values;
    vector<vect> directions;

    uniform_real_distribution<double> random;
    for (size_t i = 0; i < N; i++) {
        values.push_back(random(randomGen));
        directions.push_back(randomVectOnSphere(nDims));
    }
    LargestCosine3OnSphere func(nDims, values, directions);

    testProducer(func, makeConstantVect(nDims, -1), makeConstantVect(nDims, 1), 1000);
}

TEST(FunctionProducer, CleverCosine3OnSphereInterpolation)
{
    initializeLogger();

    size_t const nDims = 25;
    size_t const N = 5;

    vector<double> values;
    vector<vect> directions;

    uniform_real_distribution<double> random;
    for (size_t i = 0; i < N; i++) {
        values.push_back(random(randomGen));
        directions.push_back(randomVectOnSphere(nDims));
    }

    CleverCosine3OnSphereInterpolation func(nDims, values, directions);

//    testProducer(func, makeConstantVect(nDims, -1), makeConstantVect(nDims, 1), 1000);
}


TEST(FunctionProducer, Calculations)
{
    size_t const nDims = 10;

    SqrNorm norm(nDims);
    auto func = 1.3 * norm + (-0.3) * norm;

    for (size_t i = 0; i < 10; i++) {
        auto v = makeRandomVect(nDims);
        ASSERT_LE(abs(func(v) - v.dot(v)), 1e-7);
    }
    testProducer(func, makeConstantVect(nDims, -1), makeConstantVect(nDims, 1), 1000);
}


