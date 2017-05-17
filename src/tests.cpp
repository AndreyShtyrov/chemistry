#include "helper.h"

#include <gtest/gtest.h>

#include "FunctionProducers.h"

vect getRandomPoint(vect const& lowerBound, vect const& upperBound)
{
    auto p = makeRandomVect(lowerBound.rows());
    return lowerBound.array() + p.array() * (upperBound.array() - lowerBound.array());
}

template<typename FuncT>
void testGradient(FuncT& func, vect const& lowerBound, vect const& upperBound, size_t iters, double delta = 1e-4,
                  double eps = 1e-5)
{
    size_t N = lowerBound.rows();

    for (size_t i = 0; i < iters; i++) {
        auto p = getRandomPoint(lowerBound, upperBound);
        auto grad = func.grad(p);

        for (size_t j = 0; j < N; j++) {
            vect e = delta * eye(N, j);
            double predicted = 0.5 * (func(p + e) - func(p - e)) / delta;
            ASSERT_LE(abs(grad(j) - predicted), eps)
                              << boost::format("value %1% != %2% (%3% error)") % grad(j) % predicted %
                                 abs(grad(j) - predicted);
        }
    }
};

template<typename FuncT>
void
testHessian(FuncT& func, vect lowerBound, vect upperBound, size_t iters, double delta = 1e-4, double eps = 1e-5)
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
                ASSERT_LE(abs(hess(x1, x2) - predicted), eps)
                                  << boost::format("value %1% != %2% (%3% error)") % hess(x1, x2) % predicted %
                                     abs(hess(x1, x2) - predicted);
            }
    }
};

template<typename FuncT>
void
testProducer(FuncT func, vect lowerBound, vect upperBound, size_t iters, double delta = 1e-4, double eps = 1e-5)
{
    testGradient(func, lowerBound, upperBound, iters, delta, eps);
    testHessian(func, lowerBound, upperBound, iters, delta, eps);
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
    testProducer(prepareForPolar(type(), b), lowerBound, upperBound, 1000);
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

TEST(FunctionProducer, Desturbed)
{
    auto lowerBound = makeVect(0., 0.);
    auto upperBound = makeVect(2 * M_PI, 2 * M_PI);

    testProducer(Desturbed<ModelFunction>(ModelFunction()), lowerBound, upperBound, 1000);
}

TEST(FunctionProducer, ModelFunction3)
{
    auto lowerBound = makeVect(-1., -1.);
    auto upperBound = makeVect(2., 2.);

    testProducer(ModelFunction3(), lowerBound, upperBound, 1000);
}

TEST(FunctionProducer, Sum)
{
    auto lowerBound = makeVect(-2., -2.);
    auto upperBound = makeVect(2., 2.);

    testProducer(ModelFunction() + ModelFunction3(), lowerBound, upperBound, 1000);
}

TEST(FunctionProducer, LagrangeMultiplier)
{
    auto lowerBound = makeVect(-2., -2., -2.);
    auto upperBound = makeVect(2., 2., 2.);

    testProducer(make_lagrange(ModelFunction(), SqrNorm(3) + Constant(3, -1.)), lowerBound, upperBound, 1000);
}

TEST(FunctionProducer, GaussianProducer)
{
    auto lowerBound = makeConstantVect(9, -1);
    auto upperBound = makeConstantVect(9,  1);

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
    auto linear_hessian = prepareForPolar(func, startPoint);
    auto polar = makePolar(linear_hessian, .3);

    auto lowerBound = makeConstantVect(polar.nDims, 0);
    auto upperBound = makeConstantVect(polar.nDims, 2 * M_PI);

    testProducer(polar, lowerBound, upperBound, 5, 1e-3, 1e-3);
}

TEST(FunctionProducer, ModelMultidimensionalFunction)
{
//    using type = ModelMultidimensionalFunction<20>;
    using type = ModelMultidimensionalZeroHessFunction<20>;

    auto lowerBound = makeConstantVect(20, .5);
    auto upperBound = makeConstantVect(20, 1.);

    testProducer(type(), lowerBound, upperBound, 1000);
}

TEST(FixValues, rotateToFix)
{
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

