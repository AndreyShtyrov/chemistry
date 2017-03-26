#include "helper.h"

#include <gtest/gtest.h>

#include "FunctionProducers.h"

template<int N>
vect<N> getRandomPoint(vect<N> const& lowerBound, vect<N> const& upperBound)
{
    auto p = makeRandomVect<N>();
    return lowerBound.array() + p.array() * (upperBound.array() - lowerBound.array());
}

template<typename FuncT, int N = FuncT::N>
void testGradient(FuncT func, vect<N> const& lowerBound, vect<N> const& upperBound, size_t iters, double delta = 1e-4,
                  double eps = 1e-5)
{
    for (size_t i = 0; i < iters; i++) {
        auto p = getRandomPoint(lowerBound, upperBound);
        auto grad = func.grad(p);

        for (size_t j = 0; j < N; j++) {
            vect<N> e = delta * eye<N>(j);
            double predicted = 0.5 * (func(p + e) - func(p - e)) / delta;
            ASSERT_LE(abs(grad(j) - predicted), eps)
                              << boost::format("value %1% != %2% (%3% error)") % grad(j) % predicted %
                                 abs(grad(j) - predicted);
        }
    }
};

template<typename FuncT, int N = FuncT::N>
void
testHessian(FuncT func, vect<N> lowerBound, vect<N> upperBound, size_t iters, double delta = 1e-4, double eps = 1e-5)
{
    for (size_t i = 0; i < iters; i++) {
        auto p = getRandomPoint(lowerBound, upperBound);
        auto hess = func.hess(p);
        for (size_t x1 = 0; x1 < N; x1++)
            for (size_t x2 = 0; x2 <= x1; x2++) {
                vect<N> e1 = delta * eye<N>(x1), e2 = delta * eye<N>(x2);
                double predicted =
                   0.25 * (func(p + e1 + e2) - func(p - e1 + e2) - func(p + e1 - e2) + func(p - e1 - e2)) / sqr(delta);
                ASSERT_LE(abs(hess(x1, x2) - predicted), eps)
                                  << boost::format("value %1% != %2% (%3% error)") % hess(x1, x2) % predicted %
                                     abs(hess(x1, x2) - predicted);
            }
    }
};

template<typename FuncT, int N = FuncT::N>
void
testProducer(FuncT func, vect<N> lowerBound, vect<N> upperBound, size_t iters, double delta = 1e-4, double eps = 1e-5)
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

    auto b = makeRandomVect<type::N>();
//    auto hess = func.hess(b);
    cerr << func.hess(b) << endl;
    auto A = makeRandomMatrix<type::N, type::N>();
//    auto A = linearization(hess);

    cerr << endl << A << endl;

    auto lowerBound = makeConstantVect<type::N>(-1.);
    auto upperBound = makeConstantVect<type::N>(1.);

    testProducer(makeAffineTransfomation(type(), b, A), lowerBound, upperBound, 1000);
    testProducer(prepareForPolar(type(), b), lowerBound, upperBound, 1000);
}

TEST(FunctionProducer, InPolar)
{
    using FunctionType = ModelFunction;
//    using FunctionType = ModelMultidimensionalFunction<2>;
//    using FunctionType = ModelMultidimensionalZeroHessFunction<2>;

    auto lowerBound = makeConstantVect<FunctionType::N - 1>(0.);
    auto upperBound = makeConstantVect<FunctionType::N - 1>(2 * M_PI);

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

    testProducer(make_lagrange(ModelFunction(), SqrNorm<2>() + Constant<2>(-1.)), lowerBound, upperBound, 1000);
}

TEST(FunctionProducer, GaussianProducer)
{
    vect<9> ones;
    ones.setConstant(1);
    auto lowerBound = makeConstantVect<9>(-1.);
    auto upperBound = makeConstantVect<9>(1.);

    testProducer(GaussianProducer<9>({8, 1, 1}), lowerBound, upperBound, 1, 1e-4, 1e-3);
}

TEST(FunctionProducer, FixValues)
{
    vect<3> ones;
    ones.setConstant(1);
    auto lowerBound = makeConstantVect<3>(.9);
    auto upperBound = makeConstantVect<3>(1);

    testProducer(fixAtomSymmetry(GaussianProducer<9>({8, 1, 1})), lowerBound, upperBound, 5, 1e-4, 1e-3);
}

TEST(FunctionProducer, Stack)
{
    auto startPoint = makeVect(1.04218, 0.31040, 1.00456);

    vector<size_t> weights = {8, 1, 1};
    auto atomicFunc = GaussianProducer<9>(weights);
    auto func = fixAtomSymmetry(atomicFunc);
    auto linear_hessian = prepareForPolar(func, startPoint);
    auto polar = makePolar(linear_hessian, .3);

    auto lowerBound = makeConstantVect<polar.N>(0);
    auto upperBound = makeConstantVect<polar.N>(2 * M_PI);

    testProducer(polar, lowerBound, upperBound, 5, 1e-3, 1e-3);
}

TEST(FunctionProducer, ModelMultidimensionalFunction)
{
//    using type = ModelMultidimensionalFunction<20>;
    using type = ModelMultidimensionalZeroHessFunction<20>;

    auto lowerBound = makeConstantVect<type::N>(.5);
    auto upperBound = makeConstantVect<type::N>(1.);

    testProducer(type(), lowerBound, upperBound, 1000);
}
