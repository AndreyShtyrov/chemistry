#include "helper.h"

#include <gtest/gtest.h>

#include "FunctionProducers.h"

template<int N>
vect<N> getRandomPoint(vect<N> const& lower_bound, vect<N> const&  upper_bound)
{
    auto p = make_random_vect<N>();
    return lower_bound.array() + p.array() * (upper_bound.array() - lower_bound.array());
}

template<typename FuncT, int N=FuncT::N>
void testGradient(FuncT func, vect<N> const& lower_bound, vect<N> const& upper_bound, size_t iters, double delta = 1e-4, double eps=1e-5)
{
    for (size_t i = 0; i < iters; i++) {
        auto p = getRandomPoint(lower_bound, upper_bound);
        auto grad = func.grad(p);

        for (size_t j = 0; j < N; j++) {
            vect<N> e = delta * eye<N>(j);
            double predicted = 0.5 * (func(p + e) - func(p - e)) / delta;
            assert(abs(grad(j) - predicted) < eps);
        }
    }
};

template<typename FuncT, int N=FuncT::N>
void testHessian(FuncT func, vect<N> lower_bound, vect<N> upper_bound, size_t iters, double delta = 1e-4, double eps=1e-5)
{
    for (size_t i = 0; i < iters; i++) {
        auto p = getRandomPoint(lower_bound, upper_bound);
        auto hess = func.hess(p);
        for (size_t x1 = 0; x1 < N; x1++)
            for (size_t x2 = 0; x2 < N; x2++) {
                vect<N> e1 = delta * eye<N>(x1), e2 = delta * eye<N>(x2);
                double predicted = 0.25 * (func(p + e1 + e2) - func(p - e1 + e2) - func(p + e1 - e2) + func(p - e1 - e2)) / sqr(delta);
                assert(abs(hess(x1, x2) - predicted) < eps);
            }
    }
};

template<typename FuncT, int N=FuncT::N>
void testProducer(FuncT func, vect<N> lower_bound, vect<N> upper_bound, size_t iters, double delta = 1e-4, double eps=1e-5)
{
    testGradient(func, lower_bound, upper_bound, iters, delta, eps);
    testHessian(func, lower_bound, upper_bound, iters, delta, eps);
};


TEST(FunctionProducer, ModelFunction)
{
    auto lower_bound = make_vect(-1., -1.);
    auto upper_bound = make_vect(1., 1.);

    testProducer(ModelFunction(), lower_bound, upper_bound, 1000);
}

TEST(FunctionProducer, InNewBasis)
{
    auto A = make_random_matrix<2, 2>();
    auto lower_bound = make_vect(-1., -1.);
    auto upper_bound = make_vect(1., 1.);

    testProducer(AffineTransformation<ModelFunction>(ModelFunction(), A), lower_bound, upper_bound, 1000);
}

TEST(FunctionProducer, PolarCoordinates)
{
    auto lower_bound = make_vect(0.);
    auto upper_bound = make_vect(2 * M_PI);

    for (int ri = 0; ri < 10; ri++)
        testGradient(make_polar(ModelFunction(), (ri + 1) * 0.2), lower_bound, upper_bound, 1000);
}

TEST(FunctionProducer, Desturbed)
{
    auto lower_bound = make_vect(0., 0.);
    auto upper_bound = make_vect(2 * M_PI, 2 * M_PI);

    testProducer(Desturbed<ModelFunction>(ModelFunction()), lower_bound, upper_bound, 1000);
}

TEST(FunctionProducer, ModelFunction3)
{
    auto lower_bound = make_vect(-1., -1.);
    auto upper_bound = make_vect(2., 2.);

    testProducer(ModelFunction3(), lower_bound, upper_bound, 1000);
}

TEST(FunctionProducer, Sum)
{
    auto lower_bound = make_vect(-2., -2.);
    auto upper_bound = make_vect(2., 2.);

    testProducer(ModelFunction() + ModelFunction3(), lower_bound, upper_bound, 1000);
}

TEST(FunctionProducer, LagrangeMultiplier)
{
    auto lower_bound = make_vect(-2., -2., -2.);
    auto upper_bound = make_vect(2., 2., 2.);

    testProducer(make_lagrange(ModelFunction(), SqrNorm<2>() + Constant<2>(-1.)), lower_bound, upper_bound, 1000);
}