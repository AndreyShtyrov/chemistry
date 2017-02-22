#include "helper.h"

#include "linearization.hpp"
#include "FunctionProducers.h"
#include "PythongraphicsFramework.h"
#include "GradientDescent.h"

using namespace std;

constexpr double EPS = 1e-7;

constexpr double calculate_delta(double min, double max, int n)
{
    return (max - min) / (n - 1);
}

constexpr double MAX_VAL = 1;
constexpr double MIN_X = -MAX_VAL;
constexpr double MAX_X =  MAX_VAL;
constexpr double MIN_Y = -MAX_VAL;
constexpr double MAX_Y =  MAX_VAL;

constexpr size_t N = 250;
constexpr double DX = calculate_delta(MIN_X, MAX_X, N);
constexpr double DY = calculate_delta(MIN_Y, MAX_Y, N);

constexpr size_t PHI = 1000;

template<int N>
vect<N> optimize(vect<N> pos, vect<N> grad, matrix<N, N> hess)
{
    return pos - hess.inverse() * grad;
}

template<typename FuncT, typename Optimizer1T, typename Optimizer2T>
vector<vect<FuncT::N>> shs(FuncT& func, Optimizer1T optimizer1, Optimizer2T optimizer2, double deltaR, vect<FuncT::N> pos) {
    vector<vect<FuncT::N>> path;
    path.push_back(pos);

    double lastValue = func(pos);

    vect<FuncT::N - 1> lastPolar;
    lastPolar.setRandom();
    cout << lastPolar << endl << endl;

    for (int i = 0; i < 20; ++i) {
//        cerr << "!" << endl;

        auto polars = make_polar(func, deltaR * (i + 1));
        lastPolar = optimizer1(polars, lastPolar).back();

        pos = polars.fromPolar(lastPolar);

        double curValue = func(pos);
        if (curValue < lastValue) {
            break;
        }
        lastValue = curValue;
        path.push_back(pos);
    }


    auto saddle = pos;
    for (int i = 0; i < 3; i++) {
        path.push_back(saddle);
        saddle = optimize(pos, func.grad(saddle), func.hess(saddle));
    }
    path.push_back(saddle);

//    auto pathToMinimum = optimizer2(func, pos);
//    path.insert(path.end(), pathToMinimum.begin(), pathToMinimum.end());

    return path;
}

double get_linear_comb(double from, double to, double t)
{
    return from + (to - from) * t;
}

template<typename FuncT>
string drawPlot(FuncT&& func, double from, double to, size_t iters)
{
    vector<double> xs, ys;
    for (size_t i = 0; i < iters; i++) {
        auto x = get_linear_comb(from, to, (double) i / (iters - 1));
        xs.push_back(x);
        ys.push_back(func(make_vect(x)));
    }

    auto axis = framework.newPlot();
    framework.plot(axis, xs, ys);

    return axis;
}

template<typename FuncT>
string draw3dPlot(FuncT&& func, vect<2> from, vect<2> to, size_t iters)
{
    vector<double> xs, ys, zs;
    for (size_t i = 0; i < iters; i++)
        for (size_t j = 0; j < iters; j++) {
            double x = get_linear_comb(from(0), to(0), (double) i / (iters - 1));
            double y = get_linear_comb(from(1), to(1), (double) j / (iters - 1));
            xs.push_back(x);
            ys.push_back(y);
            zs.push_back(func(make_vect(x, y)));
        }

    auto axis = framework.newPlot();
    framework.contour(axis, reshape(xs, iters), reshape(ys, iters), reshape(zs, iters), 250);

    return axis;
}

int main() {
    auto model = ModelFunction();
//    auto model = SqrVectorNorm<1>();
    constexpr int N = model.N;

    auto axis = draw3dPlot(model, make_vect(-1.5, -1.5), make_vect(1.5, 1.5), 250);

    auto func = make_lagrange(model, SqrNorm<N>() + Constant<N>(-1.));
//    draw3dPlot(func, make_vect(-1.5, -1.5), make_vect(1.5, 1.5), 250);
//    auto axis = draw3dPlot(model, make_vect(-1, -1), make_vect(1, 1), 250);
//    auto axis = drawPlot(model, make_vect(-1, -1), make_vect(1, 1), 250);
    auto path = HessianGradientDescent<func.N>(0.001)(func, 0.001 * make_random_vect<func.N>());
    {
        vector<double> xs;
        vector<double> ys;
        for (auto p : path)
            xs.push_back(p(0)), ys.push_back(p(1));
        framework.plot(axis, xs, ys);
    }
}

int main2() {
//    auto func = make_desturbed(ModelFunction());
    auto func = ModelFunction3(5.);
    auto contourAxis = framework.newPlot();
    {
        vector<double> xs, ys, zs;
        for (size_t x = 0; x < N; x++)
            for (size_t y = 0; y < N; y++) {
                double vx = MIN_X + x * DX;
                double vy = MIN_Y + y * DY;
                double vz = func({vx, vy});

                xs.push_back(vx);
                ys.push_back(vy);
                zs.push_back(vz);
            }

        framework.contour(contourAxis, reshape(xs, N), reshape(ys, N), reshape(zs, N), 250);
    }

    auto hess = func.hess({0., 0.});
    auto A = linearization(hess);
    auto func2 = make_affine_transfomation(func, A);
    auto path = shs(func2, Adam<1>(), NesterovGradientDescent<2>(), 0.1, {0, 0});

    cerr << "!" << endl;

    {
        vector<double> xs, ys, zs;
        for (size_t x = 0; x < N; x++)
            for (size_t y = 0; y < N; y++) {
                double vx = MIN_X + x * DX;
                double vy = MIN_Y + y * DY;
                double vz = func2({vx, vy});

                xs.push_back(vx);
                ys.push_back(vy);
                zs.push_back(vz);
            }

        auto axis = framework.newPlot();
        framework.contour(axis, reshape(xs, N), reshape(ys, N), reshape(zs, N), 250);
        xs.clear(), ys.clear();
        for (auto const& p : path)
            xs.push_back(p(0)), ys.push_back(p(1));
        framework.plot(axis, xs, ys);
    }

    {
        vector<double> xs, ys;
        for (auto const& p : path) {
            auto pNew = A * p;
            xs.push_back(pNew(0)), ys.push_back(pNew(1));
        }
        framework.plot(contourAxis, xs, ys);
    }

    {
        vector<double> xs, ys, zs;
        for (auto const& p : path) {
            xs.push_back(p.norm());
            ys.push_back(func2.grad(p).norm());
            zs.push_back(func2(p));
        }

        auto axis = framework.newPlot();
        framework.plot(axis, xs, ys);
        framework.plot(axis, xs, zs);
    }

    return 0;
}
