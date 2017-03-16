#include "helper.h"

#include <boost/format.hpp>
#include <boost/lexical_cast.hpp>
#include <Eigen/LU>
#include <optimization/GradientLengthStopCriteria.h>
#include <optimization/GradientOptimization.h>

#include "linearization.hpp"
#include "FunctionProducers.h"
#include "PythongraphicsFramework.h"

#include "optimization/optimizations.h"

using namespace std;
using namespace optimization;

constexpr double EPS = 1e-7;

constexpr double calculate_delta(double min, double max, int n)
{
    return (max - min) / (n - 1);
}

constexpr double MAX_VAL = 1;
constexpr double MIN_X = -MAX_VAL;
constexpr double MAX_X = MAX_VAL;
constexpr double MIN_Y = -MAX_VAL;
constexpr double MAX_Y = MAX_VAL;

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
vector<vect<FuncT::N>>
shs(FuncT& func, Optimizer1T optimizer1, Optimizer2T optimizer2, double deltaR, vect<FuncT::N> pos)
{
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

        pos = polars.transform(lastPolar);

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

template<int N>
vect<N> getRandomPoint(vect<N> const& lowerBound, vect<N> const& upperBound)
{
    auto p = make_random_vect<N>();
    return lowerBound.array() + p.array() * (upperBound.array() - lowerBound.array());
}

template<int N>
string to_chemcraft_coords(vector<size_t> const& sizes, vect<N> p)
{
    assert(N == sizes.size() * 3);

    stringstream result;
    for (size_t i = 0; i < sizes.size(); i++)
        result << boost::format("%1%\t%2%\t%3%\t%4%") % sizes[i] % p(i * 3 + 0) % p(i * 3 + 1) % p(i * 3 + 2) << endl;
    return result.str();
}

struct PlotInfo
{
    vector<double> xs, ys;
};


template<typename FuncT, typename DeltaStrategyT, typename StopStrategyT, int N>
tuple<vector<double>, vector<double>>
testOptimizer(FuncT& func, DeltaStrategyT&& deltaStrategy, StopStrategyT&& stopStrategy, vect<N> const& p)
{
    auto optimizer = make_gradient_descent(make_history_strategy(forward<DeltaStrategyT>(deltaStrategy)),
                                           forward<StopStrategyT>(stopStrategy));
    auto path = optimizer(func, p);
    auto vals = optimizer.getDeltaStrategy().getValues();

    return make_tuple(arange(vals.size()), vals);
};

template<typename FuncT>
vect<FuncT::N> optimize(FuncT& func, vect<FuncT::N> const& x)
{
    auto optimizer = make_gradient_descent(FollowGradientDeltaStrategy<FuncT::N>(),
                                           make_atomic_stop_strategy(0.00045, 0.0003, 0.018, 0.012, func));
    return optimizer(func, x).back();
};

#include <typeinfo>

int main()
{
    auto v = make_vect(1.04218, 0.31040, 1.00456);

    vector<size_t> weights = {8, 1, 1};
    auto atomicFunc = GaussianProducer<9>(weights);
    auto func = fix_atom_symmetry(atomicFunc);

    auto local_minima = optimize(func, v);
    cout << boost::format("local minima:\n%1%\ngradient: %2%\nhessian:\n%3%\n\n") %
            to_chemcraft_coords(weights, func.transform(local_minima))
            % func.grad(local_minima).transpose()
            % func.hess(local_minima);

//    auto linear_hessian = prepare_for_polar(func, v);
//    auto stopStrategy = make_atomic_stop_strategy(0, 0, 0, 0, linear_hessian);

    return 0;
}
