#pragma once

#include "helper.h"

#include "FunctionProducer.h"

template<typename FuncT, int N_FIXED>
class FixValues : public FunctionProducer<FuncT::N - N_FIXED>
{
public:
    static constexpr int N_DIMS = FuncT::N;
    using FunctionProducer<N_DIMS - N_FIXED>::N;

    FixValues(FuncT func, array<size_t, N_FIXED> const& poss, array<double, N_FIXED> const& vals) : mFunc(move(func)),
                                                                                                    mPoss(poss),
                                                                                                    mVals(vals)
    {
        assert(is_sorted(mPoss.begin(), mPoss.end()));
        assert(unique(mPoss.begin(), mPoss.end()) == mPoss.end());
    }

    double operator()(vect<N> const& x) override
    {
        return mFunc(transform(x));
    }

    vect<N> grad(vect<N> const& x) override
    {
        auto grad = mFunc.grad(transform(x));
        vect<N> result;
        for (size_t i = 0, j = 0, k = 0; i < (size_t) N_DIMS; i++)
            if (i == mPoss[j])
                j++;
            else
                result(k++) = grad(i);
        return result;
    }

    matrix<N, N> hess(vect<N> const& x) override
    {
        auto hess = mFunc.hess(transform(x));
        matrix<N, N> result;
        for (size_t i1 = 0, j1 = 0, k1 = 0; i1 < (size_t) N_DIMS; i1++)
            if (i1 == mPoss[j1])
                j1++;
            else {
                for (size_t i2 = 0, j2 = 0, k2 = 0; i2 < (size_t) N_DIMS; i2++)
                    if (i2 == mPoss[j2])
                        j2++;
                    else {
                        result(k1, k2++) = hess(i1, i2);
                    }
                k1++;
            }
        return result;
    };

    vect<N_DIMS> transform(vect<N> const& from) const
    {
        vect<N_DIMS> to;
        for (size_t i = 0, j = 0, k = 0; i < (size_t) N_DIMS; i++)
            if (i == mPoss[j])
                to(i) = mVals[j++];
            else
                to(i) = from(k++);
        return to;
    }

    FuncT const& getInnerFunction() const
    {
        return mFunc;
    }

private:
    FuncT mFunc;
    array<size_t, N_FIXED> mPoss;
    array<double, N_FIXED> mVals;
};

template<int N_FIXED, typename FuncT>
auto fix(FuncT&& func, array<size_t, N_FIXED> const& poss, array<double, N_FIXED> const& vals)
{
    return FixValues<decay_t<FuncT>, N_FIXED>(forward<FuncT>(func), poss, vals);
};

template<typename FuncT>
auto fixAtomSymmetry(FuncT&& func)
{
    return fix<6>(forward<FuncT>(func), {0, 1, 2, 4, 5, 8}, {0., 0., 0., 0., 0., 0.});
};

template<typename FuncT, int N>
auto fixAtomSymmetry(FuncT&& func, vect<N> const& pos)
{
    return fix<6>(forward<FuncT>(func), {0, 1, 2, 4, 5, 8}, {pos(0), pos(1), pos(2), pos(4), pos(5), pos(8)});
};

