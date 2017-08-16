#pragma once

#include "helper.h"

#include "FunctionProducer.h"

template<typename FuncT>
class FixValues : public FunctionProducer
{
public:
//    static constexpr int N_DIMS = FuncT::N;
//    using FunctionProducer<N_DIMS - N_FIXED>::N;

    FixValues(FuncT func, vector<size_t> poss, vector<double> const& vals)
       : FunctionProducer(func.nDims - poss.size()), mFunc(move(func)), mPoss(move(poss)), mVals(move(vals))
    {
        assert(mPoss.size() == mVals.size());
        assert(is_sorted(mPoss.begin(), mPoss.end()));
        assert(unique(mPoss.begin(), mPoss.end()) == mPoss.end());
    }

    double operator()(vect const& x) override
    {
        assert((size_t) x.rows() == nDims);

        return mFunc(transform(x));
    }

    vect grad(vect const& x) override
    {
        assert((size_t) x.rows() == nDims);

        auto grad = mFunc.grad(transform(x));
        vect result(nDims);
        for (size_t i = 0, j = 0, k = 0; i < mFunc.nDims; i++)
            if (i == mPoss[j])
                j++;
            else
                result(k++) = grad(i);
        return result;
    }

    matrix hess(vect const& x) override
    {
        assert((size_t) x.rows() == nDims);

        auto hess = mFunc.hess(transform(x));
        matrix result(nDims, nDims);
        for (size_t i1 = 0, j1 = 0, k1 = 0; i1 < mFunc.nDims; i1++)
            if (i1 == mPoss[j1])
                j1++;
            else {
                for (size_t i2 = 0, j2 = 0, k2 = 0; i2 < mFunc.nDims; i2++)
                    if (i2 == mPoss[j2])
                        j2++;
                    else {
                        result(k1, k2++) = hess(i1, i2);
                    }
                k1++;
            }

        return result;
    };

    vect transform(vect const& from) const
    {
        vect to(nDims + mPoss.size());
        for (size_t i = 0, j = 0, k = 0; i < mFunc.nDims; i++)
            if (i == mPoss[j])
                to(i) = mVals[j++];
            else
                to(i) = from(k++);
        return to;
    }

    vect backTransform(vect const& from) const
    {
        vect result(nDims);
        for (size_t i = 0, j = 0, k = 0; (int) i < from.rows(); i++)
            if (i == mPoss[k]) {
                assert(abs(from(i) - mVals[k]) < 1e-7);
                k++;
            }
            else {
                result(j++) = from(i);
            }
        return result;
    }

    vect fullTransform(vect const& from) const
    {
        return mFunc.fullTransform(transform(from));
    }

    FuncT const& getInnerFunction() const
    {
        return mFunc;
    }

    auto const& getFullInnerFunction() const
    {
        return mFunc.getFullInnerFunction();
    }

private:
    FuncT mFunc;
    vector<size_t> mPoss;
    vector<double> mVals;
};

template<typename FuncT>
auto fix(FuncT&& func, vector<size_t> const& poss, vector<double> const& vals)
{
    return FixValues<decay_t<FuncT>>(forward<FuncT>(func), poss, vals);
};

template<typename FuncT>
auto fixAtomSymmetry(FuncT&& func)
{
    return fix(forward<FuncT>(func), {0, 1, 2, 4, 5, 8}, {0., 0., 0., 0., 0., 0.});
};

template<typename FuncT>
auto fixAtomSymmetry(FuncT&& func, size_t a, size_t b, size_t c)
{
    return fix(forward<FuncT>(func), {a * 3, a * 3 + 1, a * 3 + 2, b * 3 + 1, b * 3 + 2, c * 3 + 2}, {0., 0., 0., 0., 0., 0.});
}

template<typename FuncT>
auto fixAtomTranslations(FuncT&& func)
{
    return fix(forward<FuncT>(func), {0, 1, 2, 4, 5}, {0., 0., 0., 0., 0.});
};

template<typename FuncT>
auto fixAtomSymmetry(FuncT&& func, vect const& pos)
{
    return fix(forward<FuncT>(func), {0, 1, 2, 4, 5, 8}, {pos(0), pos(1), pos(2), pos(4), pos(5), pos(8)});
};

vect rotateToFix(vect p);
vect rotateToXYZ(vect v, size_t a, size_t b, size_t c);
