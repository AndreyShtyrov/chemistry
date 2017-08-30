#pragma once

#include "helper.h"

#include "FunctionProducer.h"

template<typename FuncT>
class FixValues : public FunctionProducer
{
public:
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

        return transformGrad(mFunc.grad(transform(x)));
    }

    matrix hess(vect const& x) override
    {
        assert((size_t) x.rows() == nDims);

        return transformHess(mFunc.hess(transform(x)));
    };

    tuple<double, vect> valueGrad(vect const& x) override
    {
        assert((size_t) x.rows() == nDims);

        auto result = mFunc.valueGrad(transform(x));
        return make_tuple(get<0>(result), transformGrad(get<1>(result)));
    };

    tuple<double, vect, matrix> valueGradHess(vect const& x) override
    {
        assert((size_t) x.rows() == nDims);

        auto result = mFunc.valueGradHess(transform(x));
        return make_tuple(get<0>(result), transformGrad(get<1>(result)), transformHess(get<2>(result)));
    };

    vect transform(vect const& from) const
    {
        vect to(nDims + mPoss.size());
        for (size_t i = 0, j = 0, k = 0; i < mFunc.nDims; i++)
            if (j < mPoss.size() && i == mPoss[j])
                to(i) = mVals[j++];
            else
                to(i) = from(k++);
        return to;
    }

    vect backTransform(vect const& from) const
    {
        vect result(nDims);
        for (size_t i = 0, j = 0, k = 0; (int) i < from.rows(); i++)
            if (k < mPoss.size() && i == mPoss[k]) {
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

    vect transformGrad(vect const& grad)
    {
        vect transformed(nDims);
        for (size_t i = 0, j = 0, k = 0; i < mFunc.nDims; i++)
            if (j < mPoss.size() && i == mPoss[j])
                j++;
            else
                transformed(k++) = grad(i);
        return transformed;
    }

    matrix transformHess(matrix const& hess)
    {
        matrix transformed(nDims, nDims);
        for (size_t i1 = 0, j1 = 0, k1 = 0; i1 < mFunc.nDims; i1++)
            if (j1 < mPoss.size() && i1 == mPoss[j1])
                j1++;
            else {
                for (size_t i2 = 0, j2 = 0, k2 = 0; i2 < mFunc.nDims; i2++)
                    if (j2 < mPoss.size() && i2 == mPoss[j2])
                        j2++;
                    else {
                        transformed(k1, k2++) = hess(i1, i2);
                    }
                k1++;
            }
        return transformed;
    }
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
