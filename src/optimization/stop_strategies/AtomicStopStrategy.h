#pragma once

#include "helper.h"
#include <typeinfo>

namespace optimization
{
    template<typename T, typename... StackTs>
    struct StackExtractor
    {
        using next_type = decay_t<decltype(((T*) nullptr)->getInnerFunction())>;
        using NextStactExtractor = StackExtractor<next_type, StackTs..., T>;

        using TupleType = typename NextStactExtractor::TupleType;
        using AtomicType = typename NextStactExtractor::AtomicType;

        static AtomicType const& extractAtomicFunc(T const& t)
        {
            return NextStactExtractor::extractAtomicFunc(t.getInnerFunction());
        }

        static vect applyTransformation(vect const& x, T const& t)
        {
            return NextStactExtractor::applyTransformation(t.transform(x), t.getInnerFunction());
        }
    };

    template<typename... StackTs>
    struct StackExtractor<GaussianProducer, StackTs...>
    {
        using TupleType = tuple<StackTs const& ...>;
        using AtomicType = GaussianProducer;

        static AtomicType const& extractAtomicFunc(AtomicType const& t)
        {
            return t;
        }

        static vect applyTransformation(vect const& x, AtomicType const& t)
        {
            return x;
        }
    };


    template<typename FuncT>
    class AtomicStopStrategy
    {
    public:
        using ExtractorType = StackExtractor<FuncT>;
        using TupleType = typename ExtractorType::TupleType;
        using AtomicType = typename ExtractorType::AtomicType;

        static constexpr int N = FuncT::N;

        AtomicStopStrategy(double maxForce, double rmsForce, double maxAtomDelta, double rmsAtomDelta,
                           FuncT const& func) : mMaxForce(maxForce), mRmsForce(rmsForce), mMaxAtomDelta(maxAtomDelta),
                                                mRmsAtomDelta(rmsAtomDelta), mFunc(func),
                                                mAtomicFunc(ExtractorType::extractAtomicFunc(func))
        {}

        bool check(vect const& v, double singleMax, double rmsMax, const char* header)
        {
            double sum = 0;
            size_t atomCnt = (size_t) v.rows() / 3;

            assert(atomCnt * 3 == (size_t) v.rows());

            for (size_t i = 0; i < atomCnt; i++) {
                double singleValue = v.template block<3, 1>(i * 3, 0).norm();
                if (singleValue >= singleMax) {
                    LOG_INFO("too large {} single value: {} > {} [{}]", header, singleValue, singleMax, v.transpose());

                    return false;
                }

                sum += sqr(singleValue);
            }

            if (sqrt(sum / atomCnt) >= rmsMax)
                LOG_INFO("too large {} rms value: {} > {} [{}]", header, sqrt(sum / atomCnt), rmsMax, v.transpose());

            return sqrt(sum / atomCnt) < rmsMax;
        }

        bool operator()(size_t iter, vect const& p, vect const& grad, vect const& delta)
        {
            auto x0 = ExtractorType::applyTransformation(p, mFunc);
            auto x1 = ExtractorType::applyTransformation(p + delta, mFunc);

            assert((x0 - mAtomicFunc.getLastPos()).norm() < 1e-7);

            return check(x1 - x0, mMaxAtomDelta, mRmsAtomDelta, "delta") &&
                   check(mAtomicFunc.getLastGrad(), mMaxForce, mRmsForce, "grad");
        }

    private:
        double mMaxForce;
        double mRmsForce;
        double mMaxAtomDelta;
        double mRmsAtomDelta;
        FuncT const& mFunc;
        AtomicType const& mAtomicFunc;
    };

    template<typename FuncT>
    auto makeAtomicStopStrategy(double maxForce, double rmsForce, double maxAtomDelta, double rmsAtomDelta,
                                FuncT const& func)
    {
        return AtomicStopStrategy<FuncT>(maxForce, rmsForce, maxAtomDelta, rmsAtomDelta, func);
    }

    template<typename FuncT>
    auto makeStandardAtomicStopStrategy(FuncT const& func)
    {
        return makeAtomicStopStrategy(0.000045, 0.00003, 0.0018, 0.0012, func);
    }
}