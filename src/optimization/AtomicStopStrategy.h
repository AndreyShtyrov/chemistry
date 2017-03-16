#pragma once

#include "helper.h"
#include <typeinfo>

namespace optimization
{
//    template<typename T>
//    class HasInnerFunction
//    {
//        typedef char one;
//        typedef long two;
//
//        template<typename C>
//        static one test(decltype(&C::getInnerFunction));
//
//        template<typename C>
//        static two test(...);
//
//    public:
//        enum
//        {
//            value = sizeof(test<T>(0)) == sizeof(char)
//        };
//    };
//
//    template<typename T, bool Flag = HasInnerFunction<T>::value>
//    struct InnerFunctionExtractor;
//
//    template<typename T> using InnerFunctionExtractor_t = typename InnerFunctionExtractor<T>::type;
//
//    template<typename T>
//    struct InnerFunctionExtractor<T, true>
//    {
//        using next_type = decay_t<decltype(((T*) nullptr)->getInnerFunction())>
//        using type = InnerFunctionExtractor_t<next_type>;
//
//        static type const& extract(T const& t)
//        {
//            return InnerFunctionExtractor<next_type>::extract(t.getInnerFunction());
//        }
//    };
//
//    template<typename T>
//    struct InnerFunctionExtractor<T, false>
//    {
//        using type = T;
//
//        static type const& extract(T const& t)
//        {
//            return t;
//        }
//    };

    template<typename T, typename... StackTs>
    struct StackExtractor
    {
        using next_type = decay_t<decltype(((T*) nullptr)->getInnerFunction())>;
        using NextStactExtractor = StackExtractor<next_type, StackTs..., T>;

        using TupleType = typename NextStactExtractor::TupleType;
        using AtomicType = typename NextStactExtractor::AtomicType;

//        template<typename... ArgsT>
//        static auto extractStack(T const& t, ArgsT const& ... args)
//        {
//            return NextStactExtractor::extractStack(t.getInnerFunction(), args..., t);
//        }

        static auto extractAtomicFunc(T const& t)
        {
            return NextStactExtractor::extractAtomicFunc(t.getInnerFunction());
        }

        static auto applyTransformation(vect<T::N> x, T const& t)
        {
            return NextStactExtractor::applyTransformation(t.transform(x), t.getInnerFunction());
        }
    };

    template<int N, typename... StackTs>
    struct StackExtractor<GaussianProducer<N>, StackTs...>
    {
        using TupleType = tuple<StackTs const& ...>;
        using AtomicType = GaussianProducer<N>;

//        template<typename... ArgsT>
//        static auto extractStack(AtomicType const& t, ArgsT const& ... args)
//        {
//            return TupleType(args...);
//        }

        static auto extractAtomicFunc(AtomicType const& t)
        {
            return t;
        }

        static auto applyTransformation(vect<AtomicType::N> x, AtomicType const& t)
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
//                                                mFuncStack(ExtractorType::extractStack(func))
        {}

        bool check(vect<AtomicType::N> const& v, double singleMax, double rmsMax)
        {
            double sum = 0;
            size_t atomCnt = (size_t) v.rows() / 3;

            assert(atomCnt * 3 == (size_t) v.rows());
            cerr << atomCnt << endl;

            for (size_t i = 0; i < atomCnt; i++) {
                double singlValue = v.block(i * 3, 0, 3, 1).norm();
                if (singlValue >= singlValue)
                    return false;

                sum += sqr(singlValue);
            }

            return sqrt(sum / atomCnt) < rmsMax;
        }

        bool operator()(size_t iter, vect<N> const& p, vect<N> const& grad, vect<N> const& delta)
        {
//            auto x0 = back_transformation<0>(p);
//            auto x1 = back_transformation<0>(vect<N>(p + delta));
            auto x0 = ExtractorType::applyTransformation(p, mFunc);
            auto x1 = ExtractorType::applyTransformation(p + delta, mFunc);

            cerr << x0.transpose() << endl << mAtomicFunc.getLastPos().transpose() << endl
                 << (x0 - mAtomicFunc.getLastPos()).norm() << endl;
            assert((x0 - mAtomicFunc.getLastPos()).norm() < 1e-7);

            return check(mAtomicFunc.getLastGrad(), mMaxForce, mMaxAtomDelta) &&
                   check(x1 - x0, mMaxAtomDelta, mRmsAtomDelta);
        }

    private:
//        template<int I, int CUR_N>
//        auto back_transformation(vect<CUR_N> const& v, enable_if_t<I < sizeof...(FuncsT), int> _ = 0)
//        {
//            return back_transformation<I + 1>(get<I>(mFuncStack).transform(v));
//        }
//
//        template<int I, int CUR_N>
//        auto back_transformation(vect<CUR_N> const& v, enable_if_t<I == sizeof...(FuncsT), int> _ = 0)
//        {
//            return v;
//        }

        double mMaxForce;
        double mRmsForce;
        double mMaxAtomDelta;
        double mRmsAtomDelta;
        FuncT const& mFunc;
        AtomicType const& mAtomicFunc;
//        TupleType mFuncStack;

    };

    template<typename FuncT>
    auto make_atomic_stop_strategy(double maxForce, double rmsForce, double maxAtomDelta, double rmsAtomDelta, FuncT const& func)
    {
        return AtomicStopStrategy<FuncT>(maxForce, rmsForce, maxAtomDelta, rmsAtomDelta, func);
    }
}