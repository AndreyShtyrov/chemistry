#pragma once

#include "helper.h"

#include "FunctionProducer.h"

template<int N>
class GradientDescent
{
public:
    constexpr static double EPS = 1e-7;

    GradientDescent(double speed=1.) : mSpeed(speed)
    { }

    vector<vect<N>> operator()(FunctionProducer<N>& func, vect<N> p0)
    {
        history.clear();

        vector<vect<N>> path;

        for (int i = 0; ; i++) {
            cerr << p0.transpose() << " : " << func.grad(p0).transpose() << endl;

            if (p0.norm() > 100)
                break;

            path.push_back(p0);
            history.push_back(func(p0));

            auto grad = func.grad(p0);
            if (grad.norm() < EPS)
                break;
            p0 -= mSpeed * grad;

        }

        return path;
    }

    vector<double> history;

private:
    double mSpeed;
};

template<int N>
class MomentumGradientDescent
{
public:
    constexpr static double EPS = 1e-7;

    MomentumGradientDescent(double speed=1.0, double momentum=.9) : mS(speed), mM(momentum)
    { }

    vector<vect<N>> operator()(FunctionProducer<N>& func, vect<N> p0)
    {
        history.clear();

        vector<vect<N>> path;

        vect<N> g;
        g.setZero();

        for (int i = 0; ; i++) {
            path.push_back(p0);
            history.push_back(func(p0));

            auto grad = func.grad(p0);
            if (grad.norm() < EPS)
                break;

            g = mM * g + mS * grad;
            p0 -= g;
        }

        return path;
    }


    vector<double> history;

private:
    double mS;
    double mM;
};

template<int N>
class NesterovGradientDescent
{
public:
    constexpr static double EPS = 1e-7;

    NesterovGradientDescent(double speed=0.5, double momentum=.9) : mS(speed), mM(momentum)
    { }

    vector<vect<N>> operator()(FunctionProducer<N>& func, vect<N> p0)
    {
        history.clear();

        vector<vect<N>> path;

        vect<N> g;
        g.setZero();

        for (int i = 0; ; i++) {
            path.push_back(p0);
            history.push_back(func(p0));

            auto grad = func.grad(p0 - mM * g);
            if (grad.norm() < EPS)
                break;

            g = mM * g + mS * grad;
            p0 -= g;
        }

        return path;
    }

    vector<double> history;

private:
    double mS;
    double mM;
};

template<int N>
class Adagrad
{
public:
    constexpr static double EPS = 1e-7;
    constexpr static double E = 1e-8;

    Adagrad(double speed=1.0) : mS(speed)
    {
        mG.setZero();
    }

    vector<vect<N>> operator()(FunctionProducer<N>& func, vect<N> p0)
    {
        history.clear();

        vector<vect<N>> path;

        for (int i = 0; ; i++) {
            cerr << p0 << ' ';
            path.push_back(p0);
            history.push_back(func(p0));

            auto grad = func.grad(p0);
            if (grad.norm() < EPS)
                break;

            double e = E;
            if (i)
                p0 -= (1 / sqrt(mG + e) * grad.array()).matrix();
            mG += grad.array() * grad.array();
        }

        return path;
    }

    vector<double> history;

private:
    double mS;
    double mM;
    Eigen::Array<double, N, 1> mG;
};


template<int N>
class Adadelta
{
public:
    constexpr static double EPS = 1e-7;
    constexpr static double E = 1e-9;

    Adadelta(double decay=.99) : mD(decay)
    {
        mMeanGrad.setZero();
        mMeanDelta.setZero();
    }

    vector<vect<N>> operator()(FunctionProducer<N>& func, vect<N> p0)
    {
        history.clear();

        vector<vect<N>> path;

        cerr << endl;
        for (int i = 0; ; i++) {
            cerr << p0 << endl;
            path.push_back(p0);
            history.push_back(func(p0));

            auto grad = func.grad(p0);
            if (grad.norm() < EPS)
                break;

            if (!i)
                mMeanGrad = grad.array() * grad.array();
            mMeanGrad = mD * mMeanGrad + (1 - mD) * grad.array() * grad.array();
            cerr << (grad.array() * grad.array()) << ' ' << mMeanGrad << ' ' << mMeanDelta << endl;
            Eigen::Array<double, N, 1> delta = sqrt(mMeanDelta + (double) E) / sqrt(mMeanGrad + (double) E) * grad.array();
            p0 -= delta.matrix();
            if (!i)
                mMeanDelta = sqr(delta);
            mMeanDelta = mD * mMeanDelta + (1 - mD) * sqr(delta);
        }

        return path;
    }

    vector<double> history;

private:
    double mD;
    Eigen::Array<double, N, 1> mMeanGrad;
    Eigen::Array<double, N, 1>  mMeanDelta;
};

template<int N>
class Adam
{
public:
    constexpr static double EPS = 1e-7;

    explicit Adam(double speed=2., double beta1=.9, double beta2=0.999, double eps=1e-8)
            : mSpeed(speed), mBeta1(beta1), mBeta2(beta2), mEps(eps)
    {
        mMean.setZero();
        mStd2.setZero();
    }

    vector<vect<N>> operator()(FunctionProducer<N>& func, vect<N> p0)
    {
        history.clear();

        vector<vect<N>> path;

        cerr << endl;
        for (int i = 0; ; i++) {
//            cerr << p0 << ' ' << mMean << ' ' << mStd2 << endl;
            path.push_back(p0);
            history.push_back(func(p0));

            auto grad = func.grad(p0);
            if (grad.norm() < EPS)
                break;

            if (!i)
                mMean = grad.array(), mStd2 = grad.array() * grad.array();
            mMean = mBeta1 * mMean + (1 - mBeta1) * grad.array();
            mStd2 = mBeta2 * mStd2 + (1 - mBeta2) * grad.array() * grad.array();
            p0 -= (mMean / (sqrt(mStd2) + mEps) * grad.array()).matrix();
        }

        return path;
    }

    vector<double> history;

private:
    double mSpeed;
    double mBeta1;
    double mBeta2;
    double mEps;
    Eigen::Array<double, N, 1> mMean;
    Eigen::Array<double, N, 1>  mStd2;
};

template<int N>
class HessianGradientDescent
{
public:
    constexpr static double EPS = 1e-7;

    HessianGradientDescent(double speed=1.) : mSpeed(speed)
    { }

    vector<vect<N>> operator()(FunctionProducer<N>& func, vect<N> p0)
    {
        history.clear();

        vector<vect<N>> path;

        for (int i = 0; ; i++) {
            cerr << p0.transpose() << " : " << endl << endl << func.grad(p0).transpose() << endl;
            cerr << func.hess(p0) << endl << endl << endl;

            if (p0.norm() > 100000)
                break;

            path.push_back(p0);
            history.push_back(func(p0));

            auto grad = func.grad(p0);
            auto hess = func.hess(p0);
            if (grad.norm() < EPS)
                break;
            p0 -= mSpeed * hess.inverse() * grad;

        }

        return path;
    }

    vector<double> history;

private:
    double mSpeed;
};
