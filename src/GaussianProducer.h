#pragma once

#include <boost/algorithm/string/predicate.hpp>

#include "FunctionProducer.h"

template<int N_DIMS>
class GaussianProducer : public FunctionProducer<N_DIMS>
{
public:
    static constexpr double MAGIC_CONSTANT = 1.88972585931612435672;

    using FunctionProducer<N_DIMS>::N;

    GaussianProducer(vector<size_t> const& charges) : mCharges(charges)
    {
        assert(charges.size() * 3 == N);
    }

    virtual double operator()(vect<N> const& x)
    {
        processNewPos(x);
        return mLastValue;
    }

    virtual vect<N> grad(vect<N> const& x)
    {
        processNewPos(x);
        return mLastGrad;
    }

    virtual matrix<N, N> hess(vect<N> const& x)
    {
        processNewPos(x);
        return mLastHess;
    };

    vect<N> const& getLastPos() const
    {
        return mLastPos;
    }

    double getLastValue() const
    {
        return mLastValue;
    }

    vect<N> getLastGrad() const
    {
        return mLastGrad;
    }

    matrix<N, N> getLastHess() const
    {
        return mLastHess;
    }

public:
    vector<size_t> mCharges;

    void createInputFile(vect<N> const& x)
    {
        ofstream f("tmp.inp");
        f.precision(7);
        f << "%nproc=5\n"
                "%chk=chk\n"
                "%mem=1000mb\n"
                "# B3lyp/3-21g nosym  freq\n"
                "\n"
                "hessian\n"
                "\n"
                "0 1" << endl;
        for (size_t i = 0; i < mCharges.size(); i++) {
            f << mCharges[i];
            for (size_t j = 0; j < 3; j++)
                f << "\t" << fixed << x(i * 3 + j);
            f << endl;
        }
        f << endl;
    }

    void runGaussian()
    {
        if (system("mg09D tmp.inp tmp.out > /dev/null") || system("formchk chk.chk > /dev/null")) {
            cerr << "Error" << endl;
            exit(-1);
        }
    }

    void parseOutput()
    {
        ifstream f("chk.fchk");

        string s;
        while (!boost::starts_with(s, "Total Energy"))
            getline(f, s);
        stringstream ss(s);
        ss >> s >> s >> s;
        ss >> mLastValue;

        while (!boost::starts_with(s, "Cartesian Gradient"))
            getline(f, s);
        for (size_t i = 0; i < N; i++) {
            f >> mLastGrad(i);
            mLastGrad(i) *= MAGIC_CONSTANT;
        }

        while (!boost::starts_with(s, "Cartesian Force Constants"))
            getline(f, s);
        for (size_t i = 0; i < N; i++)
            for (size_t j = 0; j <= i; j++) {
                f >> mLastHess(i, j);
                mLastHess(i, j) *= MAGIC_CONSTANT * MAGIC_CONSTANT;
                mLastHess(j, i) = mLastHess(i, j);
            }
    }

    void processNewPos(vect<N> const& x)
    {
        if (mLastPos != x) {
            mLastPos = x;
            createInputFile(x);
            runGaussian();
            parseOutput();
            system("rm *.chk");
        }
    }

    vect<N> mLastPos;
    double mLastValue;
    vect<N> mLastGrad;
    matrix<N, N> mLastHess;
};