#pragma once

#include <boost/algorithm/string/predicate.hpp>

#include "FunctionProducer.h"

string const GAUSSIAN_SCF_HEADER = "%nproc=3\n"
   "%chk=chk\n"
   "%mem=1000mb\n"
   "# B3lyp/3-21g nosym scf\n"
   "\n"
   "\n"
   "0 1";


string const GAUSSIAN_FORCE_HEADER = "%nproc=3\n"
   "%chk=chk\n"
   "%mem=1000mb\n"
   "# B3lyp/3-21g nosym force\n"
   "\n"
   "\n"
   "0 1";

string const GAUSSIAN_HESS_HEADER = "%nproc=3\n"
   "%chk=chk\n"
   "%mem=1000mb\n"
   "# B3lyp/3-21g nosym freq\n"
   "\n"
   "\n"
   "0 1";

class GaussianException : public exception
{ };

template<typename T>
class Cache
{
public:
    Cache() : mIsEmpty(true)
    { }

    template<typename U>
    explicit Cache(U&& value) : mIsEmpty(false), mValue(forward<U>(value))
    { }

    void set()
    {
        mIsEmpty = false;
    }

    T const& get() const
    {
        return mValue;
    }

    T& get()
    {
        return mValue;
    }

    void clear()
    {
        mIsEmpty = true;
    }

    bool empty() const
    {
        return mIsEmpty;
    }

private:
    bool mIsEmpty;
    T mValue;
};

class GaussianProducer : public FunctionProducer
{
public:
    static constexpr double MAGIC_CONSTANT = 1.88972585931612435672;

    GaussianProducer(vector<size_t> charges) : FunctionProducer(charges.size() * 3), mLastPos(nDims), mCharges(move(charges))
    { }

    virtual double operator()(vect const& x)
    {
        if (testCache(mValue, x))
            return mValue.get();
        processNewPos(x, false, false);
        return mValue.get();
    }

    virtual vect grad(vect const& x)
    {
        if (testCache(mGrad, x))
            return mGrad.get();
        processNewPos(x, true, false);
        return mGrad.get();
    }

    virtual matrix hess(vect const& x)
    {
        if (testCache(mHess, x))
            return mHess.get();
        processNewPos(x, true, true);
        return mHess.get();
    };

    vect const& getLastPos() const
    {
        return mLastPos;
    }

    double const& getLastValue() const
    {
        return mValue.get();
    }

    vect const& getLastGrad() const
    {
        return mGrad.get();
    }

    matrix const& getLastHess() const
    {
        return mHess.get();
    }

    vector<size_t> const& getCharges() const
    {
        return mCharges;
    }

private:
    vect mLastPos;
    Cache<double> mValue;
    Cache<vect> mGrad;
    Cache<matrix> mHess;
    vector<size_t> mCharges;

    template<typename T>
    bool testCache(Cache<T> const& cache, vect const& x)
    {
        return mLastPos == x && !cache.empty();
    }

    void createInputFile(vect const& x, bool withGrad, bool withHess)
    {
        ofstream f("tmp.inp");
        f.precision(30);
        if (withHess)
            f << GAUSSIAN_HESS_HEADER << endl;
        else if (withGrad)
            f << GAUSSIAN_FORCE_HEADER  << endl;
        else
            f << GAUSSIAN_SCF_HEADER << endl;

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
            throw GaussianException();
        }
    }

    void parseOutput(bool withGrad, bool withHess)
    {
        ifstream f("chk.fchk");

        string s;
        while (!boost::starts_with(s, "Total Energy"))
            getline(f, s);
        stringstream ss(s);
        ss >> s >> s >> s;

        ss >> mValue.get();
        mValue.set();

        if (withGrad) {
            while (!boost::starts_with(s, "Cartesian Gradient"))
                getline(f, s);

            vect& grad = mGrad.get();
            grad = vect(nDims);
            for (size_t i = 0; i < nDims; i++) {
                f >> grad(i);
                grad(i) *= MAGIC_CONSTANT;
            }
            mGrad.set();
        }

        if (withHess) {
            while (!boost::starts_with(s, "Cartesian Force Constants"))
                getline(f, s);

            matrix& hess = mHess.get();
            hess = matrix(nDims, nDims);
            for (size_t i = 0; i < nDims; i++)
                for (size_t j = 0; j <= i; j++) {
                    f >> hess(i, j);
                    hess(i, j) *= MAGIC_CONSTANT * MAGIC_CONSTANT;
                    hess(j, i) = hess(i, j);
                }
            mHess.set();
        }
    }

    void processNewPos(vect const& x, bool withGrad, bool withHess)
    {
        mValue.clear();
        mGrad.clear();
        mHess.clear();

        mLastPos = x;
        createInputFile(x, withGrad, withHess);
        runGaussian();
        parseOutput(withGrad, withHess);
    }
};

GaussianProducer readMolecule(istream& input)
{
    size_t cnt;
    input >> cnt;

    vector<size_t> charges(cnt);
    for (size_t i = 0; i < cnt; i++) {
        input >> charges[i];
    }

    return GaussianProducer(charges);
}