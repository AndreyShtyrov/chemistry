#pragma once

#include <boost/algorithm/string/predicate.hpp>

#include "FunctionProducer.h"
#include "InputOutputUtils.h"

string const GAUSSIAN_HEADER = "%%chk=%1%.chk\n"
        "# B3lyp/3-21g nosym %2%\n"
        "%%nproc=%3%\n"
        "%%mem=%4%mb\n"
        "\n"
        "\n"
        "0 1";

string const SCF_METHOD = "scf";
string const FORCE_METHOD = "force";
string const HESS_METHOD = "freq";
//
//string const GAUSSIAN_SCF_HEADER = "%nproc=3\n"
//   "%chk=./tmp/chk\n"
//   "%mem=1000mb\n"
//   "# B3lyp/3-21g nosym scf\n"
//   "\n"
//   "\n"
//   "0 1";
//
//
//string const GAUSSIAN_FORCE_HEADER = "%nproc=3\n"
//   "%chk=./tmp/chk\n"
//   "%mem=1000mb\n"
//   "# B3lyp/3-21g nosym force\n"
//   "\n"
//   "\n"
//   "0 1";
//
//string const GAUSSIAN_HESS_HEADER = "%nproc=3\n"
//   "%chk=%1%\n"
//   "%mem=1000mb\n"
//   "# B3lyp/3-21g nosym freq\n"
//   "\n"
//   "\n"
//   "0 1";

class GaussianException : public exception {
};

template<typename T>
class Cache {
public:
    Cache() : mIsEmpty(true) {}

    template<typename U>
    explicit Cache(U &&value) : mIsEmpty(false), mValue(forward<U>(value)) {}

    void set() {
        mIsEmpty = false;
    }

    T const &get() const {
        return mValue;
    }

    T &get() {
        return mValue;
    }

    void clear() {
        mIsEmpty = true;
    }

    bool empty() const {
        return mIsEmpty;
    }

private:
    bool mIsEmpty;
    T mValue;
};

class GaussianProducer : public FunctionProducer {
public:
    static constexpr double MAGIC_CONSTANT = 1.88972585931612435672;

    GaussianProducer(vector<size_t> charges, size_t nProc = 1, size_t mem = 1000) :
            FunctionProducer(charges.size() * 3), mLastPos(nDims), mCharges(move(charges)), mNProc(nProc), mMem(mem)
    {}

    virtual double operator()(vect const &x) {
        assert((size_t) x.rows() == nDims);

        if (testCache(mValue, x))
            return mValue.get();
        processNewPos(x, false, false);
        return mValue.get();
    }

    virtual vect grad(vect const &x) {
        assert((size_t) x.rows() == nDims);

        if (testCache(mGrad, x))
            return mGrad.get();
        processNewPos(x, true, false);
        return mGrad.get();
    }

    virtual matrix hess(vect const &x) {
        assert((size_t) x.rows() == nDims);

        if (testCache(mHess, x))
            return mHess.get();
        processNewPos(x, true, true);
        return mHess.get();
    };

    vect const &getLastPos() const {
        return mLastPos;
    }

    double const &getLastValue() const {
        return mValue.get();
    }

    vect const &getLastGrad() const {
        return mGrad.get();
    }

    matrix const &getLastHess() const {
        return mHess.get();
    }

    vector<size_t> const &getCharges() const {
        return mCharges;
    }

    vect transform(vect from) const {
        return from;
    }

    vect fullTransform(vect from) const {
        return from;
    }

    GaussianProducer const &getFullInnerFunction() const {
        return *this;
    }

private:
    vect mLastPos;

    size_t mNProc;
    size_t mMem;

    Cache<double> mValue;
    Cache<vect> mGrad;
    Cache<matrix> mHess;
    vector<size_t> mCharges;

    template<typename T>
    bool testCache(Cache<T> const &cache, vect const &x) {
        return !cache.empty() && mLastPos == x;
    }

    string createInputFile(vect const &x, bool withGrad, bool withHess) {
        string filemask = boost::str(
                boost::format("./tmp/tmp%1%") % std::hash<std::thread::id>()(this_thread::get_id()));
        LOG_INFO("GaussianProducer using file with mask {}", filemask);

        ofstream f(filemask + ".in");
        f.precision(30);
        auto const& method = withHess ? HESS_METHOD : (withGrad ? FORCE_METHOD : SCF_METHOD);
        f << boost::format(GAUSSIAN_HEADER) % filemask % method % mNProc % mMem << endl;

        for (size_t i = 0; i < mCharges.size(); i++) {
            f << mCharges[i];
            for (size_t j = 0; j < 3; j++)
                f << "\t" << fixed << x(i * 3 + j);
            f << endl;
        }
        f << endl;

        return filemask;
    }

    void runGaussian(string const &filemask) {
        if (system(boost::str(boost::format("mg09D %1%.in %1%.out > /dev/null") % filemask).c_str())
            || system(boost::str(boost::format("formchk %1%.chk > /dev/null") % filemask).c_str())) {
            throw GaussianException();
        }
    }

    void parseOutput(string const &filemask, bool withGrad, bool withHess) {
//        string filemask = "";
        ifstream f(str(boost::format("%1%.fchk") % filemask));

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

            vect &grad = mGrad.get();
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

            matrix &hess = mHess.get();
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

    void processNewPos(vect const &x, bool withGrad, bool withHess) {
        mValue.clear();
        mGrad.clear();
        mHess.clear();

        mLastPos = x;
        auto filemask = createInputFile(x, withGrad, withHess);
        runGaussian(filemask);
        parseOutput(filemask, withGrad, withHess);
//        parseOutput(withGrad, withHess);
    }
};

