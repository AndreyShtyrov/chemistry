#include "GaussianProducer.h"

string const GAUSSIAN_HEADER = "%%RWF=%1%rwf\n"
   "%%Int=%1%int\n"
   "%%D2E=%1%d2e\n"
   "%%Scr=%1%skr\n"
   "%%NoSave\n"
   "%%chk=%1%chk\n"
   "%%nproc=%3%\n"
   "%%mem=%4%mb\n"
   "# B3lyp/3-21g nosym %2%\n"
   "\n"
   "\n"
   "0 1";

string const SCF_METHOD = "scf";
string const FORCE_METHOD = "force";
string const HESS_METHOD = "freq";


GaussianProducer::GaussianProducer(vector<size_t> charges, size_t nProc, size_t mem) : FunctionProducer(
   charges.size() * 3), mCharges(move(charges)), mNProc(nProc), mMem(mem)
{}

double GaussianProducer::operator()(vect const& x)
{
    assert((size_t) x.rows() == nDims);

    auto result = runGaussian(x, SCF_METHOD);
    return parseValue(result);
}

tuple<double, vect> GaussianProducer::valueGrad(vect const& x)
{
    assert((size_t) x.rows() == nDims);

    auto result = runGaussian(x, FORCE_METHOD);

    auto value = parseValue(result);
    auto grad = parseGrad(result);

    return make_tuple(value, grad);
}

tuple<double, vect, matrix> GaussianProducer::valueGradHess(vect const& x)
{
    assert((size_t) x.rows() == nDims);

    auto result = runGaussian(x, HESS_METHOD);

    auto value = parseValue(result);
    auto grad = parseGrad(result);
    auto hess = parseHess(result);

    return make_tuple(value, grad, hess);
}

vect GaussianProducer::grad(vect const& x)
{
    return get<1>(valueGrad(x));
}

matrix GaussianProducer::hess(vect const& x)
{
    return get<2>(valueGradHess(x));
};

ifstream GaussianProducer::runGaussian(vect const& x, string const& method)
{
    auto fileMask = createInputFile(x, method);

    if (system(boost::str(boost::format("GAUSS_SCRDIR=%1% mg09D %1%input %1%output > /dev/null") % fileMask).c_str())) {
        throw GaussianException(this_thread::get_id());
    }

    if (system(boost::str(boost::format("formchk %1%chk.chk > /dev/null") % fileMask).c_str())) {
        throw GaussianException(this_thread::get_id());
    }

    return ifstream(fileMask + "chk.fchk");
}

vector<size_t> const& GaussianProducer::getCharges() const
{
    return mCharges;
}

vect GaussianProducer::transform(vect from) const
{
    return from;
}

vect GaussianProducer::fullTransform(vect from) const
{
    return from;
}

GaussianProducer const& GaussianProducer::getFullInnerFunction() const
{
    return *this;
}

string GaussianProducer::createInputFile(vect const& x, string const& method)
{
    string fileMask = boost::str(boost::format("./tmp/%1%/") % std::hash<std::thread::id>()(this_thread::get_id()));

    system(("mkdir -p " + fileMask).c_str());

    ofstream f(fileMask + "input");
    f.precision(30);
    f << boost::format(GAUSSIAN_HEADER) % fileMask % method % mNProc % mMem << endl;

    for (size_t i = 0; i < mCharges.size(); i++) {
        f << mCharges[i];
        for (size_t j = 0; j < 3; j++)
            f << "\t" << fixed << x(i * 3 + j);
        f << endl;
    }
    f << endl;

    return fileMask;
}

double GaussianProducer::parseValue(ifstream& input)
{
    string s;
    while (!boost::starts_with(s, "Total Energy"))
        getline(input, s);
    stringstream ss(s);
    ss >> s >> s >> s;

    double value;
    ss >> value;

    return value;
}

vect GaussianProducer::parseGrad(ifstream& input)
{
    string s;
    while (!boost::starts_with(s, "Cartesian Gradient"))
        getline(input, s);

    vect grad(nDims);
    for (size_t i = 0; i < nDims; i++) {
        input >> grad(i);
        grad(i) *= MAGIC_CONSTANT;
    }

    return grad;
}

matrix GaussianProducer::parseHess(ifstream& input)
{
    string s;
    while (!boost::starts_with(s, "Cartesian Force Constants"))
        getline(input, s);

    matrix hess(nDims, nDims);
    for (size_t i = 0; i < nDims; i++)
        for (size_t j = 0; j <= i; j++) {
            input >> hess(i, j);
            hess(i, j) *= MAGIC_CONSTANT * MAGIC_CONSTANT;
            hess(j, i) = hess(i, j);
        }

    return hess;
}