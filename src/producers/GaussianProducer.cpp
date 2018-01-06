#include "GaussianProducer.h"

string const GAUSSIAN_HEADER = "%RWF={0}rwf\n"
   "%Int={0}int\n"
   "%D2E={0}d2e\n"
   "%Scr={0}skr\n"
   "%NoSave\n"
   "%chk={0}chk\n"
   "%nproc={2}\n"
   "%mem={3}mb\n"
   "# B3lyp/3-21g nosym {1}\n"
   "\n"
   "\n"
   "0 1";

string const SCF_METHOD = "scf";
string const FORCE_METHOD = "force";
//string const FORCE_METHOD = "freq=ReadFC";
string const HESS_METHOD = "freq";
string const OPT_METHOD = "FOpt";

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

ifstream GaussianProducer::runGaussian(vect const& x, string const& method) const
{
    auto fileMask = createInputFile(x, method);

    if (system(format("GAUSS_SCRDIR={0} mg09D {0}input {0}output > /dev/null", fileMask).c_str())) {
        throw GaussianException(this_thread::get_id());
    }

    if (system(format("formchk {}chk.chk > /dev/null", fileMask).c_str())) {
        throw GaussianException(this_thread::get_id());
    }

    return ifstream(fileMask + "chk.fchk");
}

vect GaussianProducer::optimize(vect const& structure) const
{
    auto result = runGaussian(structure, OPT_METHOD);
    return parseStructure(result);
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

void GaussianProducer::setGaussianNProc(size_t nProc)
{
    mNProc = nProc;
}

void GaussianProducer::setGaussianMem(size_t mem)
{
    mMem = mem;
}

GaussianProducer const& GaussianProducer::getFullInnerFunction() const
{
    return *this;
}

GaussianProducer& GaussianProducer::getFullInnerFunction()
{
    return *this;
}

string GaussianProducer::createInputFile(vect const& x, string const& method) const
{
    string fileMask = format("./tmp/{}/", std::hash<std::thread::id>()(this_thread::get_id()));

    system(("mkdir -p " + fileMask).c_str());

    ofstream f(fileMask + "input");
    f.precision(30);
    f << format(GAUSSIAN_HEADER, fileMask, method, mNProc, mMem) << endl;

    for (size_t i = 0; i < mCharges.size(); i++) {
        f << mCharges[i];
        for (size_t j = 0; j < 3; j++)
            f << "\t" << fixed << x(i * 3 + j);
        f << endl;
    }
    f << endl;

    return fileMask;
}

double GaussianProducer::parseValue(ifstream& input) const
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

vect GaussianProducer::parseGrad(ifstream& input) const
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

matrix GaussianProducer::parseHess(ifstream& input) const
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

vect GaussianProducer::parseStructure(ifstream& input) const
{
    string s;
    while (!boost::starts_with(s, "Current cartesian coordinates"))
        getline(input, s);


    vect structure(nDims);
    for (size_t i = 0; i < nDims; i++) {
        input >> structure(i);
        structure(i) /= MAGIC_CONSTANT;
    }

    return structure;
}