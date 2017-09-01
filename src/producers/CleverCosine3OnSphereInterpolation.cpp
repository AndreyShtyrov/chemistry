#include "CleverCosine3OnSphereInterpolation.h"

#include "linearAlgebraUtils.h"

CleverCosine3OnSphereInterpolation::CleverCosine3OnSphereInterpolation(size_t nDims, vector<double> const& values, vector<vect> const& directions)
   : FunctionProducer(nDims)
{
    size_t n = values.size();
    if (n) {
        vector<OnSphereCosineSupplement> temporarySupplements;
        for (size_t i = 0; i < n; i++)
            temporarySupplements.emplace_back(directions[i], 1.);

        matrix A(n, n);
        for (size_t i = 0; i < n; i++)
            for (size_t j = 0; j < n; j++)
                A(i, j) = temporarySupplements[j](directions[i]);

        vect b(n);
        for (size_t i = 0; i < n; i++)
            b(i) = values[i];

        vect x = A.colPivHouseholderQr().solve(b);

        LOG_INFO("\n{}\n\n{}\n\n{}", A, b.transpose(), x.transpose());

        for (size_t i = 0; i < n; i++)
            mSupplements.emplace_back(directions[i], x(i));
    }
}

double CleverCosine3OnSphereInterpolation::operator()(vect const& x)
{
    double value = 0;
    for (auto& supplement : mSupplements)
        value += supplement(x);
}

vect CleverCosine3OnSphereInterpolation::grad(vect const& x)
{
    auto grad = makeConstantVect(nDims, 0);
    for (auto& supplement : mSupplements)
        grad += supplement.grad(x);
    return grad;
}

matrix CleverCosine3OnSphereInterpolation::hess(vect const& x)
{
    auto hess = makeConstantMatrix(nDims, nDims, 0);
    for (auto& supplement : mSupplements)
        hess += supplement.hess(x);
    return hess;
}

tuple<double, vect> CleverCosine3OnSphereInterpolation::valueGrad(vect const& x)
{
    return FunctionProducer::valueGrad(x);
};

tuple<double, vect, matrix> CleverCosine3OnSphereInterpolation::valueGradHess(vect const& x)
{
    return FunctionProducer::valueGradHess(x);
};
