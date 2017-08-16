#pragma once

#include "helper.h"

vect makeRandomVect(size_t n);

vect makeRandomVect(vect const& lowerBound, vect const& upperBound);

vect makeConstantVect(size_t n, double constant=0.);

vect eye(size_t n, size_t i);

matrix makeRandomMatrix(size_t rows, size_t cols);

matrix makeConstantMatrix(size_t rows, size_t cols, double constant=0.);

matrix identity(size_t nDims);

matrix identity(size_t rows, size_t cols);

matrix linearization(matrix m);

matrix linearizationNormalization(matrix m);

matrix rotationMatrix(vect u, vect v, double alpha);

matrix rotationMatrix(vect from, vect to);

vect randomVectOnSphere(size_t nDims, double r=1.);

vect projection(vect wich, vect to);

matrix singularValues(matrix m);

vect toDistanceSpace(vect v, bool sorted=true);

class RandomProjection
{
public:
    explicit RandomProjection(size_t n, size_t m=2) : mA(m, n) {
        mA.setRandom();
    }

    vect operator()(vect v) {
        assert(v.rows() == mA.cols());
        return mA * v;
    }

private:
    matrix mA;
};
