#pragma once

#include "helper.h"

vect makeRandomVect(size_t n);

vect makeRandomVect(vect const& lowerBound, vect const& upperBound);

vect makeConstantVect(size_t n, double constant);

vect eye(size_t n, size_t i);

matrix makeRandomMatrix(size_t rows, size_t cols);

matrix makeConstantMatrix(size_t rows, size_t cols, double constant);

matrix identity(size_t nDims);

matrix identity(size_t rows, size_t cols);

matrix linearization(matrix m);

matrix linearizationNormalization(matrix m);

matrix rotationMatrix(vect u, vect v, double alpha);

matrix rotationMatrix(vect from, vect to);

vect randomVectOnSphere(size_t nDims, double r=1.);
