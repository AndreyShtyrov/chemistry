#pragma once

#include "helper.h"

#include "linearAlgebraUtils.h"
#include "producers/AffineTransformation.h"

template<typename FuncT>
auto remove6LesserHessValues(FuncT&& func, vect const& structure)
{
    size_t n = func.nDims / 3;
    vector<vect> vs;
    for (size_t i = 0; i < 3; i++) {
        vect v(structure.size());

        for (size_t j = 0; j < n; j++)
            v.block(j * 3, 0, 3, 1) = eye(3, i);
        vs.push_back(normalized(v));
    }

    for (size_t i = 0; i < 3; i++) {
        vect v(structure.size());

        for (size_t j = 0; j < n; j++) {
            vect block = structure.block(j * 3, 0, 3, 1);

            block(i) = 0;
            if (i == 0)
                swap(block(1), block(2)), block(1) *= -1;
            else if (i == 1)
                swap(block(0), block(2)), block(0) *= -1;
            else
                swap(block(0), block(1)), block(0) *= -1;

            v.block(j * 3, 0, 3, 1) = block;
        }

        vs.push_back(normalized(v));
    }

    matrix basis(func.nDims, vs.size());
    for (size_t i = 0; i < vs.size(); i++)
        basis.block(0, i, func.nDims, 1) = vs[i];

    for (size_t i = vs.size(); i < func.nDims; i++) {
        auto v = makeRandomVect(func.nDims);
        vect x = basis.colPivHouseholderQr().solve(v);

        v = v - basis * x;
        basis = horizontalStack(basis, normalized(v));
    }

    auto transformed = makeAffineTransfomation(func, structure,
                                               basis.block(0, vs.size(), basis.rows(), basis.cols() - vs.size()));
    auto hess = transformed.hess(makeConstantVect(transformed.nDims, 0.));
    auto A = linearizationNormalization(hess);

    return makeAffineTransfomation(transformed, A);
}

template<typename FuncT>
auto remove6LesserHessValues2(FuncT&& func, vect const& structure)
{
    size_t n = func.nDims / 3;
    vector<vect> vs;
    for (size_t i = 0; i < 3; i++) {
        vect v(structure.size());

        for (size_t j = 0; j < n; j++)
            v.block(j * 3, 0, 3, 1) = eye(3, i);
        vs.push_back(normalized(v));
    }

    for (size_t i = 0; i < 3; i++) {
        vect v(structure.size());

        for (size_t j = 0; j < n; j++) {
            vect block = structure.block(j * 3, 0, 3, 1);

            block(i) = 0;
            if (i == 0)
                swap(block(1), block(2)), block(1) *= -1;
            else if (i == 1)
                swap(block(0), block(2)), block(0) *= -1;
            else
                swap(block(0), block(1)), block(0) *= -1;

            v.block(j * 3, 0, 3, 1) = block;
        }

        vs.push_back(normalized(v));
    }

    matrix basis(func.nDims, vs.size());
    for (size_t i = 0; i < vs.size(); i++)
        basis.block(0, i, func.nDims, 1) = vs[i];

    for (size_t i = vs.size(); i < func.nDims; i++) {
        auto v = makeRandomVect(func.nDims);
        vect x = basis.colPivHouseholderQr().solve(v);

        v = v - basis * x;
        basis = horizontalStack(basis, normalized(v));
    }

    return makeAffineTransfomation(func, structure, basis.block(0, vs.size(), basis.rows(), basis.cols() - vs.size()));
}
