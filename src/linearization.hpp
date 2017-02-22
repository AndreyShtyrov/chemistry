#pragma once

#include "helper.h"

template<int N>
matrix<N, N> isqrt(matrix<N, N> m)
{
    for (int i = 0; i < N; i++)
        m(i, i) = 1. / sqrt(abs(m(i, i)));
    return m;
};

template<int N>
matrix<N, N> linearization(matrix<N, N> m)
{
    matrix<N, N> result = matrix<N, N>::Identity();

    for (size_t i = 0; i < N; i++) {
        assert(abs(m(i, i)) > 1e-9);

        matrix<N, N> trans;
        trans.setZero();

        for (size_t j = 0; j < N; j++)
            for (size_t k = 0; k < N; k++)
                if (j == i)
                    trans(i, k) = -m(i, k) / m(i, i);
                else if (j == k)
                    trans(j, j) = 1;
        trans(i, i) = 1;

        for (size_t j = i + 1; j < N; j++)
            for (size_t k = 0; k < N; k++)
                m(j, k) -= m(i, j) * m(i, k) / m(i, i);
        for (size_t j = 0; j < N; j++)
            if (j != i)
                m(i, j)= m(j, i) = 0;

        result = result * trans;
//        cout << "trans:\n" << trans << endl << "m:\n" << m << endl << endl;
    }

    result *= isqrt(m);

    return result;
}