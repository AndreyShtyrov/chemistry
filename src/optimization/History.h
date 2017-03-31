#pragma once

#include "helper.h"

namespace optimization
{
    template<int N_DIMS>
    struct History
    {
        static constexpr int N = N_DIMS;

        vector<double> vals;
        vector<vect<N>> poss;
        vector<vect<N>> grads;
        vector<matrix<N, N>> hess;

        void nextPoint(vect const &pos)
        {
            poss.push_back(pos);

            cerr << pos.transpose() << ":" << endl;
        }

        void nextPoint(vect const &pos, double val)
        {
            nextPoint(pos);
            vals.push_back(val);

            cerr << "\t" << val << endl;
        }

        void nextPoint(vect const &pos, double val, vect const &grad)
        {
            nextPoint(pos, val);
            grads.push_back(grad);

            cerr << "\t" << grad.norm() << endl;
        }

        void nextPoint(vect const &pos, double val, vect const &grad, matrix const &hess)
        {
            nextPoint(pos, val, grad);
            hess.push_back(hess);

            auto A = linearization(hess);
            cerr.precision(5);
            cerr << fixed << A.transpose() * hess * A << endl << endl;
        }

        void clear()
        {
            poss.clear();
            vals.clear();
            grads.clear();
            hess.clear();
        }

        size_t size() const
        {
            return vals.size();
        }
    };
}