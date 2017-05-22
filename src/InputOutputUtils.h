#pragma once

#include "helper.h"

template<typename StreamT>
vect readVect(StreamT&& stream)
{
    size_t nDims;
    stream >> nDims;

    vect v(nDims);
    for (size_t i = 0; i < nDims; i++)
        stream >> v(i);

    return v;
}

template<typename StreamT>
tuple<vector<size_t>, vect> readMolecule(StreamT&& stream)
{
    vector<size_t> charges;
    vector<Eigen::Vector3d> poss;

    size_t charge = 0;
    while (stream >> charge) {
        Eigen::Vector3d pos;
        for (size_t i = 0; i < 3; i++)
            stream >> pos(i);

        charges.push_back(charge);
        poss.push_back(pos);
    }

    vect pos(charges.size() * 3, 1);
    for (size_t i = 0; i < charges.size(); i++) {
        pos.block(i * 3, 0, 3, 1) = poss[i];
    }

    return make_tuple(charges, pos);
}


string toChemcraftCoords(vector<size_t> const& charges, vect p);

