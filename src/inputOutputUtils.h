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
vect readVect(StreamT&& stream, size_t rows)
{
    vect v(rows);
    for (size_t i = 0; i < rows; i++)
        stream >> v(i);
    return v;
}

template<typename StreamT>
vector<size_t> readCharges(StreamT&& stream)
{
    size_t n;
    stream >> n;

    vector<size_t> charges(n);
    for (size_t i = 0; i < n; i++)
        stream >> charges[i];
    return charges;
}

template<typename StreamT>
tuple<vector<size_t>, vect> readChemcraft(StreamT&& stream)
{

    size_t cnt = 0;
    stream >> cnt;

    string comment;
    getline(stream, comment);
    getline(stream, comment);

    vector<size_t> charges(cnt);
    vect structure(cnt * 3);
    for (size_t i = 0; i < cnt; i++) {
        stream >> charges[i];
        structure.block(i * 3, 0, 3, 1) = readVect(stream, 3);
    }

    return make_tuple(charges, structure);
}

template<typename StreamT>
tuple<vector<vector<size_t>>, vector<vect>> readWholeChemcraft(StreamT&& stream)
{
    vector<vector<size_t>> charges;
    vector<vect> structures;

    size_t cnt = 0;

    while (stream >> cnt) {
        string comment;
        getline(stream, comment);
        getline(stream, comment);

        vector<size_t> currentCharges(cnt);
        vect structure(cnt * 3);
        for (size_t i = 0; i < cnt; i++) {
            stream >> currentCharges[i];
            structure.block(i * 3, 0, 3, 1) = readVect(stream, 3);
        }

        charges.push_back(currentCharges);
        structures.push_back(structure);
    }

    return make_tuple(charges, structures);
}

string toChemcraftCoords(vector<size_t> const& charges, vect p, string comment="");

string print(vect const& v, size_t precision=7);
