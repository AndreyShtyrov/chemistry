#include "inputOutputUtils.h"

string toChemcraftCoords(vector<size_t> const& charges, vect p, string comment)
{
    stringstream result;
    result << charges.size() << endl << comment << endl;
    for (size_t i = 0; i < charges.size(); i++)
        result << format("{}\t{:.11f}\t{:.11f}\t{:.11f}", charges[i], p(i * 3 + 0), p(i * 3 + 1), p(i * 3 + 2));
    return result.str();
}

string print(vect const& v, size_t precision)
{
    stringstream ss;
    ss.precision(precision);
    for (size_t i = 0; i < v.size(); i++) {
        ss << fixed << v(i);
        if (i + 1 < v.size())
            ss << ", ";
    }
    return ss.str();
}
