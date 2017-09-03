#include "InputOutputUtils.h"

string toChemcraftCoords(vector<size_t> const& charges, vect p, string comment)
{
    stringstream result;
    result << charges.size() << endl << comment << endl;
    for (size_t i = 0; i < charges.size(); i++)
        result << boost::format("%1%\t%2$.11f\t%3$.11f\t%4$.11f") % charges[i] % p(i * 3 + 0) % p(i * 3 + 1) % p(i * 3 + 2) << endl;
    return result.str();
}