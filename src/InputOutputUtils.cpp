#include "InputOutputUtils.h"

string toChemcraftCoords(vector<size_t> const& charges, vect p)
{
    stringstream result;
    for (size_t i = 0; i < charges.size(); i++)
        result << boost::format("%1%\t%2%\t%3%\t%4%") % charges[i] % p(i * 3 + 0) % p(i * 3 + 1) % p(i * 3 + 2) << endl;
    return result.str();
}
