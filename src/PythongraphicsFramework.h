#pragma once

#include "helper.h"

class PythongraphicsFramework
{
public:
    PythongraphicsFramework(string const& filePath) : mOutput(filePath), mCounter(0)
    {
        mOutput.precision(10);
    }

    string newPlot(string title="")
    {
        auto fig = next_var_name();
        mOutput << fig << " = plt.figure()" << endl << fig << ".suptitle('" << title << "')" << endl;

        auto axis = next_var_name();
        mOutput << axis << " = " << fig << ".gca()" << endl;

        return axis;
//        auto axis = next_var_name();
//        mOutput << axis << " = plt.figure().gca()" << endl;
//        return axis;
    }

    string new3dPlot(string title="")
    {
        auto fig = next_var_name();
        mOutput << fig << " = plt.figure()" << endl << fig << ".suptitle('" << title << "')" << endl;

        auto axis = next_var_name();
        mOutput << axis << " = " << fig << ".gca(projection='3d')" << endl;

        return axis;
    }

    void plot(string const& axis, vector<double> const& ys)
    {
        plot(axis, arange(ys.size()), ys);
    }

    void plot(string const& axis, vector<double> const& xs, vector<double> const& ys)
    {
        auto vxs = createArray(xs);
        auto vys = createArray(ys);
        mOutput << axis << ".plot(" << vxs << ", " << vys << ")" << endl;
    }

    template<typename MatrixT>
    void plot3d(string const& axis, MatrixT const& xs, MatrixT const& ys, MatrixT const& zs)
    {
        auto vxs = create_matrix(xs);
        auto vys = create_matrix(ys);
        auto vzs = create_matrix(zs);

        mOutput << axis << ".plot_surface(" << vxs << ", " << vys << ", " << vzs << ", cmap=cm.coolwarm)" << endl;
    }

    void contour(string const& axis, vector<vector<double>> const& xs, vector<vector<double>> const& ys, vector<vector<double>> const& zs, int cnt)
    {
        auto vxs = create_matrix(xs);
        auto vys = create_matrix(ys);
        auto vzs = create_matrix(zs);

        mOutput << axis << ".contour(" << vxs << ", " << vys << ", " << vzs << ", " << cnt << ", cmap=plt.cm.rainbow)" << endl;
    }

    void scatter(string const& axis, vector<double> const& xs, vector<double> const& ys, vector<double> const& zs)
    {
        auto vxs = createArray(xs);
        auto vys = createArray(ys);
        auto vzs = createArray(zs);

        mOutput << axis << ".scatter(" << vxs << ", " << vys << ", " << vzs << ", color='r')" << endl;
    }
    void scatter(string const& axis, vector<double> const& xs, vector<double> const& ys)
    {
        auto vxs = createArray(xs);
        auto vys = createArray(ys);

        mOutput << axis << ".scatter(" << vxs << ", " << vys << ", color='r')" << endl;
    }

//    void scatter(vector<double> const& xs, vector<double> const& ys)
//    {
//        auto vxs = create_array(xs);
//        auto vys = create_array(ys);
//        mOutput << "plt.plot(" << vxs << ", " << vys << ")" << endl;
//    }

    string createArray(vector<double> const& xs)
    {
        string var = "_" + to_string(next_counter());
        mOutput << var << " = [";
        for (auto val : xs) {
            mOutput << val << ",";
        }
        mOutput << "]" << endl;

        return var;
    }

    string create_matrix(vector<vector<double>> const& data)
    {
        auto var = next_var_name();

        mOutput << var << " = [";
        for (auto const& row : data) {
            mOutput << "[";
            for (auto const& val : row)
                mOutput << val << ", ";
            mOutput << "], ";
        }
        mOutput << "]" << endl;

        return var;
    }


private:
    ofstream mOutput;
    size_t mCounter;

    string next_var_name()
    {
        return "_" + to_string(next_counter());
    }

    size_t next_counter()
    {
        return mCounter++;
    }
};

inline vector<vector<double>> reshape(vector<double> data, size_t rows) {
    assert(data.size() % rows == 0);

    size_t cols = data.size() / rows;

    vector<vector<double>> reshaped(rows, vector<double>(cols));
    for (size_t i = 0; i < rows; i++)
        for (size_t j = 0; j < cols; j++)
            reshaped[i][j] = data[i * cols + j];

    return reshaped;
}

extern PythongraphicsFramework framework;