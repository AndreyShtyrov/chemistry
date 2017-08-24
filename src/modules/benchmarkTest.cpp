#include "helper.h"

#include <gtest/gtest.h>

#include "producers/GaussianProducer.h"

string const PATTERN = "%%chk=%4%.chk\n"
   "%%nproc=%1%\n"
   "%%mem=%2%mb\n"
   "# B3lyp/3-21g nosym %3%\n"
   "\n"
   "\n"
   "0 1\n"
   "6\t0.000000000000000000000000000000\t0.000000000000000000000000000000\t0.000000000000000000000000000000\n"
   "6\t1.329407574910000056078729357978\t0.000000000000000000000000000000\t0.000000000000000000000000000000\n"
   "1\t-0.573933341279999953421508962492\t0.921778012560000026276441076334\t0.000000000000000000000000000000\n"
   "1\t-0.573933775450000016604690245003\t-0.921778567219999955817399950320\t-0.000004707069999999999810257837\n"
   "1\t1.903341015899999932869945951097\t0.921778658150000040905069909059\t0.000004367929999999999699671939\n"
   "1\t1.903341021419999945507584016013\t-0.921778294680000054306390211423\t-0.000003291940000000000028635132\n"
   "\n";


double getTimeFromNow(chrono::time_point<chrono::system_clock> const& timePoint)
{
    return chrono::duration<double>(chrono::system_clock::now() - timePoint).count();
}

void createInputAndExecute(string const &method, size_t nProc, size_t mem)
{
    string filemask = boost::str(boost::format("./tmp/tmp%1%") % std::hash<std::thread::id>()(this_thread::get_id()));
    auto localStartTime = chrono::system_clock::now();

    ofstream inputFile(filemask + ".in");
    inputFile << boost::format(PATTERN) % nProc % mem % method % filemask;
    inputFile.close();

    system(str(boost::format("mg09D %1%.in %1%.out > /dev/null") % filemask).c_str());
}

void runBenchmark(string const &method, size_t nProc, size_t mem, size_t iters, bool parallel)
{
    auto startTime = chrono::system_clock::now();

    if (parallel)
        #pragma omp parallel for
        for (size_t i = 0; i < iters; i++)
            createInputAndExecute(method, nProc, mem);
    else
        for (size_t i = 0; i < iters; i++)
            createInputAndExecute(method, nProc, mem);

    double duration = getTimeFromNow(startTime);
    LOG_INFO("{}.{}.{} {} per iter ({} total for {} iters", method, nProc, mem, duration / iters, duration, iters);
}

TEST(Benchmark, NonParallel)
{
    initializeLogger();

    for (string const& method : {"scf", "force", "freq"})
        for (size_t nProc : {1ul, 2ul, 4ul})
            for (size_t mem : {250ul, 1000ul})
                runBenchmark(method, nProc, mem, 100, false);
}

TEST(Benchmark, Parallel)
{
    initializeLogger();

    for (string const& method : {"scf", "force", "freq"})
        for (size_t nProc : {1ul, 2ul, 4ul})
            for (size_t mem : {250ul, 1000ul})
                runBenchmark(method, nProc, mem, 100, true);
}

TEST(Benchmark, ScfParallel)
{
    initializeLogger();
    runBenchmark("scf", 1, 1000, 1000, true);
}

TEST(Benchmark, ScfNonParallel)
{
    initializeLogger();
    runBenchmark("scf", 1, 1000, 1000, false);
}

