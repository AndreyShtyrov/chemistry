#include "helper.h"

#include "PythongraphicsFramework.h"

//thread_local mt19937 randomGen;
//thread_local mt19937 randomGen(std::hash<std::thread::id>()(std::this_thread::get_id()));
thread_local mt19937 randomGen(random_device{}());

PythongraphicsFramework framework("func.out");
shared_ptr<spdlog::logger> logger;

void initializeLogger()
{
    system("mkdir -p tmp");
    system("mkdir -p logs");

    vector<spdlog::sink_ptr> sinks;
    auto stdout_sink = spdlog::sinks::stdout_sink_mt::instance();
    auto color_sink = std::make_shared<spdlog::sinks::ansicolor_sink>(stdout_sink);
    color_sink->set_level(spdlog::level::debug);
    sinks.push_back(color_sink);
//    sinks.push_back(stdout_sink);
    sinks.push_back(make_shared<spdlog::sinks::daily_file_sink_st>("logs/log", 0, 0));

    logger = make_shared<spdlog::logger>("logger", sinks.begin(), sinks.end());
    logger->set_pattern("[%H:%M:%S %t] %v");
    logger->set_error_handler([](string const& msg) { throw spdlog::spdlog_ex(msg); });

    logger->set_level(spdlog::level::debug);
    //todo: be careful with this flush strategy
    logger->flush_on(spdlog::level::debug);

}