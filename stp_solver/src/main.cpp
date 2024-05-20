#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include "cpu/mst.hpp"
#include "iograph.hpp"

#define __PROGRAM_VERSION__ "v1.1.5"

constexpr char INPUT_GRAPH_PATH[] = "./graph.txt";
constexpr char OUTPUT_GRAPH_PATH[] = "./mst.txt";
constexpr char STREAM[] = "no";
constexpr char DEVICE[] = "cpu";

inline std::unordered_map<std::string, std::string> parse_arguments(int argc, char const* argv[])
{
    if (argc % 2 == 0) {
        throw std::runtime_error("Invalid number of arguments. Arguments must be in key-value pairs.");
    }

    std::unordered_map<std::string, std::string> options {
        { "--graph", INPUT_GRAPH_PATH },
        { "--mst", OUTPUT_GRAPH_PATH },
        { "--device", DEVICE },
        { "--stream", STREAM }
    };

    for (int i = 1; i < argc; i += 2) {
        std::string key = argv[i];

        if (options.count(key) == 0) {
            throw std::runtime_error("Unknown argument: " + key);
        }

        std::string value = argv[i + 1];

        if (key == "--device" && value != "cpu" && value != "gpu") {
            throw std::runtime_error("Invalid device option: " + value + ". Choose 'cpu' or 'gpu'.");
        }

        if (key == "--stream" && value != "no" && value != "yes") {
            throw std::runtime_error("Invalid stream option: " + value + ". Choose 'no' or 'yes'.");
        }

        options[key] = value;
    }

    return options;
}

inline int32_t process_graph(const std::unordered_map<std::string, std::string>& t_options)
{
    std::cout << "Steiner tree problem.\n";
    std::cout << "Program version: " << __PROGRAM_VERSION__ << '\n';
    std::cout << "C++ Standard: " << __cplusplus << "\n\n";
    std::cout << "Program options:\n";
    std::cout << "[--graph] Path to the input graph file: " << t_options.at("--graph") << '\n';
    std::cout << "[--mst] Path to the output mst file: " << t_options.at("--mst") << '\n';
    std::cout << "[--device] Device to use: " << t_options.at("--device") << "\n";
    std::cout << "[--stream] Stream mode: " << t_options.at("--stream") << "\n\n";
    std::cout << std::flush;

    auto start = std::chrono::high_resolution_clock::now();

    ReadGraph reader {};
    WriteGraph writer {};
    std::unique_ptr<InGraph> in_graph {};
    std::unique_ptr<OutGraph> out_graph {};
    std::ifstream in_file(t_options.at("--graph"), std::ios::binary);

    if (!in_file) {
        throw std::runtime_error("Can't open the file: " + t_options.at("--graph"));
    }

    in_graph = reader(in_file);

    std::chrono::duration<double> read_time = std::chrono::high_resolution_clock::now() - start;
    std::cout << "Read time: " << read_time.count() << "s\n";

    if (t_options.at("--device") == "cpu") {
        CpuMST mst {};
        out_graph = mst(std::move(in_graph));
    }

    std::chrono::duration<double> solve_time = (std::chrono::high_resolution_clock::now() - start) - read_time;
    std::cout << "Solve time: " << solve_time.count() << "s\n";

    std::ofstream out_file(t_options.at("--mst"), std::ios::binary);

    if (!out_file) {
        throw std::runtime_error("Can't open the file: " + t_options.at("--mst"));
    }

    writer(std::move(out_graph), out_file, true);

    std::chrono::duration<double> write_time = (std::chrono::high_resolution_clock::now() - start) - (read_time + solve_time);
    std::cout << "Write time: " << write_time.count() << "s\n\n";
    std::cout << "Total time: " << (solve_time + read_time + write_time).count() << "s" << std::endl;

    return EXIT_SUCCESS;
};

inline int stream_graph(const std::unordered_map<std::string, std::string>& t_options)
{
    ReadGraph reader {};
    WriteGraph writer {};
    std::unique_ptr<InGraph> in_graph {};
    std::unique_ptr<OutGraph> out_graph {};

    while (true) {
        in_graph = reader(std::cin);

        if (t_options.at("--device") == "cpu") {
            CpuMST mst {};
            out_graph = mst(std::move(in_graph));
        }

        writer(std::move(out_graph), std::cout);
    }
}

int main(int32_t argc, char const* argv[])
{
    try {
        auto options = parse_arguments(argc, argv);

        if (options["--stream"] == "no") {
            return process_graph(options);
        } else {
            return stream_graph(options);
        }
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}