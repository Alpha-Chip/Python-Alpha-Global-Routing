#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>

#include "iograph.hpp"

TEST(Input_Graph, Parse_File)
{
    ReadGraph reader {};

    if (std::filesystem::exists("../../tests/files/Track1/instance001.gr")) {
        std::ifstream in_file("../../tests/files/Track1/instance001.gr", std::ios::binary);

        if (!in_file) {
            throw std::runtime_error("Can't open the file: ../../tests/files/Track1/instance001.gr");
        }

        std::unique_ptr<InGraph> in_graph = reader(in_file);

        EXPECT_TRUE(!in_graph->adj_list.nodes.empty());
        EXPECT_TRUE(!in_graph->terminal_nodes.empty());
    } else {
        FAIL() << "File does not exist: ../../tests/files/Track1/instance001.gr\n";
    }
}

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}