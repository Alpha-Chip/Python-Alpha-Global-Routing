#include <gtest/gtest.h>

#include "cpu/disjoin_set.hpp"

TEST(Cpu_Disjoin_Set, Make_Set_And_Find)
{
    CpuDisjointSet disjoin_set {};

    disjoin_set.make_set(1);
    disjoin_set.make_set(2);
    disjoin_set.make_set(3);

    EXPECT_EQ(disjoin_set.find(1), 1);
    EXPECT_EQ(disjoin_set.find(2), 2);
    EXPECT_EQ(disjoin_set.find(3), 3);
    EXPECT_THROW(disjoin_set.find(4), std::runtime_error);
}

TEST(Cpu_Disjoin_Set, Make_Set_Unite_And_Find)
{
    CpuDisjointSet disjoin_set {};

    disjoin_set.make_set(1);
    disjoin_set.make_set(2);
    disjoin_set.make_set(3);

    disjoin_set.union_sets(1, 2);

    EXPECT_EQ(disjoin_set.find(1), disjoin_set.find(2));
    EXPECT_EQ(disjoin_set.find(3), 3);
}

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}