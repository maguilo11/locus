#include <mpi.h>
#include <gtest/gtest.h>

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    testing::InitGoogleTest(&argc, argv);
    int return_value = RUN_ALL_TESTS();

    MPI_Finalize();

    return (return_value);
}
