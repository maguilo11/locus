/*
 * TRROM_EpetraVectorTest.C
 *
 *  Created on: Sep 22, 2016
 */

#include <gtest/gtest.h>

#include <mpi.h>

#include "TRROM_EpetraVector.hpp"

namespace EpetraVectorTest
{

TEST(EpetraVector, size)
{
    const int index_base = 0;
    const int element_size = 1;
    const int global_num_elements = 20;
    Epetra_MpiComm comm(MPI_COMM_WORLD);
    Epetra_BlockMap map(global_num_elements, element_size, index_base, comm);
    trrom::EpetraVector x(map);

    EXPECT_EQ(map.NumMyElements(), x.size());
}

TEST(EpetraVector, fill)
{
    const int index_base = 0;
    const int element_size = 1;
    const int global_num_elements = 20;
    Epetra_MpiComm comm(MPI_COMM_WORLD);
    Epetra_BlockMap map(global_num_elements, element_size, index_base, comm);
    trrom::EpetraVector x(map);

    x.fill(42.);
    double tolerance = 1e-6;
    for(int local_index = 0; local_index < x.size(); ++local_index)
    {
        EXPECT_NEAR(42., x[local_index], tolerance);
    }
}

TEST(EpetraVector, sum)
{
    const int index_base = 0;
    const int element_size = 1;
    const int global_num_elements = 20;
    Epetra_MpiComm comm(MPI_COMM_WORLD);
    Epetra_BlockMap map(global_num_elements, element_size, index_base, comm);
    trrom::EpetraVector x(map);

    x.fill(42.);
    double tolerance = 1e-6;
    double output = x.sum();
    double gold = static_cast<double>(42.) * global_num_elements;
    EXPECT_NEAR(gold, output, tolerance);
}

TEST(EpetraVector, norm)
{
    const int index_base = 0;
    const int element_size = 1;
    const int global_num_elements = 20;
    Epetra_MpiComm comm(MPI_COMM_WORLD);
    Epetra_BlockMap map(global_num_elements, element_size, index_base, comm);
    trrom::EpetraVector x(map);

    x.fill(42.);
    double tolerance = 1e-6;
    double norm_of_vector_x = x.norm();
    EXPECT_NEAR(187.82971010998, norm_of_vector_x, tolerance);
}

TEST(EpetraVector, abs)
{
    const int index_base = 0;
    const int element_size = 1;
    const int global_num_elements = 20;
    Epetra_MpiComm comm(MPI_COMM_WORLD);
    Epetra_BlockMap map(global_num_elements, element_size, index_base, comm);
    trrom::EpetraVector x(map);
    x.fill(-42.);

    x.modulus();
    double tolerance = 1e-6;
    for(int local_index = 0; local_index < x.size(); ++local_index)
    {
        EXPECT_NEAR(42., x[local_index], tolerance);
    }
}

TEST(EpetraVector, dot)
{
    const int index_base = 0;
    const int element_size = 1;
    const int global_num_elements = 20;
    Epetra_MpiComm comm(MPI_COMM_WORLD);
    Epetra_BlockMap map(global_num_elements, element_size, index_base, comm);
    trrom::EpetraVector x(map);
    trrom::EpetraVector y(map);

    x.fill(2.);
    y.fill(2.);

    double tolerance = 1e-6;
    double output = x.dot(y);
    double gold = 80;
    EXPECT_NEAR(gold, output, tolerance);
}

TEST(EpetraVector, copy)
{
    const int index_base = 0;
    const int element_size = 1;
    const int global_num_elements = 20;
    Epetra_MpiComm comm(MPI_COMM_WORLD);
    Epetra_BlockMap map(global_num_elements, element_size, index_base, comm);
    trrom::EpetraVector x(map);
    trrom::EpetraVector y(map);

    x.fill(2.);
    y.fill(0.);
    double tolerance = 1e-6;
    y.update(1., x, 0.);
    for(int local_index = 0; local_index < x.size(); ++local_index)
    {
        EXPECT_NEAR(y[local_index], x[local_index], tolerance);
    }
}

TEST(EpetraVector, scale)
{
    const int index_base = 0;
    const int element_size = 1;
    const int global_num_elements = 20;
    Epetra_MpiComm comm(MPI_COMM_WORLD);
    Epetra_BlockMap map(global_num_elements, element_size, index_base, comm);
    trrom::EpetraVector x(map);
    x.fill(2.);

    x.scale(3);
    double tolerance = 1e-6;
    for(int local_index = 0; local_index < x.size(); ++local_index)
    {
        EXPECT_NEAR(6., x[local_index], tolerance);
    }
}

TEST(EpetraVector, create)
{
    const int index_base = 0;
    const int element_size = 1;
    const int global_num_elements = 20;
    Epetra_MpiComm comm(MPI_COMM_WORLD);
    Epetra_BlockMap map(global_num_elements, element_size, index_base, comm);
    trrom::EpetraVector x(map);
    x.fill(2.);

    double tolerance = 1e-6;
    std::shared_ptr<trrom::Vector<double>> y = x.create();
    for(int local_index = 0; local_index < x.size(); ++local_index)
    {
        EXPECT_NEAR(2., x[local_index], tolerance);
        EXPECT_NEAR(0., (*y)[local_index], tolerance);
    }
    EXPECT_EQ(x.size(), y->size());

    y->fill(3);
    for(int local_index = 0; local_index < x.size(); ++local_index)
    {
        EXPECT_NEAR(2., x[local_index], tolerance);
        EXPECT_NEAR(3., (*y)[local_index], tolerance);
    }
}

TEST(EpetraVector, cwiseProd)
{
    const int index_base = 0;
    const int element_size = 1;
    const int global_num_elements = 20;
    Epetra_MpiComm comm(MPI_COMM_WORLD);
    Epetra_BlockMap map(global_num_elements, element_size, index_base, comm);
    trrom::EpetraVector x(map);
    trrom::EpetraVector y(map);

    for(int local_index = 0; local_index < x.size(); ++local_index)
    {
        x[local_index] = 2. + static_cast<double>(local_index);
        y[local_index] = 4. + static_cast<double>(local_index);
    }

    double tolerance = 1e-6;
    x.elementWiseMultiplication(y);
    for(int local_index = 0; local_index < x.size(); ++local_index)
    {
        double gold = (2. + static_cast<double>(local_index)) * (4. + static_cast<double>(local_index));
        EXPECT_NEAR(gold, x[local_index], tolerance);
    }
}

TEST(EpetraVector, create2)
{
    const int index_base = 0;
    const int element_size = 1;
    const int global_num_elements = 20;
    Epetra_MpiComm comm(MPI_COMM_WORLD);
    Epetra_BlockMap map(global_num_elements, element_size, index_base, comm);
    trrom::EpetraVector x(map);
    x.fill(2.);

    int gold = static_cast<int>(global_num_elements / x.getNumProc());
    EXPECT_EQ(gold, x.size());
    double tolerance = 1e-6;
    for(int local_index = 0; local_index < x.size(); ++local_index)
    {
        EXPECT_NEAR(2., x[local_index], tolerance);
    }

    int new_num_global_entries = 40;
    std::shared_ptr<trrom::Vector<double>> y = x.create(new_num_global_entries);

    gold = static_cast<int>(new_num_global_entries / x.getNumProc());
    EXPECT_EQ(gold, y->size());
    for(int local_index = 0; local_index < y->size(); ++local_index)
    {
        EXPECT_NEAR(0., (*y)[local_index], tolerance);
    }

    y->fill(3);
    for(int local_index = 0; local_index < y->size(); ++local_index)
    {
        EXPECT_NEAR(3., (*y)[local_index], tolerance);
    }
}

}
