/*
 * TRROM_ReducedBasisSolutionCaseTest.cpp
 *
 *  Created on: Oct 25, 2016
 *      Author: maguilo
 */

#include <mpi.h>

#include "gtest/gtest.h"

#include "Epetra_Map.h"
#include "Teuchos_RCP.hpp"
#include "Epetra_MpiComm.h"
#include "Epetra_CrsMatrix.h"
#include "Epetra_MultiVector.h"

#include "TRROM_UtestUtils.hpp"
#include "TRROM_TeuchosSerialDenseSolver.hpp"
#include "TRROM_TeuchosSerialDenseVector.hpp"
#include "TRROM_TeuchosSerialDenseMatrix.hpp"

namespace ReducedBasisSolutionCaseTest
{

void fillTriDiagonalMatrix(const Epetra_Map & map_, Epetra_CrsMatrix & matrix_);
void fillRightHandSide(const Epetra_Map & map_, Epetra_MultiVector & multi_vector_);
void fillNonZerosArray(const Epetra_Map & map_, std::vector<int> & nonzeros_per_row_);

TEST(Epetra_CrsMatrix, constructor)
{
    Epetra_MpiComm comm(MPI_COMM_WORLD);
    if(comm.NumProc() > 1)
    {
        int local_num_equations = 3;
        int num_processors = comm.NumProc();
        int global_num_equations = local_num_equations * num_processors;

        Epetra_Map map(global_num_equations, local_num_equations, 0, comm);
        EXPECT_EQ(local_num_equations, map.NumMyElements());
        EXPECT_EQ(global_num_equations, map.NumGlobalElements());

        // Create an integer vector nonzeros_per_row that is used to build the Epetra Matrix.
        std::vector<int> nonzeros_per_row(local_num_equations);
        fillNonZerosArray(map, nonzeros_per_row);

        // Create an Epetra_Matrix
        Epetra_CrsMatrix A(Copy, map, nonzeros_per_row.data());
        fillTriDiagonalMatrix(map, A);
        A.FillComplete(false);
        EXPECT_FALSE(A.StorageOptimized());
        A.OptimizeStorage();
        EXPECT_TRUE(A.StorageOptimized());

        std::vector<int> my_global_elements(map.NumMyElements());
        map.MyGlobalElements(my_global_elements.data());
        for(int index = 0; index < map.NumMyElements(); ++index)
        {
            int global_row_index = my_global_elements[index];
            EXPECT_TRUE(A.NumGlobalEntries(global_row_index) == nonzeros_per_row[index]);
        }

        double gold = 129.1278436279;
        double tolerance = 1e-6;
        EXPECT_NEAR(gold, A.NormFrobenius(), tolerance);
    }
}

void fillRightHandSide(const Epetra_Map & map_, Epetra_MultiVector & multi_vector_)
{
    // Get update list and number of local equations from newly created Map
    std::vector<int> my_global_elements(map_.NumMyElements());
    map_.MyGlobalElements(my_global_elements.data());

    for(int vector_index = 0; vector_index < multi_vector_.NumVectors(); ++vector_index)
    {
        for(int local_row_index = 0; local_row_index < map_.NumMyElements(); ++local_row_index)
        {
            int global_row_index = my_global_elements[local_row_index];
            switch(global_row_index)
            {
                case 0:
                {
                    multi_vector_.ReplaceGlobalValue(global_row_index, vector_index, 45.);
                    break;
                }
                case 1:
                {
                    multi_vector_.ReplaceGlobalValue(global_row_index, vector_index, 43.);
                    break;
                }
                case 2:
                {
                    multi_vector_.ReplaceGlobalValue(global_row_index, vector_index, 33.);
                    break;
                }
                case 3:
                {
                    multi_vector_.ReplaceGlobalValue(global_row_index, vector_index, 67.);
                    break;
                }
                case 4:
                {
                    multi_vector_.ReplaceGlobalValue(global_row_index, vector_index, 89.);
                    break;
                }
                case 5:
                {
                    multi_vector_.ReplaceGlobalValue(global_row_index, vector_index, 76.);
                    break;
                }
            }
        }
    }
}

void fillNonZerosArray(const Epetra_Map & map_, std::vector<int> & nonzeros_per_row_)
{
    // Get update list and number of local equations from newly created Map
    std::vector<int> my_global_elements(map_.NumMyElements());
    map_.MyGlobalElements(my_global_elements.data());
    for(int index = 0; index < map_.NumMyElements(); ++index)
    {
        int global_row_index = my_global_elements[index];
        switch(global_row_index)
        {
            case 0:
            case 5:
            {
                nonzeros_per_row_[index] = 2;
                break;
            }
            case 1:
            case 2:
            case 3:
            case 4:
            {
                nonzeros_per_row_[index] = 3;
                break;
            }
        }
    }
}

void fillTriDiagonalMatrix(const Epetra_Map & map_, Epetra_CrsMatrix & matrix_)
{
    // Get update list and number of local equations from newly created Map
    std::vector<int> my_global_elements(map_.NumMyElements());
    map_.MyGlobalElements(my_global_elements.data());
    int local_num_equations = map_.NumMyElements();
    for(int index = 0; index < local_num_equations; ++index)
    {
        int global_row_index = my_global_elements[index];
        switch(global_row_index)
        {
            case 0:
            {
                int num_entries = 2;
                std::vector<double> values(num_entries);
                values[0] = 3.;
                values[1] = 23.;
                std::vector<int> global_columns(num_entries);
                global_columns[0] = global_row_index;
                global_columns[1] = global_row_index + 1;
                matrix_.InsertGlobalValues(global_row_index, num_entries, values.data(), global_columns.data());
                break;
            }
            case 1:
            {
                int num_entries = 3;
                std::vector<double> values(num_entries);
                values[0] = 1.;
                values[1] = 34.;
                values[2] = 43.;
                std::vector<int> global_columns(num_entries);
                global_columns[0] = global_row_index - 1;
                global_columns[1] = global_row_index;
                global_columns[2] = global_row_index + 1;
                matrix_.InsertGlobalValues(global_row_index, num_entries, values.data(), global_columns.data());
                break;
            }
            case 2:
            {
                int num_entries = 3;
                std::vector<double> values(num_entries);
                values[0] = 2.;
                values[1] = 55.;
                values[2] = 22.;
                std::vector<int> global_columns(num_entries);
                global_columns[0] = global_row_index - 1;
                global_columns[1] = global_row_index;
                global_columns[2] = global_row_index + 1;
                matrix_.InsertGlobalValues(global_row_index, num_entries, values.data(), global_columns.data());
                break;
            }
            case 3:
            {
                int num_entries = 3;
                std::vector<double> values(num_entries);
                values[0] = 3.;
                values[1] = 18.;
                values[2] = 3.;
                std::vector<int> global_columns(num_entries);
                global_columns[0] = global_row_index - 1;
                global_columns[1] = global_row_index;
                global_columns[2] = global_row_index + 1;
                matrix_.InsertGlobalValues(global_row_index, num_entries, values.data(), global_columns.data());
                break;
            }
            case 4:
            {
                int num_entries = 3;
                std::vector<double> values(num_entries);
                values[0] = 4.;
                values[1] = 13.;
                values[2] = 77.;
                std::vector<int> global_columns(num_entries);
                global_columns[0] = global_row_index - 1;
                global_columns[1] = global_row_index;
                global_columns[2] = global_row_index + 1;
                matrix_.InsertGlobalValues(global_row_index, num_entries, values.data(), global_columns.data());
                break;
            }
            case 5:
            {
                int num_entries = 2;
                std::vector<double> values(num_entries);
                values[0] = 5.;
                values[1] = 56.;
                std::vector<int> global_columns(num_entries);
                global_columns[0] = global_row_index - 1;
                global_columns[1] = global_row_index;
                matrix_.InsertGlobalValues(global_row_index, num_entries, values.data(), global_columns.data());
                break;
            }
        }
    }
}

}
