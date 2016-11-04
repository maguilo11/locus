/*
 * DOTk_ParallelUtils.cpp
 *
 *  Created on: Apr 3, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#include <mpi.h>
#include <typeinfo>
#include <algorithm>
#include "DOTk_ParallelUtils.hpp"

namespace dotk
{

namespace parallel
{

void loadBalance(const int & global_dim_, const int & comm_size_, int* input_)
{
    int default_size = global_dim_ / comm_size_;
    std::fill(input_, input_ + comm_size_, default_size);

    int remainder = global_dim_ % comm_size_;

    if(remainder != 0)
    {
        for(int i = 0; i < remainder; ++ i)
        {
            input_[i] = default_size + 1;
        }
    }
}

void loadBalance(const int & nrows_, const int & ncols_, const int & comm_size_, int* row_counts_, int* data_counts_)
{
    int default_local_num_rows = (nrows_ / comm_size_);
    int default_local_size = default_local_num_rows * ncols_;
    std::fill(data_counts_, data_counts_ + comm_size_, default_local_size);
    std::fill(row_counts_, row_counts_ + comm_size_, default_local_num_rows);

    int remainder = nrows_ % comm_size_;
    if(remainder != 0)
    {
        for(int i = 0; i < remainder; ++ i)
        {
            row_counts_[i] = default_local_num_rows + 1;
            data_counts_[i] = default_local_size + ncols_;
        }
    }
}

MPI_Datatype mpiDataType(const std::type_info & type_id_)
{
    MPI_Datatype mpi_type_id = MPI_DOUBLE;

    if(type_id_ == typeid(signed int))
    {
        mpi_type_id = MPI_INT;
    }
    else if(type_id_ == typeid(signed long int))
    {
        mpi_type_id = MPI_LONG;
    }
    else if(type_id_ == typeid(signed long long int))
    {
        mpi_type_id = MPI_LONG_LONG;
    }
    else if(type_id_ == typeid(unsigned short int))
    {
        mpi_type_id = MPI_UNSIGNED_SHORT;
    }
    else if(type_id_ == typeid(unsigned int))
    {
        mpi_type_id = MPI_UNSIGNED;
    }
    else if(type_id_ == typeid(unsigned long int))
    {
        mpi_type_id = MPI_UNSIGNED_LONG;
    }
    else if(type_id_ == typeid(double))
    {
        mpi_type_id = MPI_DOUBLE;
    }
    else if(type_id_ == typeid(long double))
    {
        mpi_type_id = MPI_LONG_DOUBLE;
    }
    else if(type_id_ == typeid(float))
    {
        mpi_type_id = MPI_FLOAT;
    }

    return (mpi_type_id);
}

}

}
