/*
 * DOTk_MpiArray.cpp
 *
 *  Created on: Apr 4, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#include <typeinfo>
#include <algorithm>
#include <tr1/memory>
#include "DOTk_MpiArray.hpp"
#include "DOTk_ParallelUtils.hpp"

namespace dotk
{

namespace mpi
{

template<class Type>
array<Type>::array(int global_dim_, Type value_) :
        m_LocalDim(0),
        m_GlobalDim(global_dim_),
        m_Comm(MPI_COMM_WORLD),
        m_Data(NULL),
        m_LocalCounts(NULL),
        m_Displacements(NULL)
{
    this->allocate(global_dim_);
    this->fill(value_);
}

template<class Type>
array<Type>::array(MPI_Comm comm_, int global_dim_, Type value_) :
        m_LocalDim(0),
        m_GlobalDim(global_dim_),
        m_Comm(comm_),
        m_Data(NULL),
        m_LocalCounts(NULL),
        m_Displacements(NULL)
{
    this->allocate(global_dim_);
    this->fill(value_);
}

template<class Type>
array<Type>::~array()
{
    delete[] m_LocalCounts;
    m_LocalCounts = NULL;
    delete[] m_Displacements;
    m_Displacements = NULL;
    delete[] m_Data;
    m_Data = NULL;
}

template<class Type>
void array<Type>::scale(const Type & alpha_)
{
    size_t dim = this->size();
    for(size_t i = 0; i < dim; ++i)
    {
        m_Data[i] = alpha_ * m_Data[i];
    }
}

template<class Type>
void array<Type>::cwiseProd(const dotk::vector<Type> & input_)
{
    size_t dim = this->size();
    assert(dim == input_.size());
    for(size_t i = 0; i < dim; ++i)
    {
        m_Data[i] = m_Data[i] * input_[i];
    }
}

template<class Type>
void array<Type>::axpy(const Type & alpha_, const dotk::vector<Type> & input_)
{
    size_t dim = this->size();
    assert(dim == input_.size());
    for(size_t i = 0; i < dim; ++i)
    {
        m_Data[i] = alpha_ * input_[i] + m_Data[i];
    }
}

template<class Type>
Type array<Type>::max() const
{
    Type max = 0.;
    size_t dim = this->size();

    Type local_max = *std::max_element(m_Data, m_Data + dim);

    MPI_Datatype data_type = dotk::parallel::mpiDataType(typeid(Type));
    MPI_Allreduce(&local_max, &max, 1, data_type, MPI_MAX, m_Comm);

    return (max);
}

template<class Type>
Type array<Type>::min() const
{
    Type min = 0.;
    size_t dim = this->size();

    Type local_min = *std::min_element(m_Data, m_Data + dim);

    MPI_Datatype data_type = dotk::parallel::mpiDataType(typeid(Type));
    MPI_Allreduce(&local_min, &min, 1, data_type, MPI_MIN, m_Comm);

    return (min);
}

template<class Type>
void array<Type>::abs()
{
    size_t dim = this->size();
    for(size_t index = 0; index < dim; ++index)
    {
        m_Data[index] = m_Data[index] < static_cast<Type>(0.) ? -(m_Data[index]) : m_Data[index];
    }
}

template<class Type>
Type array<Type>::sum() const
{
    Type local_sum = 0.;
    size_t dim = this->size();
    MPI_Datatype data_type = dotk::parallel::mpiDataType(typeid(Type));

    for(size_t i = 0; i < dim; ++i)
    {
        local_sum += m_Data[i];
    }

    Type result = 0.;
    MPI_Allreduce(&local_sum, &result, 1, data_type, MPI_SUM, m_Comm);

    return (result);
}

template<class Type>
Type array<Type>::dot(const dotk::vector<Type> & input_) const
{
    Type value = 0.;
    Type local_sum = 0.;
    size_t dim = this->size();
    assert(dim == input_.size());
    MPI_Datatype data_type = dotk::parallel::mpiDataType(typeid(Type));

    for(size_t i = 0; i < dim; ++i)
    {
        local_sum += m_Data[i] * input_[i];
    }

    MPI_Allreduce(&local_sum, &value, 1, data_type, MPI_SUM, m_Comm);

    return (value);
}

template<class Type>
Type array<Type>::norm() const
{
    Type local_sum = 0.;
    Type global_sum = 0.;
    size_t dim = this->size();
    MPI_Datatype data_type = dotk::parallel::mpiDataType(typeid(Type));

    for(size_t i = 0; i < dim; ++i)
    {
        local_sum += m_Data[i] * m_Data[i];
    }

    MPI_Allreduce(&local_sum, &global_sum, 1, data_type, MPI_SUM, m_Comm);

    Type value = sqrt(global_sum);

    return (value);
}

template<class Type>
void array<Type>::fill(const Type & value_)
{
    size_t dim = this->size();
    for(size_t i = 0; i < dim; ++i)
    {
        m_Data[i] = value_;
    }
}

template<class Type>
void array<Type>::copy(const dotk::vector<Type> & input_)
{
    size_t dim = this->size();
    assert(dim == input_.size());
    for(size_t i = 0; i < dim; ++i)
    {
        m_Data[i] = input_[i];
    }
}

template<class Type>
void array<Type>::gather(Type* input_) const
{
    int my_rank;
    MPI_Comm_rank(m_Comm, &my_rank);

    size_t local_dim = this->size();
    Type* temp = new Type[local_dim];
    for(size_t i = 0; i < local_dim; ++i)
    {
        temp[i] = m_Data[i];
    }

    if(my_rank == 0)
    {
        int root = 0;
        MPI_Datatype data_type = dotk::parallel::mpiDataType(typeid(Type));
        MPI_Gatherv(temp, local_dim, data_type, input_, m_LocalCounts, m_Displacements, data_type, root, m_Comm);
    }
    else
    {
        int root = 0;
        MPI_Datatype data_type = dotk::parallel::mpiDataType(typeid(Type));
        MPI_Gatherv(temp, local_dim, data_type, NULL, m_LocalCounts, m_Displacements, data_type, root, m_Comm);
    }

    delete[] temp;
    temp = NULL;
}

template<class Type>
size_t array<Type>::size() const
{
    return (m_LocalDim);
}

template<class Type>
std::tr1::shared_ptr<dotk::vector<Type> > array<Type>::clone() const
{
    std::tr1::shared_ptr<dotk::mpi::array<Type> > vec(new dotk::mpi::array<Type>(m_Comm, m_GlobalDim));
    return (vec);
}

template<class Type>
Type & array<Type>::operator [](size_t index_)
{
    return (m_Data[index_]);
}

template<class Type>
const Type & array<Type>::operator [](size_t index_) const
{
    return (m_Data[index_]);
}

template<class Type>
dotk::types::container_t array<Type>::type() const
{
    return (dotk::types::MPI_ARRAY);
}

template<class Type>
size_t array<Type>::rank() const
{
    int rank_id = 0;
    MPI_Comm_rank(m_Comm, &rank_id);
    return (rank_id);
}

template<class Type>
void array<Type>::allocate(const int & global_dim_)
{
    int comm_size;
    MPI_Comm_size(m_Comm, &comm_size);

    m_LocalCounts = new int[comm_size];
    dotk::parallel::loadBalance(global_dim_, comm_size, m_LocalCounts);

    int offset = 0;
    m_Displacements = new int[comm_size];
    for(int rank = 0; rank < comm_size; ++ rank)
    {
        m_Displacements[rank] = offset;
        offset += m_LocalCounts[rank];
    }

    int my_rank;
    MPI_Comm_rank(m_Comm, &my_rank);

    Type* temp = NULL;
    if(my_rank == 0)
    {
        int root = 0;
        temp = new Type[global_dim_];
        m_LocalDim = m_LocalCounts[my_rank];
        m_Data = new Type[m_LocalDim];
        MPI_Datatype data_type = dotk::parallel::mpiDataType(typeid(Type));
        MPI_Scatterv(temp, m_LocalCounts, m_Displacements, data_type, m_Data, m_LocalDim, data_type, root, m_Comm);

        delete[] temp;
        temp = NULL;
    }
    else
    {
        int root = 0;
        m_LocalDim = m_LocalCounts[my_rank];
        m_Data = new Type[m_LocalDim];
        MPI_Datatype data_type = dotk::parallel::mpiDataType(typeid(Type));
        MPI_Scatterv(temp, m_LocalCounts, m_Displacements, data_type, m_Data, m_LocalDim, data_type, root, m_Comm);
    }
}

}

}
