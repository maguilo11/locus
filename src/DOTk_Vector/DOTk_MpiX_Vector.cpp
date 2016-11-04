/*
 * DOTk_MpiX_Vector.cpp
 *
 *  Created on: Apr 4, 2015
 *      Author: Miguel A. Aguilo Valentin
 */



#include <vector>
#include <typeinfo>
#include <tr1/memory>
#include "DOTk_MpiX_Vector.hpp"
#include "DOTk_ParallelUtils.hpp"

namespace dotk
{

namespace mpix
{

template<class Type>
vector<Type>::vector(int global_dim_, int num_threads_, Type value_) :
        m_GlobalDim(global_dim_),
        m_NumThreads(num_threads_),
        m_Comm(MPI_COMM_WORLD),
        m_LocalCounts(NULL),
        m_Displacements(NULL),
        m_Data()
{
    this->allocate(global_dim_);
    this->fill(value_);
}

template<class Type>
vector<Type>::vector(MPI_Comm comm_, int global_dim_, int num_threads_, Type value_) :
        m_GlobalDim(global_dim_),
        m_NumThreads(num_threads_),
        m_Comm(comm_),
        m_LocalCounts(NULL),
        m_Displacements(NULL),
        m_Data()
{
    this->allocate(global_dim_);
    this->fill(value_);
}

template<class Type>
vector<Type>::~vector()
{
    delete[] m_LocalCounts;
    m_LocalCounts = NULL;
    delete[] m_Displacements;
    m_Displacements = NULL;
}

template<class Type>
void vector<Type>::scale(const Type & alpha_)
{
    size_t i;
    size_t dim = this->size();

    int thread_count = this->threads();
# pragma omp parallel num_threads(thread_count) \
default( none ) \
shared ( dim, alpha_ ) \
private ( i )

# pragma omp for
    for(i = 0; i < dim; ++i)
    {
        m_Data[i] = alpha_ * m_Data[i];
    }
}

template<class Type>
void vector<Type>::cwiseProd(const dotk::vector<Type> & input_)
{
    size_t i;
    size_t dim = this->size();
    assert(dim == input_.size());

    int thread_count = this->threads();
# pragma omp parallel num_threads(thread_count) \
default( none ) \
shared ( dim, input_ ) \
private ( i )

# pragma omp for
    for(i = 0; i < dim; ++i)
    {
        m_Data[i] = m_Data[i] * input_[i];
    }
}

template<class Type>
void vector<Type>::axpy(const Type & alpha_, const dotk::vector<Type> & input_)
{
    size_t i;
    size_t dim = this->size();
    assert(dim == input_.size());

    int thread_count = this->threads();
# pragma omp parallel num_threads(thread_count) \
default( none ) \
shared ( dim, input_, alpha_ ) \
private ( i )

# pragma omp for
    for(i = 0; i < dim; ++i)
    {
        m_Data[i] = alpha_ * input_[i] + m_Data[i];
    }
}

template<class Type>
Type vector<Type>::max() const
{
    size_t i;
    Type max_value = 0;
    Type local_max_value = 0;
    size_t dim = this->size();
    MPI_Datatype data_type = dotk::parallel::mpiDataType(typeid(Type));

    int thread_count = this->threads();
# pragma omp parallel num_threads(thread_count) \
default( none ) \
shared ( dim, local_max_value ) \
private ( i )

# pragma omp for reduction( max : local_max_value )
    for(i = 0; i < dim; ++i)
    {
        if(m_Data[i] > local_max_value)
        {
            local_max_value = m_Data[i];
        }
    }

    MPI_Allreduce(&local_max_value, &max_value, 1, data_type, MPI_MAX, m_Comm);

    return (max_value);
}

template<class Type>
Type vector<Type>::min() const
{
    size_t i;
    Type min_value = 0.;
    Type local_min_value = 0.;
    size_t dim = this->size();
    MPI_Datatype data_type = dotk::parallel::mpiDataType(typeid(Type));

    int thread_count = this->threads();
# pragma omp parallel num_threads(thread_count) \
default( none ) \
shared ( dim, local_min_value ) \
private ( i )

# pragma omp for reduction( min : local_min_value )
    for(i = 0; i < dim; ++i)
    {
        if(m_Data[i] < local_min_value)
        {
            local_min_value = m_Data[i];
        }
    }

    MPI_Allreduce(&local_min_value, &min_value, 1, data_type, MPI_MIN, m_Comm);

    return (min_value);
}

template<class Type>
void vector<Type>::abs()
{
    size_t index;
    size_t dim = this->size();
    int thread_count = this->threads();

# pragma omp parallel num_threads(thread_count) \
default(none) \
shared ( dim ) \
private ( index )

# pragma omp for
    for(index = 0; index < dim; ++index)
    {
        m_Data.data()[index] =
                m_Data.data()[index] < static_cast<Type>(0.) ? -(m_Data.data()[index]) : m_Data.data()[index];
    }
}

template<class Type>
Type vector<Type>::sum() const
{
    size_t i;
    Type local_sum = 0.;
    Type global_sum = 0;
    size_t dim = this->size();
    MPI_Datatype data_type = dotk::parallel::mpiDataType(typeid(Type));

    int thread_count = this->threads();
# pragma omp parallel num_threads(thread_count) \
default(none) \
shared ( dim, local_sum ) \
private ( i )

# pragma omp for reduction( + : local_sum )
    for(i = 0; i < dim; ++i)
    {
        local_sum += m_Data[i];
    }

    MPI_Allreduce(&local_sum, &global_sum, 1, data_type, MPI_SUM, m_Comm);

    return (global_sum);
}

template<class Type>
Type vector<Type>::dot(const dotk::vector<Type> & input_) const
{
    size_t i;
    Type local_sum = 0.;
    Type global_sum = 0.;
    size_t dim = this->size();
    assert(dim == input_.size());
    MPI_Datatype data_type = dotk::parallel::mpiDataType(typeid(Type));

    int thread_count = this->threads();
# pragma omp parallel num_threads(thread_count) \
default( none ) \
shared ( dim, local_sum, input_ ) \
private ( i )

# pragma omp for reduction( + : local_sum )
    for(i = 0; i < dim; ++i)
    {
        local_sum += (m_Data[i] * input_[i]);
    }

    MPI_Allreduce(&local_sum, &global_sum, 1, data_type, MPI_SUM, m_Comm);

    return (global_sum);
}

template<class Type>
Type vector<Type>::norm() const
{
    size_t i;
    Type result = 0.;
    Type local_dot = 0.;
    Type global_dot = 0.;
    size_t dim = this->size();
    MPI_Datatype data_type = dotk::parallel::mpiDataType(typeid(Type));

    int thread_count = this->threads();
# pragma omp parallel num_threads(thread_count) \
default( none ) \
shared ( dim, local_dot ) \
private ( i )

# pragma omp for reduction( + : local_dot )
    for(i = 0; i < dim; ++i)
    {
        local_dot += (m_Data[i] * m_Data[i]);
    }

    MPI_Allreduce(&local_dot, &global_dot, 1, data_type, MPI_SUM, m_Comm);

    result = sqrt(global_dot);

    return (result);
}

template<class Type>
void vector<Type>::fill(const Type & value_)
{
    size_t i;
    size_t dim = this->size();

    int thread_count = this->threads();
# pragma omp parallel num_threads(thread_count) \
default( none ) \
shared ( dim, value_ ) \
private ( i )

# pragma omp for
    for(i = 0; i < dim; ++i)
    {
        m_Data[i] = value_;
    }
}

template<class Type>
void vector<Type>::copy(const dotk::vector<Type> & input_)
{
    size_t i;
    size_t dim = this->size();
    assert(dim == input_.size());

    int thread_count = this->threads();
# pragma omp parallel num_threads(thread_count) \
default( none ) \
shared ( dim, input_ ) \
private ( i )

# pragma omp for
    for(i = 0; i < dim; ++i)
    {
        m_Data[i] = input_[i];
    }
}

template<class Type>
void vector<Type>::gather(Type* input_) const
{
    int my_rank;
    MPI_Comm_rank(m_Comm, &my_rank);

    size_t i;
    size_t dim = this->size();
    Type* temp = new Type[dim];
    int thread_count = this->threads();

# pragma omp parallel num_threads(thread_count) \
default( none ) \
shared ( dim, temp ) \
private ( i )

# pragma omp for
    for(i = 0; i < dim; ++i)
    {
        temp[i] = m_Data[i];
    }

    if(my_rank == 0)
    {
        int root = 0;
        MPI_Datatype data_type = dotk::parallel::mpiDataType(typeid(Type));
        MPI_Gatherv(temp, dim, data_type, input_, m_LocalCounts, m_Displacements, data_type, root, m_Comm);
    }
    else
    {
        int root = 0;
        MPI_Datatype data_type = dotk::parallel::mpiDataType(typeid(Type));
        MPI_Gatherv(temp, dim, data_type, NULL, m_LocalCounts, m_Displacements, data_type, root, m_Comm);
    }

    delete[] temp;
    temp = NULL;
}

template<class Type>
size_t vector<Type>::size() const
{
    size_t dim = m_Data.size();
    return (dim);
}

template<class Type>
std::tr1::shared_ptr<dotk::vector<Type> > vector<Type>::clone() const
{
    std::tr1::shared_ptr<dotk::mpix::vector<Type> > vec(new dotk::mpix::vector<Type>(m_Comm, m_GlobalDim, m_NumThreads));
    return (vec);
}

template<class Type>
Type & vector<Type>::operator [](size_t index_)
{
    return (m_Data.operator [](index_));
}

template<class Type>
const Type & vector<Type>::operator [](size_t index_) const
{
    return (m_Data.operator [](index_));
}

template<class Type>
dotk::types::container_t vector<Type>::type() const
{
    return (dotk::types::MPIx_VECTOR);
}

template<class Type>
size_t vector<Type>::rank() const
{
    int rank_id = 0;
    MPI_Comm_rank(m_Comm, &rank_id);
    return (rank_id);
}

template<class Type>
int vector<Type>::threads() const
{
    return (m_NumThreads);
}

template<class Type>
void vector<Type>::allocate(const int & global_dim_)
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
        size_t local_dim = m_LocalCounts[my_rank];
        m_Data.resize(local_dim, 0);
        MPI_Datatype data_type = dotk::parallel::mpiDataType(typeid(Type));
        MPI_Scatterv(temp, m_LocalCounts, m_Displacements, data_type, &m_Data[0], local_dim, data_type, root, m_Comm);

        delete[] temp;
        temp = NULL;
    }
    else
    {
        int root = 0;
        size_t local_dim = m_LocalCounts[my_rank];
        m_Data.resize(local_dim, 0);
        MPI_Datatype data_type = dotk::parallel::mpiDataType(typeid(Type));
        MPI_Scatterv(temp, m_LocalCounts, m_Displacements, data_type, &m_Data[0], local_dim, data_type, root, m_Comm);
    }
}

}

}
