/*
 * DOTk_MpiX_Array.hpp
 *
 *  Created on: Apr 4, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_MPIX_ARRAY_HPP_
#define DOTK_MPIX_ARRAY_HPP_

#include <mpi.h>
#include <omp.h>
#include <cmath>
#include <cassert>
#include <typeinfo>

#include "vector.hpp"
#include "DOTk_ParallelUtils.hpp"

namespace dotk
{

template<typename ScalarType>
class MpiX_Array : public dotk::Vector<ScalarType>
{
public:
    MpiX_Array(int global_dim_, int num_threads_, ScalarType value_ = 0.) :
            m_LocalDim(0),
            m_GlobalDim(global_dim_),
            m_NumThreads(num_threads_),
            m_Comm(MPI_COMM_WORLD),
            m_Data(nullptr),
            m_LocalCounts(nullptr),
            m_Displacements(nullptr)
    {
        this->allocate(global_dim_);
        this->fill(value_);
    }
    MpiX_Array(MPI_Comm comm_, int global_dim_, int num_threads_, ScalarType value_ = 0.) :
            m_LocalDim(0),
            m_GlobalDim(global_dim_),
            m_NumThreads(num_threads_),
            m_Comm(comm_),
            m_Data(nullptr),
            m_LocalCounts(nullptr),
            m_Displacements(nullptr)
    {
        this->allocate(global_dim_);
        this->fill(value_);
    }
    virtual ~MpiX_Array()
    {
        delete[] m_LocalCounts;
        m_LocalCounts = nullptr;
        delete[] m_Displacements;
        m_Displacements = nullptr;
        delete[] m_Data;
        m_Data = nullptr;
    }
    // Scales a vector by a real constant.
    void scale(const ScalarType & alpha_)
    {
        size_t index;
        size_t dim = this->size();
        int thread_count = this->threads();
# pragma omp parallel num_threads(thread_count) \
    default( none ) \
    shared ( dim, alpha_ ) \
    private ( index )

# pragma omp for
        for(index = 0; index < dim; ++index)
        {
            m_Data[index] = alpha_ * m_Data[index];
        }
    }
    // Component wise multiplication of two vectors.
    void elementWiseMultiplication(const dotk::Vector<ScalarType> & input_)
    {
        size_t index;
        size_t dim = this->size();
        assert(dim == input_.size());

        int thread_count = this->threads();
# pragma omp parallel num_threads(thread_count) \
    default( none ) \
    shared ( dim, input_ ) \
    private ( index )

# pragma omp for
        for(index = 0; index < dim; ++index)
        {
            m_Data[index] = m_Data[index] * input_[index];
        }
    }
    //! Update vector values with scaled values of A, this = beta*this + alpha*A.
    void update(const ScalarType & alpha_, const dotk::Vector<ScalarType> & input_, const ScalarType & beta_)
    {
        assert(this->size() == input_.size());
        size_t index;
        size_t dim = this->size();
        int thread_count = this->threads();

        if(beta_ == 0.)
        {
# pragma omp parallel num_threads(thread_count) \
    default( none ) \
    shared ( dim, input_, alpha_ ) \
    private ( index )

# pragma omp for
            for(index = 0; index < dim; ++ index)
            {
                m_Data[index] = alpha_ * input_[index];
            }
        }
        else
        {
# pragma omp parallel num_threads(thread_count) \
    default( none ) \
    shared ( dim, input_, alpha_, beta_ ) \
    private ( index )

# pragma omp for
            for(index = 0; index < dim; ++ index)
            {
                m_Data[index] = alpha_ * input_[index] + beta_ * m_Data[index];
            }
        }
    }
    // Returns the maximum element in a range.
    ScalarType max() const
    {
        size_t index;
        ScalarType global_max_value = 0;
        ScalarType local_max_value = 0;
        size_t dim = this->size();
        MPI_Datatype data_type = dotk::parallel::mpiDataType(typeid(ScalarType));

        int thread_count = this->threads();
# pragma omp parallel num_threads(thread_count) \
    default( none ) \
    shared ( dim, local_max_value ) \
    private ( index )

# pragma omp for reduction( max : local_max_value )
        for(index = 0; index < dim; ++index)
        {
            if(m_Data[index] > local_max_value)
            {
                local_max_value = m_Data[index];
            }
        }
        MPI_Allreduce(&local_max_value, &global_max_value, 1, data_type, MPI_MAX, m_Comm);
        return (global_max_value);
    }
    // Returns the minimum element in a range.
    ScalarType min() const
    {
        size_t index;
        ScalarType global_min_value = 0.;
        ScalarType local_min_value = 0.;
        size_t dim = this->size();
        MPI_Datatype data_type = dotk::parallel::mpiDataType(typeid(ScalarType));

        int thread_count = this->threads();
# pragma omp parallel num_threads(thread_count) \
    default( none ) \
    shared ( dim, local_min_value ) \
    private ( index )

# pragma omp for reduction( min : local_min_value )
        for(index = 0; index < dim; ++index)
        {
            if(m_Data[index] < local_min_value)
            {
                local_min_value = m_Data[index];
            }
        }
        MPI_Allreduce(&local_min_value, &global_min_value, 1, data_type, MPI_MIN, m_Comm);
        return (global_min_value);
    }
    // Computes the absolute value of each element in the container.
    void abs()
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
            m_Data[index] = m_Data[index] < static_cast<ScalarType>(0.) ? -(m_Data[index]): m_Data[index];
        }
    }
    // Returns the sum of all the elements in the container.
    ScalarType sum() const
    {
        size_t index;
        ScalarType local_sum = 0.;
        ScalarType global_sum = 0;
        size_t dim = this->size();
        MPI_Datatype data_type = dotk::parallel::mpiDataType(typeid(ScalarType));

        int thread_count = this->threads();
# pragma omp parallel num_threads(thread_count) \
    default(none) \
    shared ( dim, local_sum ) \
    private ( index )

# pragma omp for reduction( + : local_sum )
        for(index = 0; index < dim; ++index)
        {
            local_sum += m_Data[index];
        }
        MPI_Allreduce(&local_sum, &global_sum, 1, data_type, MPI_SUM, m_Comm);
        return (global_sum);
    }
    // Returns the inner product of two vectors.
    ScalarType dot(const dotk::Vector<ScalarType> & input_) const
    {
        size_t index;
        ScalarType local_sum = 0.;
        ScalarType global_sum = 0.;
        size_t dim = this->size();
        assert(dim == input_.size());
        MPI_Datatype data_type = dotk::parallel::mpiDataType(typeid(ScalarType));

        int thread_count = this->threads();
# pragma omp parallel num_threads(thread_count) \
    default( none ) \
    shared ( dim, local_sum, input_ ) \
    private ( index )

# pragma omp for reduction( + : local_sum )
        for(index = 0; index < dim; ++index)
        {
            local_sum += (m_Data[index] * input_[index]);
        }
        MPI_Allreduce(&local_sum, &global_sum, 1, data_type, MPI_SUM, m_Comm);
        return (global_sum);
    }
    // Returns the euclidean norm of a vector.
    ScalarType norm() const
    {
        ScalarType output = this->dot(*this);
        output = std::sqrt(output);
        return (output);
    }
    // Assigns new contents to the vector, replacing its current contents, and not modifying its size.
    void fill(const ScalarType & value_)
    {
        size_t index;
        size_t dim = this->size();

        int thread_count = this->threads();
# pragma omp parallel num_threads(thread_count) \
    default( none ) \
    shared ( dim, value_ ) \
    private ( index )

# pragma omp for
        for(index = 0; index < dim; ++index)
        {
            m_Data[index] = value_;
        }
    }
    // Returns the number of elements in the vector.
    size_t size() const
    {
        return (m_LocalDim);
    }
    // Clones memory for an object of ScalarType dotk::Vector
    std::shared_ptr<dotk::Vector<ScalarType> > clone() const
    {
        std::shared_ptr< dotk::MpiX_Array<ScalarType> >
            output(new dotk::MpiX_Array<ScalarType>(m_Comm, m_GlobalDim, m_NumThreads));
        return (output);
    }
    // Operator overloads the square bracket operator
    ScalarType & operator [](size_t index_)
    {
        return (m_Data[index_]);
    }
    // Operator overloads the const square bracket operator
    const ScalarType & operator [](size_t index_) const
    {
        return (m_Data[index_]);
    }

private:
    int threads() const
    {
        return (m_NumThreads);
    }
    void allocate(const int & global_dim_)
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

        ScalarType* temp = nullptr;
        if(my_rank == 0)
        {
            int root = 0;
            temp = new ScalarType[global_dim_];
            m_LocalDim = m_LocalCounts[my_rank];
            m_Data = new ScalarType[m_LocalDim];
            MPI_Datatype data_type = dotk::parallel::mpiDataType(typeid(ScalarType));
            MPI_Scatterv(temp, m_LocalCounts, m_Displacements, data_type, &m_Data[0], m_LocalDim, data_type, root, m_Comm);

            delete[] temp;
            temp = nullptr;
        }
        else
        {
            int root = 0;
            m_LocalDim = m_LocalCounts[my_rank];
            m_Data = new ScalarType[m_LocalDim];
            MPI_Datatype data_type = dotk::parallel::mpiDataType(typeid(ScalarType));
            MPI_Scatterv(temp, m_LocalCounts, m_Displacements, data_type, &m_Data[0], m_LocalDim, data_type, root, m_Comm);
        }
    }

private:
    int m_LocalDim;
    int m_GlobalDim;
    int m_NumThreads;

    MPI_Comm m_Comm;
    ScalarType* m_Data;
    int* m_LocalCounts;
    int* m_Displacements;

private:
    MpiX_Array(const dotk::MpiX_Array<ScalarType> & rhs_);
    dotk::MpiX_Array<ScalarType> & operator=(const dotk::MpiX_Array<ScalarType> & rhs_);
};

}

#endif /* DOTK_MPIX_ARRAY_HPP_ */
