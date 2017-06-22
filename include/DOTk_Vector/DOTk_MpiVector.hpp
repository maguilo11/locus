/*
 * DOTk_MpiVector.hpp
 *
 *  Created on: Apr 4, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_MPIVECTOR_HPP_
#define DOTK_MPIVECTOR_HPP_

#include <mpi.h>
#include <vector>
#include <typeinfo>
#include <algorithm>

#include "vector.hpp"
#include "DOTk_ParallelUtils.hpp"

namespace dotk
{

template<typename ScalarType>
class MpiVector: public dotk::Vector<ScalarType>
{
public:
    MpiVector(int global_dim_, ScalarType value_ = 0.) :
            m_GlobalDim(global_dim_),
            m_Comm(MPI_COMM_WORLD),
            m_LocalCounts(nullptr),
            m_Displacements(nullptr),
            m_Data()
    {
        this->allocate(global_dim_);
        this->fill(value_);
    }
    MpiVector(MPI_Comm comm_, int global_dim_, ScalarType value_ = 0.) :
            m_GlobalDim(global_dim_),
            m_Comm(comm_),
            m_LocalCounts(nullptr),
            m_Displacements(nullptr),
            m_Data()
    {
        this->allocate(global_dim_);
        this->fill(value_);
    }
    virtual ~MpiVector()
    {
        delete[] m_LocalCounts;
        m_LocalCounts = nullptr;
        delete[] m_Displacements;
        m_Displacements = nullptr;
    }
    // Scales a vector by a real constant.
    void scale(const ScalarType & alpha_)
    {
        size_t dim = m_Data.size();
        for(size_t index = 0; index < dim; ++ index)
        {
            m_Data[index] = alpha_ * m_Data[index];
        }
    }
    // Component wise multiplication of two vectors.
    void elementWiseMultiplication(const dotk::Vector<ScalarType> & input_)
    {
        size_t dim = this->size();
        assert(dim == input_.size());
        for(size_t index = 0; index < dim; ++index)
        {
            m_Data[index] = m_Data[index] * input_[index];
        }
    }
    //! Update vector values with scaled values of A, this = beta*this + alpha*A.
    void update(const ScalarType & alpha_, const dotk::Vector<ScalarType> & input_, const ScalarType & beta_)
    {
        assert(this->size() == input_.size());
        if(beta_ == 0.)
        {
            size_t dim = this->size();
            for(size_t index = 0; index < dim; ++index)
            {
                m_Data[index] = alpha_ * input_[index];
            }
        }
        else
        {
            size_t dim = this->size();
            for(size_t index = 0; index < dim; ++index)
            {
                m_Data[index] = alpha_ * input_[index] + beta_ * m_Data[index];
            }
        }
    }
    // Returns the maximum element in a range.
    ScalarType max() const
    {
        ScalarType global_max = 0.;
        ScalarType local_max = *std::max_element(m_Data.begin(), m_Data.end());
        MPI_Datatype data_type = dotk::parallel::mpiDataType(typeid(ScalarType));
        MPI_Allreduce(&local_max, &global_max, 1, data_type, MPI_MAX, m_Comm);
        return (global_max);
    }
    // Returns the minimum element in a range.
    ScalarType min() const
    {
        ScalarType local_min = *std::min_element(m_Data.begin(), m_Data.end());
        MPI_Datatype data_type = dotk::parallel::mpiDataType(typeid(ScalarType));
        ScalarType global_min = 0.;
        MPI_Allreduce(&local_min, &global_min, 1, data_type, MPI_MIN, m_Comm);
        return (global_min);
    }
    // Computes the absolute value of each element in the container.
    void abs()
    {
        size_t dim = this->size();
        for(size_t index = 0; index < dim; ++index)
        {
            m_Data[index] = m_Data[index] < static_cast<ScalarType>(0.) ? -(m_Data[index]) : m_Data[index];
        }
    }
    // Returns the sum of all the elements in the container.
    ScalarType sum() const
    {
        ScalarType local_sum = 0.;
        size_t dim = m_Data.size();
        MPI_Datatype data_type = dotk::parallel::mpiDataType(typeid(ScalarType));
        for(size_t index = 0; index < dim; ++index)
        {
            local_sum += m_Data[index];
        }
        ScalarType global_sum = 0.;
        MPI_Allreduce(&local_sum, &global_sum, 1, data_type, MPI_SUM, m_Comm);
        return (global_sum);
    }
    // Returns the inner product of two vectors.
    ScalarType dot(const dotk::Vector<ScalarType> & input_) const
    {
        ScalarType local_inner_product = 0.;
        size_t dim = m_Data.size();
        assert(dim == input_.size());
        MPI_Datatype data_type = dotk::parallel::mpiDataType(typeid(ScalarType));
        for(size_t index = 0; index < dim; ++index)
        {
            local_inner_product += m_Data[index] * input_[index];
        }
        ScalarType global_inner_product = 0.;
        MPI_Allreduce(&local_inner_product, &global_inner_product, 1, data_type, MPI_SUM, m_Comm);
        return (global_inner_product);
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
        size_t dim = m_Data.size();
        for(size_t index = 0; index < dim; ++index)
        {
            m_Data[index] = value_;
        }
    }
    // Returns the number of elements in the vector.
    size_t size() const
    {
        size_t dim = m_Data.size();
        return (dim);
    }
    // Clones memory for an object of ScalarType dotk::Vector
    std::shared_ptr<dotk::Vector<ScalarType> > clone() const
    {
        std::shared_ptr < dotk::MpiVector<ScalarType> > output(new dotk::MpiVector<ScalarType>(m_Comm, m_GlobalDim));
        return (output);
    }
    // Operator overloads the square bracket operator
    ScalarType & operator [](size_t index_)
    {
        return (m_Data.operator [](index_));
    }
    // Operator overloads the const square bracket operator
    const ScalarType & operator [](size_t index_) const
    {
        return (m_Data.operator [](index_));
    }

private:
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
            size_t local_dim = m_LocalCounts[my_rank];
            m_Data.resize(local_dim, 0);
            MPI_Datatype data_type = dotk::parallel::mpiDataType(typeid(ScalarType));
            MPI_Scatterv(temp, m_LocalCounts, m_Displacements, data_type, &m_Data[0], local_dim, data_type, root, m_Comm);

            delete[] temp;
            temp = nullptr;
        }
        else
        {
            int root = 0;
            size_t local_dim = m_LocalCounts[my_rank];
            m_Data.resize(local_dim, 0);
            MPI_Datatype data_type = dotk::parallel::mpiDataType(typeid(ScalarType));
            MPI_Scatterv(temp, m_LocalCounts, m_Displacements, data_type, &m_Data[0], local_dim, data_type, root, m_Comm);
        }
    }

private:
    int m_GlobalDim;

    MPI_Comm m_Comm;
    int* m_LocalCounts;
    int* m_Displacements;
    std::vector<ScalarType> m_Data;

private:
    MpiVector(const dotk::MpiVector<ScalarType> &);
    dotk::MpiVector<ScalarType> & operator=(const dotk::MpiVector<ScalarType> & rhs_);
};

}

#endif /* DOTK_MPIVECTOR_HPP_ */
