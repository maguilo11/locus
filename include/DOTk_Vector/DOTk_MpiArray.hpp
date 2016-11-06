/*
 * DOTk_MpiArray.hpp
 *
 *  Created on: Apr 4, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_MPIARRAY_HPP_
#define DOTK_MPIARRAY_HPP_

#include <mpi.h>
#include <cmath>
#include <cassert>
#include <typeinfo>

#include "vector.hpp"
#include "DOTk_ParallelUtils.hpp"

namespace dotk
{

template<typename Type>
class MpiArray : public dotk::vector<Type>
{
public:
    MpiArray(int global_dim_, Type value_ = 0.) :
            m_LocalDim(0),
            m_GlobalDim(global_dim_),
            m_Comm(MPI_COMM_WORLD),
            m_Data(nullptr),
            m_LocalCounts(nullptr),
            m_Displacements(nullptr)
    {
        this->allocate(global_dim_);
        this->fill(value_);
    }
    MpiArray(MPI_Comm comm_, int global_dim_, Type value_ = 0.) :
            m_LocalDim(0),
            m_GlobalDim(global_dim_),
            m_Comm(comm_),
            m_Data(nullptr),
            m_LocalCounts(nullptr),
            m_Displacements(nullptr)
    {
        this->allocate(global_dim_);
        this->fill(value_);
    }
    virtual ~MpiArray()
    {
        delete[] m_LocalCounts;
        m_LocalCounts = nullptr;
        delete[] m_Displacements;
        m_Displacements = nullptr;
        delete[] m_Data;
        m_Data = nullptr;
    }
    // Scales a vector by a real constant.
    void scale(const Type & alpha_)
    {
        size_t dim = this->size();
        for(size_t index = 0; index < dim; ++index)
        {
            m_Data[index] = alpha_ * m_Data[index];
        }
    }
    // Component wise multiplication of two vectors.
    void cwiseProd(const dotk::vector<Type> & input_)
    {
        size_t dim = this->size();
        assert(dim == input_.size());
        for(size_t index = 0; index < dim; ++index)
        {
            m_Data[index] = m_Data[index] * input_[index];
        }
    }
    // Constant times a vector plus a vector.
    void axpy(const Type & alpha_, const dotk::vector<Type> & input_)
    {
        size_t dim = this->size();
        assert(dim == input_.size());
        for(size_t index = 0; index < dim; ++index)
        {
            m_Data[index] = alpha_ * input_[index] + m_Data[index];
        }
    }
    // Returns the maximum element in a range.
    Type max() const
    {
        Type global_max = 0.;
        size_t dim = this->size();
        Type local_max = *std::max_element(m_Data, m_Data + dim);
        MPI_Datatype data_type = dotk::parallel::mpiDataType(typeid(Type));
        MPI_Allreduce(&local_max, &global_max, 1, data_type, MPI_MAX, m_Comm);
        return (global_max);
    }
    // Returns the minimum element in a range.
    Type min() const
    {
        Type global_min = 0.;
        size_t dim = this->size();
        Type local_min = *std::min_element(m_Data, m_Data + dim);
        MPI_Datatype data_type = dotk::parallel::mpiDataType(typeid(Type));
        MPI_Allreduce(&local_min, &global_min, 1, data_type, MPI_MIN, m_Comm);
        return (global_min);
    }
    // Computes the absolute value of each element in the container.
    void abs()
    {
        size_t dim = this->size();
        for(size_t index = 0; index < dim; ++index)
        {
            m_Data[index] = m_Data[index] < static_cast<Type>(0.) ? -(m_Data[index]) : m_Data[index];
        }
    }
    // Returns the sum of all the elements in the container.
    Type sum() const
    {
        Type local_sum = 0.;
        size_t dim = this->size();
        MPI_Datatype data_type = dotk::parallel::mpiDataType(typeid(Type));
        for(size_t index = 0; index < dim; ++index)
        {
            local_sum += m_Data[index];
        }
        Type global_sum = 0.;
        MPI_Allreduce(&local_sum, &global_sum, 1, data_type, MPI_SUM, m_Comm);
        return (global_sum);
    }
    // Returns the inner product of two vectors.
    Type dot(const dotk::vector<Type> & input_) const
    {
        Type global_inner_product = 0.;
        Type local_inner_product = 0.;
        size_t dim = this->size();
        assert(dim == input_.size());
        MPI_Datatype data_type = dotk::parallel::mpiDataType(typeid(Type));
        for(size_t index = 0; index < dim; ++index)
        {
            local_inner_product += m_Data[index] * input_[index];
        }
        MPI_Allreduce(&local_inner_product, &global_inner_product, 1, data_type, MPI_SUM, m_Comm);
        return (global_inner_product);
    }
    // Returns the euclidean norm of a vector.
    Type norm() const
    {
        Type output = this->dot(*this);
        output = std::sqrt(output);
        return (output);
    }
    // Assigns new contents to the vector, replacing its current contents, and not modifying its size.
    void fill(const Type & value_)
    {
        size_t dim = this->size();
        for(size_t index = 0; index < dim; ++index)
        {
            m_Data[index] = value_;
        }
    }
    // Copies the elements in the range [first,last) into the range beginning at result.
    void copy(const dotk::vector<Type> & input_)
    {
        size_t dim = this->size();
        assert(dim == input_.size());
        for(size_t index = 0; index < dim; ++index)
        {
            m_Data[index] = input_[index];
        }
    }
    // Gathers data from private member data of a group to one member.
    void gather(Type* input_) const
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
            MPI_Gatherv(temp, local_dim, data_type, nullptr, m_LocalCounts, m_Displacements, data_type, root, m_Comm);
        }

        delete[] temp;
        temp = nullptr;
    }
    // Returns the number of elements in the vector.
    size_t size() const
    {
        return (m_LocalDim);
    }
    // Clones memory for an object of type dotk::vector<T>
    std::tr1::shared_ptr<dotk::vector<Type> > clone() const
    {
        std::tr1::shared_ptr< dotk::MpiArray<Type> > output(new dotk::MpiArray<Type>(m_Comm, m_GlobalDim));
        return (output);
    }
    // Operator overloads the square bracket operator
    Type & operator [](size_t index_)
    {
        return (m_Data[index_]);
    }
    // Operator overloads the const square bracket operator
    const Type & operator [](size_t index_) const
    {
        return (m_Data[index_]);
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

        Type* temp = nullptr;
        if(my_rank == 0)
        {
            int root = 0;
            temp = new Type[global_dim_];
            m_LocalDim = m_LocalCounts[my_rank];
            m_Data = new Type[m_LocalDim];
            MPI_Datatype data_type = dotk::parallel::mpiDataType(typeid(Type));
            MPI_Scatterv(temp, m_LocalCounts, m_Displacements, data_type, m_Data, m_LocalDim, data_type, root, m_Comm);

            delete[] temp;
            temp = nullptr;
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

private:
    int m_LocalDim;
    int m_GlobalDim;
    MPI_Comm m_Comm;
    Type* m_Data;
    int* m_LocalCounts;
    int* m_Displacements;

private:
    MpiArray(const dotk::MpiArray<Type> &);
    dotk::MpiArray<Type> & operator=(const dotk::MpiArray<Type> & rhs_);
};

}

#endif /* DOTK_MPIARRAY_HPP_ */
