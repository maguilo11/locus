/*
 * DOTk_MpiArray.hpp
 *
 *  Created on: Apr 4, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_MPIARRAY_HPP_
#define DOTK_MPIARRAY_HPP_

#include <mpi.h>
#include "vector.hpp"

namespace dotk
{

namespace mpi
{

template<class Type>
class array: public dotk::vector<Type>
{
public:
    array(int global_dim_, Type value_ = 0.);
    array(MPI_Comm comm_, int global_dim_, Type value_ = 0.);
    virtual ~array();

    // Scales a vector by a real constant.
    virtual void scale(const Type & alpha_);
    // Component wise multiplication of two vectors.
    virtual void cwiseProd(const dotk::vector<Type> & input_);
    // Constant times a vector plus a vector.
    virtual void axpy(const Type & alpha_, const dotk::vector<Type> & input_);
    // Returns the maximum element in a range.
    virtual Type max() const;
    // Returns the minimum element in a range.
    virtual Type min() const;
    // Computes the absolute value of each element in the container.
    virtual void abs();
    // Returns the sum of all the elements in the container.
    virtual Type sum() const;
    // Returns the inner product of two vectors.
    virtual Type dot(const dotk::vector<Type> & input_) const;
    // Returns the euclidean norm of a vector.
    virtual Type norm() const;
    // Assigns new contents to the vector, replacing its current contents, and not modifying its size.
    virtual void fill(const Type & value_);
    // Copies the elements in the range [first,last) into the range beginning at result.
    virtual void copy(const dotk::vector<Type> & input_);
    // Gathers data from private member data of a group to one member.
    virtual void gather(Type* input_) const;
    // Returns the number of elements in the vector.
    virtual size_t size() const;
    // Clones memory for an object of type dotk::vector<T>
    virtual std::tr1::shared_ptr<dotk::vector<Type> > clone() const;
    // Operator overloads the square bracket operator
    virtual Type & operator [](size_t index_);
    // Operator overloads the const square bracket operator
    virtual const Type & operator [](size_t index_) const;
    // Returns the dotk vector type
    virtual dotk::types::container_t type() const;
    // Returns the rank of the calling process in the communicator
    virtual size_t rank() const;

private:
    void allocate(const int & global_dim_);

private:
    int m_LocalDim;
    int m_GlobalDim;
    MPI_Comm m_Comm;
    Type* m_Data;
    int* m_LocalCounts;
    int* m_Displacements;

private:
    array(const dotk::mpi::array<Type> &);
    dotk::mpi::array<Type> & operator=(const dotk::mpi::array<Type> & rhs_);
};

}

}

#endif /* DOTK_MPIARRAY_HPP_ */
