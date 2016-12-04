/*
 * DOTk_OmpArray.hpp
 *
 *  Created on: Apr 4, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_OMPARRAY_HPP_
#define DOTK_OMPARRAY_HPP_

#include <omp.h>
#include <cmath>
#include <cassert>

#include "vector.hpp"

namespace dotk
{

template<typename Type>
class OmpArray : public dotk::vector<Type>
{
public:
    OmpArray(int dim_, int num_threads_ = 1, Type value_ = 0.) :
            m_NumDim(dim_),
            m_NumThreads(num_threads_),
            m_Data(new Type[dim_])
    {
        this->fill(value_);
    }
    virtual ~OmpArray()
    {
        delete[] m_Data;
        m_Data = nullptr;
    }
    // Scales a vector by a real constant.
    void scale(const Type & alpha_)
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
    void cwiseProd(const dotk::vector<Type> & input_)
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
    // Constant times a vector plus a vector.
    void axpy(const Type & alpha_, const dotk::vector<Type> & input_)
    {
        size_t index;
        size_t dim = this->size();
        assert(dim == input_.size());

        int thread_count = this->threads();
# pragma omp parallel num_threads(thread_count) \
    default( none ) \
    shared ( dim, input_, alpha_ ) \
    private ( index )

# pragma omp for
        for(index = 0; index < dim; ++index)
        {
            m_Data[index] = alpha_ * input_[index] + m_Data[index];
        }
    }
    // Returns the maximum element in a range.
    Type max() const
    {
        size_t index;
        Type max_value = 0;
        size_t dim = this->size();

        int thread_count = this->threads();
# pragma omp parallel num_threads(thread_count) \
    default( none ) \
    shared ( dim, max_value ) \
    private ( index )

# pragma omp for reduction( max : max_value )
        for(index = 0; index < dim; ++index)
        {
            if(m_Data[index] > max_value)
            {
                max_value = m_Data[index];
            }
        }
        return (max_value);
    }
    // Returns the minimum element in a range.
    Type min() const
    {
        size_t index;
        Type min_value = 0.;
        size_t dim = this->size();

        int thread_count = this->threads();
# pragma omp parallel num_threads(thread_count) \
    default( none ) \
    shared ( dim, min_value ) \
    private ( index )

# pragma omp for reduction( min : min_value )
        for(index = 0; index < dim; ++index)
        {
            if(m_Data[index] < min_value)
            {
                min_value = m_Data[index];
            }
        }

        return (min_value);
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
            m_Data[index] = m_Data[index] < static_cast<Type>(0.) ? -(m_Data[index]): m_Data[index];
        }
    }
    // Returns the sum of all the elements in the container.
    Type sum() const
    {
        size_t index;
        Type output = 0;
        size_t dim = this->size();

        int thread_count = this->threads();
# pragma omp parallel num_threads(thread_count) \
    default(none) \
    shared ( dim, output ) \
    private ( index )

# pragma omp for reduction(+:output)
        for(index = 0; index < dim; ++ index)
        {
            output += m_Data[index];
        }
        return (output);
    }
    // Returns the inner product of two vectors.
    Type dot(const dotk::vector<Type> & input_) const
    {
        size_t index;
        size_t dim = this->size();
        assert(dim == input_.size());
        Type output = 0.;

        int thread_count = this->threads();
# pragma omp parallel num_threads(thread_count) \
    default( none ) \
    shared ( dim, output, input_ ) \
    private ( index )

# pragma omp for reduction(+:output)
        for(index = 0; index < dim; ++ index)
        {
            output += (m_Data[index] * input_[index]);
        }

        return (output);
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
    // Copies the elements in the range [first,last) into the range beginning at result.
    void copy(const dotk::vector<Type> & input_)
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
            m_Data[index] = input_[index];
        }
    }
    // Gathers data from private member data of a group to one member.
    void gather(Type* input_) const
    {
        size_t index;
        size_t dim = this->size();

        int thread_count = this->threads();
# pragma omp parallel num_threads(thread_count) \
    default( none ) \
    shared ( dim, input_ ) \
    private ( index )

# pragma omp for
        for(index = 0; index < dim; ++ index)
        {
            input_[index] = m_Data[index];
        }
    }
    // Returns the number of elements in the vector.
    size_t size() const
    {
        return (m_NumDim);
    }
    // Clones memory for an object of type dotk::vector
    std::tr1::shared_ptr<dotk::vector<Type> > clone() const
    {
        size_t dim = this->size();
        int thread_count = this->threads();
        std::tr1::shared_ptr < dotk::OmpArray<Type> > output(new dotk::OmpArray<Type>(dim, thread_count, 0.));
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
    int threads() const
    {
        return (m_NumThreads);
    }

private:
    int m_NumDim;
    int m_NumThreads;
    Type* m_Data;

private:
    OmpArray(const dotk::OmpArray<Type> &);
    dotk::OmpArray<Type> & operator=(const dotk::OmpArray<Type> & rhs_);
};

}

#endif /* DOTK_OMPARRAY_HPP_ */
