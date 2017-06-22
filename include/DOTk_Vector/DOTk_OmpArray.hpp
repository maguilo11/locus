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

template<typename ScalarType>
class OmpArray : public dotk::Vector<ScalarType>
{
public:
    OmpArray(int dim_, int num_threads_ = 1, ScalarType value_ = 0.) :
            m_NumDim(dim_),
            m_NumThreads(num_threads_),
            m_Data(new ScalarType[dim_])
    {
        this->fill(value_);
    }
    virtual ~OmpArray()
    {
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
        ScalarType max_value = 0;
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
    ScalarType min() const
    {
        size_t index;
        ScalarType min_value = 0.;
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
            m_Data[index] = m_Data[index] < static_cast<ScalarType>(0.) ? -(m_Data[index]): m_Data[index];
        }
    }
    // Returns the sum of all the elements in the container.
    ScalarType sum() const
    {
        size_t index;
        ScalarType output = 0;
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
    ScalarType dot(const dotk::Vector<ScalarType> & input_) const
    {
        size_t index;
        size_t dim = this->size();
        assert(dim == input_.size());
        ScalarType output = 0.;

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
        return (m_NumDim);
    }
    // Clones memory for an object of ScalarType dotk::Vector
    std::shared_ptr<dotk::Vector<ScalarType> > clone() const
    {
        size_t dim = this->size();
        int thread_count = this->threads();
        std::shared_ptr < dotk::OmpArray<ScalarType> > output(new dotk::OmpArray<ScalarType>(dim, thread_count, 0.));
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
    int threads() const
    {
        return (m_NumThreads);
    }

private:
    int m_NumDim;
    int m_NumThreads;
    ScalarType* m_Data;

private:
    OmpArray(const dotk::OmpArray<ScalarType> &);
    dotk::OmpArray<ScalarType> & operator=(const dotk::OmpArray<ScalarType> & rhs_);
};

}

#endif /* DOTK_OMPARRAY_HPP_ */
