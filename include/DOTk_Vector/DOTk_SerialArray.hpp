/*
 * DOTk_StdArray.hpp
 *
 *  Created on: Apr 3, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_SERIALARRAY_HPP_
#define DOTK_SERIALARRAY_HPP_

#include <cmath>
#include <cassert>
#include <algorithm>

#include "vector.hpp"

namespace dotk
{

template<typename ScalarType>
class StdArray : public dotk::Vector<ScalarType>
{
public:
    StdArray(int dim_, ScalarType value_ = 0.) :
            m_LocalDim(dim_),
            m_Data(new ScalarType[dim_])
    {
        this->fill(value_);
    }
    virtual ~StdArray()
    {
        delete[] m_Data;
        m_Data = nullptr;
    }
    // Scales a vector by a real constant.
    void scale(const ScalarType & alpha_)
    {
        size_t dim = this->size();
        for(size_t index = 0; index < dim; ++index)
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
        size_t dim = this->size();
        ScalarType output = *std::max_element(m_Data, m_Data + dim);
        return (output);
    }
    // Returns the minimum element in a range.
    ScalarType min() const
    {
        size_t dim = this->size();
        ScalarType output = *std::min_element(m_Data, m_Data + dim);
        return (output);
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
        ScalarType output = 0.;
        size_t dim = this->size();
        for(size_t i = 0; i < dim; ++i)
        {
            output += m_Data[i];
        }
        return (output);
    }
    // Returns the inner product of two vectors.
    ScalarType dot(const dotk::Vector<ScalarType> & input_) const
    {
        ScalarType output = 0;
        size_t dim = this->size();
        assert(dim == input_.size());
        for(size_t index = 0; index < dim; ++index)
        {
            output += m_Data[index] * input_[index];
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
        size_t dim = this->size();
        for(size_t index = 0; index < dim; ++index)
        {
            m_Data[index] = value_;
        }
    }
    // Returns the number of elements in the vector.
    size_t size() const
    {
        return (m_LocalDim);
    }
    // Clones memory for an object of ScalarType dotk::Vector<ScalarType>
    std::tr1::shared_ptr<dotk::Vector<ScalarType> > clone() const
    {
        std::tr1::shared_ptr < dotk::StdArray<ScalarType> > output(new dotk::StdArray<ScalarType>(m_LocalDim));
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
    int m_LocalDim;
    ScalarType* m_Data;

private:
    StdArray(const dotk::StdArray<ScalarType> &);
    dotk::StdArray<ScalarType> & operator=(const dotk::StdArray<ScalarType> & rhs_);
};

}

#endif /* DOTK_SERIALARRAY_HPP_ */
