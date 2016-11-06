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

template<typename Type>
class StdArray : public dotk::vector<Type>
{
public:
    StdArray(int dim_, Type value_ = 0.) :
            m_LocalDim(dim_),
            m_Data(new Type[dim_])
    {
        this->fill(value_);
    }
    virtual ~StdArray()
    {
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
        size_t dim = this->size();
        Type output = *std::max_element(m_Data, m_Data + dim);
        return (output);
    }
    // Returns the minimum element in a range.
    Type min() const
    {
        size_t dim = this->size();
        Type output = *std::min_element(m_Data, m_Data + dim);
        return (output);
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
        Type output = 0.;
        size_t dim = this->size();
        for(size_t i = 0; i < dim; ++i)
        {
            output += m_Data[i];
        }
        return (output);
    }
    // Returns the inner product of two vectors.
    Type dot(const dotk::vector<Type> & input_) const
    {
        Type output = 0;
        size_t dim = this->size();
        assert(dim == input_.size());
        for(size_t index = 0; index < dim; ++index)
        {
            output += m_Data[index] * input_[index];
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
        size_t dim = this->size();
        for(size_t index = 0; index < dim; ++index)
        {
            input_[index] = m_Data[index];
        }
    }
    // Returns the number of elements in the vector.
    size_t size() const
    {
        return (m_LocalDim);
    }
    // Clones memory for an object of type dotk::vector<Type>
    std::tr1::shared_ptr<dotk::vector<Type> > clone() const
    {
        std::tr1::shared_ptr < dotk::StdArray<Type> > output(new dotk::StdArray<Type>(m_LocalDim));
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
    int m_LocalDim;
    Type* m_Data;

private:
    StdArray(const dotk::StdArray<Type> &);
    dotk::StdArray<Type> & operator=(const dotk::StdArray<Type> & rhs_);
};

}

#endif /* DOTK_SERIALARRAY_HPP_ */
