/*
 * DOTk_StdVector.hpp
 *
 *  Created on: Aug 25, 2014
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_SERIALVECTOR_HPP_
#define DOTK_SERIALVECTOR_HPP_

#include <cmath>
#include <vector>
#include <cassert>
#include <algorithm>

#include "vector.hpp"

namespace dotk
{

template<typename Type>
class StdVector : public dotk::vector<Type>
{
public:
    explicit StdVector(std::vector<Type> & data_) :
            m_Data(data_)
    {
    }
    StdVector(int dim_, Type value_ = 0.) :
            m_Data(std::vector<Type>(dim_, value_))
    {
    }
    virtual ~StdVector()
    {
    }
    // Scales a vector by a real constant.
    void scale(const Type & alpha_)
    {
        size_t dim = this->size();
        for(size_t index = 0; index < dim; ++ index)
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
        Type output = *std::max_element(m_Data.begin(), m_Data.end());
        return (output);
    }
    // Returns the minimum element in a range.
    Type min() const
    {
        Type output = *std::min_element(m_Data.begin(), m_Data.end());
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
        for(size_t index = 0; index < dim; ++index)
        {
            output += m_Data[index];
        }
        return (output);
    }
    // Returns the inner product of two vectors.
    Type dot(const dotk::vector<Type> & input_) const
    {
        size_t dim = this->size();
        assert(dim == input_.size());
        Type output = 0.;
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
        size_t dim = m_Data.size();
        return (dim);
    }
    // Clones memory for an object of type dotk::vector
    std::tr1::shared_ptr<dotk::vector<Type> > clone() const
    {
        size_t dim = this->size();
        std::tr1::shared_ptr<dotk::StdVector<Type> > output(new dotk::StdVector<Type>(dim, 0.));
        return (output);
    }
    // Operator overloads the square bracket operator
    Type & operator [](size_t index_)
    {
        return (m_Data.operator [](index_));
    }
    // Operator overloads the const square bracket operator
    const Type & operator [](size_t index_) const
    {
        return (m_Data.operator [](index_));
    }

private:
    std::vector<Type> m_Data;

private:
    StdVector(const dotk::StdVector<Type> &);
    dotk::StdVector<Type> & operator=(const dotk::StdVector<Type> & rhs_);
};

}

#endif /* DOTK_SERIALVECTOR_HPP_ */
