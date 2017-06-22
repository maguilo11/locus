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

template<typename ScalarType>
class StdVector : public dotk::Vector<ScalarType>
{
public:
    explicit StdVector(std::vector<ScalarType> & data_) :
            m_Data(data_)
    {
    }
    StdVector(int dim_, ScalarType value_ = 0.) :
            m_Data(std::vector<ScalarType>(dim_, value_))
    {
    }
    virtual ~StdVector()
    {
    }
    // Scales a Vector by a real constant.
    void scale(const ScalarType & alpha_)
    {
        size_t dim = this->size();
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
        ScalarType output = *std::max_element(m_Data.begin(), m_Data.end());
        return (output);
    }
    // Returns the minimum element in a range.
    ScalarType min() const
    {
        ScalarType output = *std::min_element(m_Data.begin(), m_Data.end());
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
        for(size_t index = 0; index < dim; ++index)
        {
            output += m_Data[index];
        }
        return (output);
    }
    // Returns the inner product of two vectors.
    ScalarType dot(const dotk::Vector<ScalarType> & input_) const
    {
        size_t dim = this->size();
        assert(dim == input_.size());
        ScalarType output = 0.;
        for(size_t index = 0; index < dim; ++index)
        {
            output += m_Data[index] * input_[index];
        }
        return (output);
    }
    // Returns the euclidean norm of a Vector.
    ScalarType norm() const
    {
        ScalarType output = this->dot(*this);
        output = std::sqrt(output);
        return (output);
    }
    // Assigns new contents to the Vector, replacing its current contents, and not modifying its size.
    void fill(const ScalarType & value_)
    {
        size_t dim = this->size();
        for(size_t index = 0; index < dim; ++index)
        {
            m_Data[index] = value_;
        }
    }
    // Returns the number of elements in the Vector.
    size_t size() const
    {
        size_t dim = m_Data.size();
        return (dim);
    }
    // Clones memory for an object of ScalarType dotk::Vector
    std::shared_ptr<dotk::Vector<ScalarType> > clone() const
    {
        size_t dim = this->size();
        std::shared_ptr<dotk::StdVector<ScalarType> > output(new dotk::StdVector<ScalarType>(dim, 0.));
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
    std::vector<ScalarType> m_Data;

private:
    StdVector(const dotk::StdVector<ScalarType> &);
    dotk::StdVector<ScalarType> & operator=(const dotk::StdVector<ScalarType> & rhs_);
};

}

#endif /* DOTK_SERIALVECTOR_HPP_ */
