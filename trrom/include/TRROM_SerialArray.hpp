/*
 * TRROM_SerialArray.hpp
 *
 *  Created on: Apr 3, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef TRROM_SERIALARRAY_HPP_
#define TRROM_SERIALARRAY_HPP_

#include "TRROM_Vector.hpp"

namespace trrom
{

template<typename ScalarType>
class SerialArray : public trrom::Vector<ScalarType>
{
public:
    SerialArray(int dim_, ScalarType value_ = 0.) :
            m_LocalDim(dim_),
            m_Data(new ScalarType[dim_])
    {
        this->fill(value_);
    }
    virtual ~SerialArray()
    {
        delete[] m_Data;
        m_Data = nullptr;
    }

    // Scales a vector by a real constant.
    void scale(const ScalarType & alpha_)
    {
        int dim = this->size();
        for(int i = 0; i < dim; ++i)
        {
            m_Data[i] = alpha_ * m_Data[i];
        }
    }
    // Component wise multiplication of two vectors.
    void elementWiseMultiplication(const trrom::Vector<ScalarType> & input_)
    {
        int dim = this->size();
        assert(dim == input_.size());
        for(int i = 0; i < dim; ++i)
        {
            m_Data[i] = m_Data[i] * input_[i];
        }
    }
    // Constant times a vector plus a vector.
    void update(const ScalarType & alpha_, const trrom::Vector<ScalarType> & input_, const ScalarType & beta_)
    {
        int dim = this->size();
        assert(dim == input_.size());
        for(int i = 0; i < dim; ++i)
        {
            m_Data[i] = beta_ * m_Data[i] + alpha_ * input_[i];
        }
    }
    // Returns the maximum element in a range.
    ScalarType max(int & index_) const
    {
        ScalarType max_value = 0;
        int dim = this->size();
        for(int i = 0; i < dim; ++i)
        {
            if(m_Data[i] > max_value)
            {
                max_value = m_Data[i];
                index_ = i;
            }
        }
        return (max_value);
    }
    // Returns the minimum element in a range.
    ScalarType min(int & index_) const
    {
        ScalarType min_value = 0;
        for(int i = 0; i < this->size(); ++i)
        {
            if(m_Data[i] < min_value)
            {
                min_value = m_Data[i];
                index_ = i;
            }
        }
        return (min_value);
    }
    // Computes the absolute value of each element in the container.
    void modulus()
    {
        int dim = this->size();
        for(int index = 0; index < dim; ++index)
        {
            m_Data[index] = m_Data[index] < static_cast<ScalarType>(0.) ? -(m_Data[index]) : m_Data[index];
        }
    }
    // Returns the sum of all the elements in the container.
    ScalarType sum() const
    {
        ScalarType result = 0.;
        int dim = this->size();
        for(int i = 0; i < dim; ++i)
        {
            result += m_Data[i];
        }
        return (result);
    }
    // Returns the inner product of two vectors.
    ScalarType dot(const trrom::Vector<ScalarType> & input_) const
    {
        ScalarType result = 0;
        int dim = this->size();
        assert(dim == input_.size());

        for(int i = 0; i < dim; ++i)
        {
            result += m_Data[i] * input_[i];
        }

        return (result);
    }
    // Returns the euclidean norm of a vector.
    ScalarType norm() const
    {
        ScalarType value = 0.;
        int dim = this->size();

        for(int i = 0; i < dim; ++i)
        {
            value += m_Data[i] * m_Data[i];
        }

        ScalarType result = sqrt(value);

        return (result);
    }
    // Assigns new contents to the vector, replacing its current contents, and not modifying its size.
    void fill(const ScalarType & value_)
    {
        int dim = this->size();
        for(int i = 0; i < dim; ++i)
        {
            m_Data[i] = value_;
        }
    }
    // Returns the number of elements in the vector.
    int size() const
    {
        return (m_LocalDim);
    }
    // Create object of type trrom::vector
    std::shared_ptr<trrom::Vector<ScalarType> > create(int global_length_ = 0) const
    {
        std::shared_ptr<trrom::SerialArray<ScalarType> > this_copy;
        if(global_length_ == 0)
        {
            int length = this->size();
            this_copy = std::make_shared<trrom::SerialArray<ScalarType>>(length);
        }
        else
        {
            this_copy = std::make_shared<trrom::SerialArray<ScalarType>>(global_length_);
        }
        return (this_copy);
    }
    // Operator overloads the square bracket operator
    ScalarType & operator [](int index_)
    {
        return (m_Data[index_]);
    }
    // Operator overloads the const square bracket operator
    const ScalarType & operator [](int index_) const
    {
        return (m_Data[index_]);
    }

private:
    int m_LocalDim;
    ScalarType* m_Data;

private:
    SerialArray(const trrom::SerialArray<ScalarType> &);
    trrom::SerialArray<ScalarType> & operator=(const trrom::SerialArray<ScalarType> & rhs_);
};

}

#endif /* TRROM_SERIALARRAY_HPP_ */
