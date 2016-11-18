/*
 * TRROM_SerialVector.hpp
 *
 *  Created on: Aug 25, 2014
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef TRROM_SERIALVECTOR_HPP_
#define TRROM_SERIALVECTOR_HPP_

#include "TRROM_Vector.hpp"

namespace trrom
{

template<typename ScalarType>
class SerialVector : public trrom::Vector<ScalarType>
{
public:
    explicit SerialVector(std::vector<ScalarType> & data_) :
            m_Data(data_)
    {
    }
    SerialVector(int dim_, ScalarType value_ = 0.) :
            m_Data(std::vector<ScalarType>(dim_, value_))
    {
    }
    virtual ~SerialVector()
    {
    }

    // Scales a Vector by a real constant.
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
    // Constant times a Vector plus a Vector.
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
            m_Data.data()[index] =
                    m_Data.data()[index] < static_cast<ScalarType>(0.) ? -(m_Data.data()[index]) : m_Data.data()[index];
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
        int dim = this->size();
        assert(dim == input_.size());

        ScalarType result = 0.;
        for(int i = 0; i < dim; ++i)
        {
            result += m_Data[i] * input_[i];
        }

        return (result);
    }
    // Returns the euclidean norm of a Vector.
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
    // Assigns new contents to the Vector, replacing its current contents, and not modifying its size.
    void fill(const ScalarType & value_)
    {
        int dim = this->size();
        for(int i = 0; i < dim; ++i)
        {
            m_Data[i] = value_;
        }
    }
    // Returns the number of elements in the Vector.
    int size() const
    {
        int dim = m_Data.size();
        return (dim);
    }
    // Create object of type trrom::Vector
    std::tr1::shared_ptr<trrom::Vector<ScalarType> > create(int global_dim_ = 0) const
    {
        std::tr1::shared_ptr<trrom::SerialVector<ScalarType> > this_copy;
        if(global_dim_ == 0)
        {
            int length = this->size();
            this_copy.reset(new trrom::SerialVector<ScalarType>(length));
        }
        else
        {
            this_copy.reset(new trrom::SerialVector<ScalarType>(global_dim_));
        }
        return (this_copy);
    }
    // Operator overloads the square bracket operator
    ScalarType & operator [](int index_)
    {
        return (m_Data.operator [](index_));
    }
    // Operator overloads the const square bracket operator
    const ScalarType & operator [](int index_) const
    {
        return (m_Data.operator [](index_));
    }

private:
    std::vector<ScalarType> m_Data;

private:
    SerialVector(const trrom::SerialVector<ScalarType> &);
    trrom::SerialVector<ScalarType> & operator=(const trrom::SerialVector<ScalarType> & rhs_);
};

}

#endif /* TRROM_SERIALVECTOR_HPP_ */
