/*
 * TRROM_TeuchosArray.hpp
 *
 *  Created on: Oct 19, 2016
 *      Author: maguilo
 */

#ifndef TRROM_TEUCHOSARRAY_HPP_
#define TRROM_TEUCHOSARRAY_HPP_

#include <cmath>
#include "TRROM_Vector.hpp"
#include "Teuchos_Array.hpp"

namespace trrom
{

template<typename ScalarType>
class TeuchosArray : public trrom::Vector<ScalarType>
{
public:
    TeuchosArray(int dimension_, ScalarType value_ = 0) :
            m_Data(new Teuchos::Array<ScalarType>(dimension_, value_))
    {
    }
    virtual ~TeuchosArray()
    {
    }

    // Scales a Vector by a real constant.
    void scale(const ScalarType & alpha_)
    {
        int dim = this->size();
        for(int index = 0; index < dim; ++index)
        {
            (*m_Data)[index] = alpha_ * (*m_Data)[index];
        }
    }
    // Component wise multiplication of two vectors.
    void elementWiseMultiplication(const trrom::Vector<ScalarType> & input_)
    {
        int dim = this->size();
        assert(dim == input_.size());
        for(int index = 0; index < dim; ++index)
        {
            (*m_Data)[index] = (*m_Data)[index] * input_[index];
        }
    }
    // Constant times a Vector plus a Vector.
    void update(const ScalarType & alpha_, const trrom::Vector<ScalarType> & input_, const ScalarType & beta_)
    {
        int dim = this->size();
        assert(dim == input_.size());
        for(int index = 0; index < dim; ++index)
        {
            (*m_Data)[index] = beta_ * (*m_Data)[index] + alpha_ * input_[index];
        }
    }
    // Returns the maximum element in a range.
    ScalarType max(int & index_) const
    {
        ScalarType max_value = 0;
        int dim = this->size();
        for(int index = 0; index < dim; ++index)
        {
            if((*m_Data)[index] > max_value)
            {
                max_value = (*m_Data)[index];
                index_ = index;
            }
        }
        return (max_value);
    }
    // Returns the minimum element in a range.
    ScalarType min(int & index_) const
    {
        ScalarType min_value = 0;
        for(int index = 0; index < this->size(); ++index)
        {
            if((*m_Data)[index] < min_value)
            {
                min_value = (*m_Data)[index];
                index_ = index;
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
            (*m_Data)[index] = (*m_Data)[index] < static_cast<ScalarType>(0.) ? -((*m_Data)[index]) : (*m_Data)[index];
        }
    }
    // Returns the sum of all the elements in the container.
    ScalarType sum() const
    {
        ScalarType result = 0.;
        int dim = this->size();
        for(int index = 0; index < dim; ++index)
        {
            result += (*m_Data)[index];
        }
        return (result);
    }
    // Returns the inner product of two vectors.
    ScalarType dot(const trrom::Vector<ScalarType> & input_) const
    {
        ScalarType result = 0;
        int dim = this->size();
        assert(dim == input_.size());

        for(int index = 0; index < dim; ++index)
        {
            result += (*m_Data)[index] * input_[index];
        }

        return (result);
    }
    // Returns the euclidean norm of a Vector.
    ScalarType norm() const
    {
        ScalarType value = 0.;
        int dim = this->size();
        for(int index = 0; index < dim; ++index)
        {
            value += (*m_Data)[index] * (*m_Data)[index];
        }
        ScalarType result = std::sqrt(value);

        return (result);
    }
    // Assigns new contents to the Vector, replacing its current contents, and not modifying its size.
    void fill(const ScalarType & value_)
    {
        int dim = this->size();
        for(int index = 0; index < dim; ++index)
        {
            (*m_Data)[index] = value_;
        }
    }
    // Returns the number of elements in the Vector.
    int size() const
    {
        int num_elements = m_Data->size();
        return (num_elements);
    }
    // Create object of type trrom::Vector
    std::tr1::shared_ptr<trrom::Vector<ScalarType> > create(int global_dim_ = 0) const
    {
        std::tr1::shared_ptr<trrom::TeuchosArray<ScalarType> > vector;
        if(global_dim_ == 0)
        {
            int dimension = this->size();
            vector.reset(new trrom::TeuchosArray<ScalarType>(dimension));
        }
        else
        {
            vector.reset(new trrom::TeuchosArray<ScalarType>(global_dim_));
        }
        return (vector);
    }
    // Operator overloads the square bracket operator
    ScalarType & operator [](int index_)
    {
        return ((*m_Data)[index_]);
    }
    // Operator overloads the square bracket operator
    const ScalarType & operator [](int index_) const
    {
        return ((*m_Data)[index_]);
    }

    const std::tr1::shared_ptr<Teuchos::Array<ScalarType> > & data() const
    {
        return (m_Data);
    }

private:
    std::tr1::shared_ptr<Teuchos::Array<ScalarType> > m_Data;

private:
    TeuchosArray(const trrom::TeuchosArray<ScalarType> &);
    trrom::TeuchosArray<ScalarType> & operator=(const trrom::TeuchosArray<ScalarType> &);
};

}

#endif /* TRROM_TEUCHOSARRAY_HPP_ */
