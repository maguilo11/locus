/*
 * TRROM_TeuchosSerialDenseVector.hpp
 *
 *  Created on: Oct 19, 2016
 *      Author: maguilo
 */

#ifndef TRROM_TEUCHOSSERIALDENSEVECTOR_HPP_
#define TRROM_TEUCHOSSERIALDENSEVECTOR_HPP_

#include "TRROM_Vector.hpp"
#include "Teuchos_SerialDenseVector.hpp"

namespace trrom
{

template<typename ScalarType>
class TeuchosSerialDenseVector : public trrom::Vector<ScalarType>
{
public:
    explicit TeuchosSerialDenseVector(int num_elements_) :
            m_Data(new Teuchos::SerialDenseVector<int, ScalarType>(num_elements_))
    {
    }
    virtual ~TeuchosSerialDenseVector()
    {
    }

    // Scales a Vector by a real constant.
    void scale(const ScalarType & alpha_)
    {
        m_Data->scale(alpha_);
    }
    // Component wise multiplication of two vectors.
    void elementWiseMultiplication(const trrom::Vector<ScalarType> & input_)
    {
        int num_elements = this->size();
        assert(num_elements == input_.size());
        for(int index = 0; index < num_elements; ++index)
        {
            (*m_Data)[index] = (*m_Data)[index] * input_[index];
        }
    }
    // Constant times a Vector plus a Vector.
    void axpy(const ScalarType & alpha_, const trrom::Vector<ScalarType> & input_)
    {
        int this_increment = 1;
        int input_increment = 1;
        int num_elements = this->size();

        m_Data->AXPY(num_elements, alpha_, &(input_)[0], input_increment, m_Data->values(), this_increment);
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
            (*m_Data)[index] = (*m_Data)[index] < static_cast<ScalarType>(0.) ? -(*m_Data)[index] : (*m_Data)[index];
        }
    }
    // Returns the sum of all the elements in the container.
    ScalarType sum() const
    {
        int this_increment = 1;
        int num_elements = this->size();
        ScalarType output = m_Data->ASUM(num_elements, m_Data->values(), this_increment);
        return (output);
    }
    // Returns the inner product of two vectors.
    ScalarType dot(const trrom::Vector<ScalarType> & input_) const
    {
        int num_elements = 0;
        trrom::TeuchosSerialDenseVector<ScalarType> in(num_elements);
        in.copy(input_);
        ScalarType output = m_Data->dot(*in.data());
        return (output);
    }
    // Returns the euclidean norm of a Vector.
    ScalarType norm() const
    {
        ScalarType output = m_Data->normFrobenius();
        return (output);
    }
    // Assigns new contents to the Vector, replacing its current contents, and not modifying its size.
    void fill(const ScalarType & value_)
    {
        m_Data->putScalar(value_);
    }
    // Copies the elements in the range [first,last) into the range beginning at result.
    void copy(const trrom::Vector<ScalarType> & input_)
    {
        assert(this->size() == input_.size());
        for(int index = 0; index < this->size(); ++index)
        {
            m_Data->operator [](index) = input_.operator [](index);
        }
    }
    // Returns the number of elements in the Vector.
    int size() const
    {
        return (m_Data->length());
    }
    // Clones memory for an object of type trrom::Vector
    std::tr1::shared_ptr<trrom::Vector<ScalarType> > create() const
    {
        int num_elements = this->size();
        std::tr1::shared_ptr<trrom::TeuchosSerialDenseVector<ScalarType> > a_copy(new trrom::TeuchosSerialDenseVector<
                ScalarType>(num_elements));
        return (a_copy);
    }
    // Create object of type trrom::Vector
    std::tr1::shared_ptr<trrom::Vector<ScalarType> > create(const int & global_dim_) const
    {
        std::tr1::shared_ptr<trrom::TeuchosSerialDenseVector<ScalarType> > a_copy(new trrom::TeuchosSerialDenseVector<
                ScalarType>(global_dim_));
        return (a_copy);
    }
    // Operator overloads the square bracket operator
    ScalarType & operator [](int index_)
    {
        return (m_Data->operator ()(index_));
    }
    // Operator overloads the square bracket operator
    const ScalarType & operator [](int index_) const
    {
        return (m_Data->operator ()(index_));
    }

    const std::tr1::shared_ptr<Teuchos::SerialDenseVector<int, ScalarType> > & data() const
    {
        return (m_Data);
    }

private:
    std::tr1::shared_ptr<Teuchos::SerialDenseVector<int, ScalarType> > m_Data;

private:
    TeuchosSerialDenseVector(const trrom::TeuchosSerialDenseVector<ScalarType> &);
    trrom::TeuchosSerialDenseVector<ScalarType> & operator=(const trrom::TeuchosSerialDenseVector<ScalarType> &);
};

}

#endif /* TRROM_TEUCHOSSERIALDENSEVECTOR_HPP_ */
