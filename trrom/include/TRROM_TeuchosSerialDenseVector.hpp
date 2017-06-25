/*
 * TRROM_TeuchosSerialDenseVector.hpp
 *
 *  Created on: Oct 19, 2016
 *      Author: maguilo
 */

#ifndef TRROM_TEUCHOSSERIALDENSEVECTOR_HPP_
#define TRROM_TEUCHOSSERIALDENSEVECTOR_HPP_

#include <cmath>
#include "TRROM_Vector.hpp"
#include "Teuchos_SerialDenseVector.hpp"

namespace trrom
{

class TeuchosSerialDenseVector : public trrom::Vector<double>
{
public:
    explicit TeuchosSerialDenseVector(int num_elements_) :
            m_Data(new Teuchos::SerialDenseVector<int, double>(num_elements_))
    {
    }
    virtual ~TeuchosSerialDenseVector()
    {
    }

    // Scales a Vector by a real constant.
    void scale(const double & alpha_)
    {
        m_Data->scale(alpha_);
    }
    // Component wise multiplication of two vectors.
    void elementWiseMultiplication(const trrom::Vector<double> & input_)
    {
        int num_elements = this->size();
        assert(num_elements == input_.size());
        for(int index = 0; index < num_elements; ++index)
        {
            (*m_Data)[index] = (*m_Data)[index] * input_[index];
        }
    }
    // Constant times a Vector plus a Vector.
    void update(const double & alpha_, const trrom::Vector<double> & input_, const double & beta_)
    {
        int num_elements = this->size();
        for(int index = 0; index < num_elements; ++index)
        {
            (*m_Data)[index] = beta_ * (*m_Data)[index] + alpha_ * input_[index];
        }
    }
    // Returns the maximum element in a range.
    double max(int & index_) const
    {
        double max_value = 0;
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
    double min(int & index_) const
    {
        double min_value = 0;
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
            (*m_Data)[index] = (*m_Data)[index] < static_cast<double>(0.) ? -(*m_Data)[index] : (*m_Data)[index];
        }
    }
    // Returns the sum of all the elements in the container.
    double sum() const
    {
        int this_increment = 1;
        int num_elements = this->size();
        double output = m_Data->ASUM(num_elements, m_Data->values(), this_increment);
        return (output);
    }
    // Returns the inner product of two vectors.
    double dot(const trrom::Vector<double> & input_) const
    {
        const trrom::TeuchosSerialDenseVector & input = dynamic_cast<const trrom::TeuchosSerialDenseVector &>(input_);
        double output = m_Data->dot(*input.data());
        return (output);
    }
    // Returns the euclidean norm of a Vector.
    double norm() const
    {
        double output = this->dot(*this);
        output = std::sqrt(output);
        return (output);
    }
    // Assigns new contents to the Vector, replacing its current contents, and not modifying its size.
    void fill(const double & value_)
    {
        m_Data->putScalar(value_);
    }
    // Returns the number of elements in the Vector.
    int size() const
    {
        return (m_Data->length());
    }
    /*! Create object of type trrom::Vector */
    std::shared_ptr<trrom::Vector<double> > create(int global_length_ = 0) const
    {
        std::shared_ptr<trrom::TeuchosSerialDenseVector> this_copy;
        if(global_length_ == 0)
        {
            int num_elements = this->size();
            this_copy.reset(new trrom::TeuchosSerialDenseVector(num_elements));
        }
        else
        {
            this_copy.reset(new trrom::TeuchosSerialDenseVector(global_length_));
        }
        return (this_copy);
    }
    // Operator overloads the square bracket operator
    double & operator [](int index_)
    {
        return (m_Data->operator ()(index_));
    }
    // Operator overloads the square bracket operator
    const double & operator [](int index_) const
    {
        return (m_Data->operator ()(index_));
    }
    const std::shared_ptr<Teuchos::SerialDenseVector<int, double> > & data() const
    {
        return (m_Data);
    }

private:
    std::shared_ptr<Teuchos::SerialDenseVector<int, double> > m_Data;

private:
    TeuchosSerialDenseVector(const trrom::TeuchosSerialDenseVector &);
    trrom::TeuchosSerialDenseVector & operator=(const trrom::TeuchosSerialDenseVector &);
};

}

#endif /* TRROM_TEUCHOSSERIALDENSEVECTOR_HPP_ */
