/*
 * TRROM_Vector.hpp
 *
 *  Created on: Aug 25, 2014
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef TRROM_VECTOR_HPP_
#define TRROM_VECTOR_HPP_

#include "TRROM_Types.hpp"

namespace trrom
{

template<typename ScalarType>
class Vector
{
public:
    virtual ~Vector()
    {
    }

    // Scales a Vector by a real constant.
    virtual void scale(const ScalarType & alpha_) = 0;
    // Element-wise multiplication of two vectors.
    virtual void elementWiseMultiplication(const trrom::Vector<ScalarType> & input_) = 0;
    // Constant times a Vector plus a Vector.
    virtual void update(const ScalarType & alpha_,
                        const trrom::Vector<ScalarType> & input_,
                        const ScalarType & beta_) = 0;
    // Returns the maximum element in a range and its global position index.
    virtual ScalarType max(int & index_) const = 0;
    // Returns the minimum element in a range and its global position index.
    virtual ScalarType min(int & index_) const = 0;
    // Computes the absolute value of each element in the container.
    virtual void modulus() = 0;
    // Returns the sum of all the elements in the container.
    virtual ScalarType sum() const = 0;
    // Returns the inner product of two vectors.
    virtual ScalarType dot(const trrom::Vector<ScalarType> & input_) const = 0;
    // Returns the euclidean norm of a Vector.
    virtual ScalarType norm() const = 0;
    // Assigns new contents to the Vector, replacing its current contents, and not modifying its size.
    virtual void fill(const ScalarType & value_) = 0;
    // Returns the number of local elements in the Vector.
    virtual int size() const = 0;
    // Creates memory for an object of type trrom::Vector
    virtual std::tr1::shared_ptr<trrom::Vector<ScalarType> > create() const = 0;
    // Creates object of type trrom::Vector
    virtual std::tr1::shared_ptr<trrom::Vector<ScalarType> > create(int global_dim_ = 0) const = 0;
    // Operator overloads the square bracket operator
    virtual ScalarType & operator [](int index_) = 0;
    // Operator overloads the square bracket operator
    virtual const ScalarType & operator [](int index_) const = 0;
};

}

#endif /* TRROM_VECTOR_HPP_ */
