/*
 * Locus_Vector.hpp
 *
 *  Created on: Oct 6, 2017
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef LOCUS_VECTOR_HPP_
#define LOCUS_VECTOR_HPP_

namespace locus
{

template<typename ScalarType, typename OrdinalType = size_t>
class Vector
{
public:
    virtual ~Vector()
    {
    }

    //! Scales a Vector by a ScalarType constant.
    virtual void scale(const ScalarType & aInput) = 0;
    //! Entry-Wise product of two vectors.
    virtual void entryWiseProduct(const locus::Vector<ScalarType, OrdinalType> & aInput) = 0;
    //! Update vector values with scaled values of A, this = beta*this + alpha*A.
    virtual void update(const ScalarType & aAlpha,
                        const locus::Vector<ScalarType, OrdinalType> & aInputVector,
                        const ScalarType & aBeta) = 0;
    //! Computes the absolute value of each element in the container.
    virtual void modulus() = 0;
    //! Returns the inner product of two vectors.
    virtual ScalarType dot(const locus::Vector<ScalarType, OrdinalType> & aInputVector) const = 0;
    //! Assigns new contents to the Vector, replacing its current contents, and not modifying its size.
    virtual void fill(const ScalarType & aValue) = 0;
    //! Returns the number of local elements in the Vector.
    virtual OrdinalType size() const = 0;
    //! Creates an object of type locus::Vector
    virtual std::shared_ptr<locus::Vector<ScalarType, OrdinalType>> create() const = 0;
    //! Operator overloads the square bracket operator
    virtual ScalarType & operator [](const OrdinalType & aIndex) = 0;
    //! Operator overloads the square bracket operator
    virtual const ScalarType & operator [](const OrdinalType & aIndex) const = 0;
};

}

#endif /* LOCUS_VECTOR_HPP_ */
