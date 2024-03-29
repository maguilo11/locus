/*
 * vector.hpp
 *
 *  Created on: Aug 25, 2014
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef VECTOR_HPP_
#define VECTOR_HPP_

#include <cstdio>
#include <memory>
#include <cstdlib>

namespace dotk
{

template<typename ScalarType>
class Vector
{
public:
    Vector()
    {
    }
    virtual ~Vector()
    {
    }
    //! Scales a vector by a real constant.
    virtual void scale(const ScalarType & alpha_) = 0;
    //! Performs element wise multiplication of two vectors.
    virtual void elementWiseMultiplication(const dotk::Vector<ScalarType> & input_) = 0;
    //! Update vector values with scaled values of A, this = beta*this + alpha*A.
    virtual void update(const ScalarType & alpha_,
                        const dotk::Vector<ScalarType> & input_,
                        const ScalarType & beta_) = 0;
    //! Returns the maximum element in a range.
    virtual ScalarType max() const = 0;
    //! Returns the minimum element in a range.
    virtual ScalarType min() const = 0;
    //! Computes the absolute value of each element in the container.
    virtual void abs() = 0;
    //! Returns the sum of all the elements in the container.
    virtual ScalarType sum() const = 0;
    //! Returns the inner product of two vectors.
    virtual ScalarType dot(const dotk::Vector<ScalarType> & input_) const = 0;
    //! Returns the euclidean norm of a vector.
    virtual ScalarType norm() const = 0;
    //! Assigns new contents to the vector, replacing its current contents, and not modifying its size.
    virtual void fill(const ScalarType & value_) = 0;
    //! Returns the number of elements in the vector.
    virtual size_t size() const = 0;
    //! Clones memory for an object of type dotk::Vector
    virtual std::shared_ptr< dotk::Vector<ScalarType> > clone() const = 0;
    //! Operator overloads the square bracket operator
    virtual ScalarType & operator [](size_t index_) = 0;
    //! Operator overloads the square bracket operator
    virtual const ScalarType & operator [](size_t index_) const = 0;

    //! Returns shared pointer to dual vector
    virtual const std::shared_ptr<dotk::Vector<ScalarType> > & dual() const
    {
        std::perror("\n**** Unimplemented Function dotk::Vector::dual. ABORT. ****\n");
        std::abort();
    }
    //! Returns shared pointer to state vector
    virtual const std::shared_ptr<dotk::Vector<ScalarType> > & state() const
    {
        std::perror("\n**** Unimplemented Function dotk::Vector::state. ABORT. ****\n");
        std::abort();
    }
    //! Returns shared pointer to control vector
    virtual const std::shared_ptr<dotk::Vector<ScalarType> > & control() const
    {
        std::perror("\n**** Unimplemented Function dotk::Vector::control. ABORT. ****\n");
        std::abort();
    }

private:
    Vector(const dotk::Vector<ScalarType> &);
    dotk::Vector<ScalarType> & operator=(const dotk::Vector<ScalarType> & rhs_);
};

}

#endif /* VECTOR_HPP_ */
