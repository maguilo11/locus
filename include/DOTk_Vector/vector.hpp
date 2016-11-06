/*
 * vector.hpp
 *
 *  Created on: Aug 25, 2014
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef VECTOR_HPP_
#define VECTOR_HPP_

#include <cstdio>
#include <cstdlib>
#include <tr1/memory>

namespace dotk
{

template<typename Type>
class vector
{
public:
    vector()
    {
    }
    virtual ~vector()
    {
    }
    // Scales a vector by a real constant.
    virtual void scale(const Type & alpha_) = 0;
    // Component wise multiplication of two vectors.
    virtual void cwiseProd(const dotk::vector<Type> & input_) = 0;
    // Constant times a vector plus a vector.
    virtual void axpy(const Type & alpha_, const dotk::vector<Type> & input_) = 0;
    // Returns the maximum element in a range.
    virtual Type max() const = 0;
    // Returns the minimum element in a range.
    virtual Type min() const = 0;
    // Computes the absolute value of each element in the container.
    virtual void abs() = 0;
    // Returns the sum of all the elements in the container.
    virtual Type sum() const = 0;
    // Returns the inner product of two vectors.
    virtual Type dot(const dotk::vector<Type> & input_) const = 0;
    // Returns the euclidean norm of a vector.
    virtual Type norm() const = 0;
    // Assigns new contents to the vector, replacing its current contents, and not modifying its size.
    virtual void fill(const Type & value_) = 0;
    // Copies the elements in the range [first,last) into the range beginning at result.
    virtual void copy(const dotk::vector<Type> & input_) = 0;
    // Gathers data from private member data of a group to one member.
    virtual void gather(Type* input_) const = 0;
    // Returns the number of elements in the vector.
    virtual size_t size() const = 0;
    // Clones memory for an object of type dotk::vector
    virtual std::tr1::shared_ptr< dotk::vector<Type> > clone() const = 0;
    // Operator overloads the square bracket operator
    virtual Type & operator [](size_t index_) = 0;
    // Operator overloads the square bracket operator
    virtual const Type & operator [](size_t index_) const = 0;
    // Returns shared pointer to dual vector
    virtual const std::tr1::shared_ptr<dotk::vector<Type> > & dual() const
    {
        std::perror("\n**** Unimplemented Function dotk::vector::dual. ABORT. ****\n");
        std::abort();
    }
    // Returns shared pointer to state vector
    virtual const std::tr1::shared_ptr<dotk::vector<Type> > & state() const
    {
        std::perror("\n**** Unimplemented Function dotk::vector::state. ABORT. ****\n");
        std::abort();
    }
    // Returns shared pointer to control vector
    virtual const std::tr1::shared_ptr<dotk::vector<Type> > & control() const
    {
        std::perror("\n**** Unimplemented Function dotk::vector::control. ABORT. ****\n");
        std::abort();
    }

private:
    vector(const dotk::vector<Type> &);
    dotk::vector<Type> & operator=(const dotk::vector<Type> & rhs_);
};

}

#endif /* VECTOR_HPP_ */
