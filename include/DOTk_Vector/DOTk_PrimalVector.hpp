/*
 * DOTk_PrimalVector.hpp
 *
 *  Created on: Dec 20, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef PRIMALVECTOR_HPP_
#define PRIMALVECTOR_HPP_

#include "vector.hpp"

namespace dotk
{

class DOTk_Primal;

template<typename Type>
class DOTk_PrimalVector: public dotk::vector<Type>
{
public:
    explicit DOTk_PrimalVector(const dotk::DOTk_Primal & primal_);
    explicit DOTk_PrimalVector(const dotk::vector<Type> & control_);
    DOTk_PrimalVector(const dotk::vector<Type> & control_, const dotk::vector<Type> & state_);
    virtual ~DOTk_PrimalVector();

    // Scales a vector by a real constant.
    virtual void scale(const Type & alpha_);
    // Component wise multiplication of two vectors.
    virtual void cwiseProd(const dotk::vector<Type> & input_);
    // Constant times a vector plus a vector.
    virtual void axpy(const Type & alpha_, const dotk::vector<Type> & input_);
    // Returns the maximum element in a range.
    virtual Type max() const;
    // Returns the minimum element in a range.
    virtual Type min() const;
    // Computes the absolute value of each element in the container.
    virtual void abs();
    // Returns the sum of all the elements in the container.
    virtual Type sum() const;
    // Returns the inner product of two vectors.
    virtual Type dot(const dotk::vector<Type> & input_) const;
    // Returns the euclidean norm of a vector.
    virtual Type norm() const;
    // Assigns new contents to the vector, replacing its current contents, and not modifying its size.
    virtual void fill(const Type & value_);
    // Copies the elements in the range [first,last) into the range beginning at result.
    virtual void copy(const dotk::vector<Type> & input_);
    // Gathers data from private member data of a group to one member.
    virtual void gather(Type* input_) const;
    // Returns the number of elements in the vector.
    virtual size_t size() const;
    // Clones memory for an object of type dotk::DOTk_MultiVector
    virtual std::tr1::shared_ptr<dotk::vector<Type> > clone() const;
    // Returns shared pointer to state vector
    virtual const std::tr1::shared_ptr<dotk::vector<Type> > & state() const;
    // Returns shared pointer to control vector
    virtual const std::tr1::shared_ptr<dotk::vector<Type> > & control() const;
    // Operator overloads the square bracket operator
    virtual Type & operator [](size_t index_);
    // Operator overloads the const square bracket operator
    virtual const Type & operator [](size_t index_) const;

private:
    void initialize(const dotk::DOTk_Primal & primal_);
    void initialize(const dotk::vector<Type> & control_, const dotk::vector<Type> & state_);

private:
    size_t m_Size;
    std::tr1::shared_ptr<dotk::vector<Type> > m_State;
    std::tr1::shared_ptr<dotk::vector<Type> > m_Control;

private:
    DOTk_PrimalVector(const dotk::DOTk_PrimalVector<Type> &);
    dotk::DOTk_PrimalVector<Type> & operator=(const dotk::DOTk_PrimalVector<Type> & rhs_);

};

}

#endif /* PRIMALVECTOR_HPP_ */
