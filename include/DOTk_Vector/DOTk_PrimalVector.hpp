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

template<typename ScalarType>
class DOTk_PrimalVector: public dotk::Vector<ScalarType>
{
public:
    explicit DOTk_PrimalVector(const dotk::DOTk_Primal & primal_);
    explicit DOTk_PrimalVector(const dotk::Vector<ScalarType> & control_);
    DOTk_PrimalVector(const dotk::Vector<ScalarType> & control_, const dotk::Vector<ScalarType> & state_);
    virtual ~DOTk_PrimalVector();

    // Scales a vector by a real constant.
    virtual void scale(const ScalarType & alpha_);
    // Component wise multiplication of two vectors.
    virtual void elementWiseMultiplication(const dotk::Vector<ScalarType> & input_);
    //! Update vector values with scaled values of A, this = beta*this + alpha*A.
    void update(const ScalarType & alpha_, const dotk::Vector<ScalarType> & input_, const ScalarType & beta_);
    // Returns the maximum element in a range.
    virtual ScalarType max() const;
    // Returns the minimum element in a range.
    virtual ScalarType min() const;
    // Computes the absolute value of each element in the container.
    virtual void abs();
    // Returns the sum of all the elements in the container.
    virtual ScalarType sum() const;
    // Returns the inner product of two vectors.
    virtual ScalarType dot(const dotk::Vector<ScalarType> & input_) const;
    // Returns the euclidean norm of a vector.
    virtual ScalarType norm() const;
    // Assigns new contents to the vector, replacing its current contents, and not modifying its size.
    virtual void fill(const ScalarType & value_);
    // Returns the number of elements in the vector.
    virtual size_t size() const;
    // Clones memory for an object of ScalarType dotk::DOTk_MultiVector
    virtual std::tr1::shared_ptr<dotk::Vector<ScalarType> > clone() const;
    // Returns shared pointer to state vector
    virtual const std::tr1::shared_ptr<dotk::Vector<ScalarType> > & state() const;
    // Returns shared pointer to control vector
    virtual const std::tr1::shared_ptr<dotk::Vector<ScalarType> > & control() const;
    // Operator overloads the square bracket operator
    virtual ScalarType & operator [](size_t index_);
    // Operator overloads the const square bracket operator
    virtual const ScalarType & operator [](size_t index_) const;

private:
    void initialize(const dotk::DOTk_Primal & primal_);
    void initialize(const dotk::Vector<ScalarType> & control_, const dotk::Vector<ScalarType> & state_);

private:
    size_t m_Size;
    std::tr1::shared_ptr<dotk::Vector<ScalarType> > m_State;
    std::tr1::shared_ptr<dotk::Vector<ScalarType> > m_Control;

private:
    DOTk_PrimalVector(const dotk::DOTk_PrimalVector<ScalarType> &);
    dotk::DOTk_PrimalVector<ScalarType> & operator=(const dotk::DOTk_PrimalVector<ScalarType> & rhs_);

};

}

#endif /* PRIMALVECTOR_HPP_ */
