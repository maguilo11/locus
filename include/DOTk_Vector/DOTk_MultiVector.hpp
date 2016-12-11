/*
 * DOTk_MultiVector.hpp
 *
 *  Created on: Jul 4, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_MULTIVECTOR_HPP_
#define DOTK_MULTIVECTOR_HPP_

#include "vector.hpp"

namespace dotk
{

class DOTk_Primal;

template<typename ScalarType>
class DOTk_MultiVector: public dotk::Vector<ScalarType>
{
public:
    explicit DOTk_MultiVector(const dotk::DOTk_Primal & primal_);
    DOTk_MultiVector(const dotk::Vector<ScalarType> & control_,
                     const dotk::Vector<ScalarType> & dual_);
    DOTk_MultiVector(const dotk::Vector<ScalarType> & control_,
                     const dotk::Vector<ScalarType> & state_,
                     const dotk::Vector<ScalarType> & dual_);
    virtual ~DOTk_MultiVector();

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
    // Gathers data from private member data of a group to one member.
    virtual void gather(ScalarType* input_) const;
    // Returns the number of elements in the vector.
    virtual size_t size() const;
    // Clones memory for an object of ScalarType dotk::DOTk_MultiVector
    virtual std::tr1::shared_ptr<dotk::Vector<ScalarType> > clone() const;
    // Operator overloads the square bracket operator
    virtual ScalarType & operator [](size_t index_);
    // Operator overloads the const square bracket operator
    virtual const ScalarType & operator [](size_t index_) const;
    // Returns shared pointer to dual vector
    virtual const std::tr1::shared_ptr<dotk::Vector<ScalarType> > & dual() const;
    // Returns shared pointer to state vector
    virtual const std::tr1::shared_ptr<dotk::Vector<ScalarType> > & state() const;
    // Returns shared pointer to control vector
    virtual const std::tr1::shared_ptr<dotk::Vector<ScalarType> > & control() const;


private:
    void initialize(const dotk::DOTk_Primal & primal_);
    void initialize(const dotk::Vector<ScalarType> & control_, const dotk::Vector<ScalarType> & dual_);
    void initialize(const dotk::Vector<ScalarType> & control_,
                    const dotk::Vector<ScalarType> & state_,
                    const dotk::Vector<ScalarType> & dual_);

private:
    size_t m_Size;
    std::tr1::shared_ptr<dotk::Vector<ScalarType> > m_Dual;
    std::tr1::shared_ptr<dotk::Vector<ScalarType> > m_State;
    std::tr1::shared_ptr<dotk::Vector<ScalarType> > m_Control;

private:
    DOTk_MultiVector(const dotk::DOTk_MultiVector<ScalarType> &);
    dotk::DOTk_MultiVector<ScalarType> & operator=(const dotk::DOTk_MultiVector<ScalarType> & rhs_);
};

}

#endif /* DOTK_MULTIVECTOR_HPP_ */
