/*
 * DOTk_MexVector.hpp
 *
 *  Created on: Dec 23, 2016
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_MEXVECTOR_HPP_
#define DOTK_MEXVECTOR_HPP_

#include <mex.h>

#include "vector.hpp"

namespace dotk
{

class MexVector : public dotk::Vector<double>
{
public:
    //! @name Constructors/destructors
    //@{
    /*!
     * Creates a MexVector object
     * Parameters:
     *    \param In
     *          length_: vector length
     *    \param In
     *          a_initial_value: if defined, fill vector with value. if not defined, fills the vector with zeros.
     *
     * \return Reference to MexVector.
     *
     **/
    MexVector(size_t length_, double a_initial_value);
    /*!
     * Creates a MexVector object by making a deep copy of the input MEX array
     * Parameters:
     *    \param In
     *          array_: MEX array pointer
     *
     * \return Reference to MexVector.
     *
     **/
    MexVector(const mxArray* array_);
    //! MexVector destructor.
    virtual ~MexVector();
    //@}

    //! @name Public functions
    //@{
    //! Scales a vector by a real constant.
    void scale(const double & alpha_);
    //! Performs element wise multiplication of two vectors.
    void elementWiseMultiplication(const dotk::Vector<double> & input_);
    //! Update vector values with scaled values of A, this = beta*this + alpha*A.
    void update(const double & alpha_, const dotk::Vector<double> & input_, const double & beta_);
    //! Returns the maximum element in a range.
    double max() const;
    //! Returns the minimum element in a range.
    double min() const;
    //! Computes the absolute value of each element in the container.
    void abs();
    //! Returns the sum of all the elements in the container.
    double sum() const;
    //! Returns the inner product of two vectors.
    double dot(const dotk::Vector<double> & input_) const;
    //! Returns the euclidean norm of a vector.
    double norm() const;
    //! Assigns new contents to the vector, replacing its current contents, and not modifying its size.
    void fill(const double & value_);
    //! Returns the number of elements in the vector.
    size_t size() const;
    //! Clones memory for an object of type dotk::Vector
    std::shared_ptr<dotk::Vector<double> > clone() const;
    //! Operator overloads the square bracket operator
    double & operator [](size_t index_);
    //! Operator overloads the square bracket operator
    const double & operator [](size_t index_) const;
    //@}

    //! @name MEX-specific public functions
    //@{
    //! Get non-constant real numeric pointer for numeric array.
    double* data();
    //! Get constant real numeric pointer for numeric array.
    const double* data() const;
    //! Get non-constant pointer to MEX array.
    mxArray* array();
    //! Get constant pointer to MEX array.
    const mxArray* array() const;
    //! Set new contents to this MEX array, replacing its current contents, and not modifying its size.
    void setMxArray(const mxArray* input_);
    //@}

private:
    mxArray* m_Data;

private:
    MexVector(const dotk::MexVector &);
    dotk::MexVector & operator=(const dotk::MexVector & rhs_);
};

}

#endif /* DOTK_MEXVECTOR_HPP_ */
