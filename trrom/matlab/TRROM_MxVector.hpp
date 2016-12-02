/*
 * TRROM_MxVector.hpp
 *
 *  Created on: Nov 21, 2016
 *      Author: maguilo
 */

#ifndef TRROM_MXVECTOR_HPP_
#define TRROM_MXVECTOR_HPP_

#include <mex.h>
#include <tr1/memory>
#include "TRROM_Vector.hpp"

namespace trrom
{

//! MxVector: A class for constructing and using mex arrays that make use of the base trrom::Vector framework.

class MxVector : public trrom::Vector<double>
{
public:
    //! @name Constructors/destructors
    //@{
    /*!
     * Creates a MxVector object
     * Parameters:
     *    \param In
     *          a_length: vector length
     *    \param In
     *          a_initial_value: if defined, fill vector with value. if not defined, fills the vector with zeros.
     *
     * \return Reference to MxVector.
     *
     **/
    MxVector(int a_length, double a_initial_value = 0);
    /*!
     * Creates a MxVector object by making a deep copy of the input MEX array
     * Parameters:
     *    \param In
     *          array_: MEX array pointer
     *
     * \return Reference to MxVector.
     *
     **/
    explicit MxVector(const mxArray* array_);
    //! MxVector destructor.
    virtual ~MxVector();
    //@}

    /*! Scales Vector by a real constant. */
    void scale(const double & a_alpha);
    /*! Element-wise multiplication of two vectors. */
    void elementWiseMultiplication(const trrom::Vector<double> & a_input);
    /*! Update vector values with scaled values of A, this = beta*this + alpha*A. */
    void update(const double & a_alpha, const trrom::Vector<double> & a_input, const double & a_beta);
    /*! Returns the maximum element in a range and its global position index. */
    double max(int & a_index) const;
    /*! Returns the minimum element in a range and its global position index. */
    double min(int & a_index) const;
    /*! Computes the absolute value of each element in the container. */
    void modulus();
    /*! Returns the sum of all the elements in the container. */
    double sum() const;
    /*! Returns the inner product of two vectors. */
    double dot(const trrom::Vector<double> & a_input) const;
    /*! Returns the euclidean norm of a Vector. */
    double norm() const;
    /*! Assigns new contents to the Vector, replacing its current contents, and not modifying its size. */
    void fill(const double & a_input);
    /*! Returns the number of local elements in the Vector. */
    int size() const;
    /*! Creates a copy of type trrom::Vector */
    std::tr1::shared_ptr<trrom::Vector<double> > create(int a_length = 0) const;
    /*! Operator overloads the square bracket operator */
    double & operator [](int a_index);
    /*! Operator overloads the square bracket operator */
    const double & operator [](int a_index) const;

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

private:
    mxArray* m_Data;

private:
    MxVector(const trrom::MxVector &);
    trrom::MxVector & operator=(const trrom::MxVector & rhs_);
};

}

#endif /* TRROM_MXVECTOR_HPP_ */
