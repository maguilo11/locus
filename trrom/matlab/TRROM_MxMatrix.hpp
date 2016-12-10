/*
 * TRROM_MxMatrix.hpp
 *
 *  Created on: Nov 21, 2016
 *      Author: maguilo
 */

#ifndef TRROM_MXMATRIX_HPP_
#define TRROM_MXMATRIX_HPP_

#include <mex.h>
#include "TRROM_Matrix.hpp"

namespace trrom
{

template<typename ScalarType>
class Vector;
class MxVector;

class MxMatrix : public trrom::Matrix<double>
{
public:
    //! @name Constructors/destructors
    //@{
    /*!
     * Creates a MxDenseMatrix object
     * Parameters:
     *    \param In
     *          num_rows_: number of rows
     *    \param In
     *          num_columns_: number of columns
     *    \param In
     *          initial_value_: if defined, fill vector with value. if not defined, fills the vector with zeros.
     *
     * \return Reference to MxDenseMatrix.
     *
     **/
    MxMatrix(int num_rows_, int num_columns_, double initial_value_ = 0);
    /*!
     * Creates a MxDenseMatrix object by making a deep copy of the input MEX array
     * Parameters:
     *    \param In
     *          array_: MEX array pointer
     *
     * \return Reference to MxDenseMatrix.
     *
     **/
    explicit MxMatrix(const mxArray* array_);
    //! MxDenseMatrix destructor.
    virtual ~MxMatrix();
    //@}

    //! Gets the number of rows.
    int getNumRows() const;
    //! Gets the number of columns.
    int getNumCols() const;
    //! Assigns new contents to the Matrix, replacing its current contents, and not modifying its size.
    void fill(double input_);
    //! Scales Matrix by a real constant.
    void scale(double input_);
    //! Update matrix values with scaled values of A, this = beta*this + alpha*A.
    void update(const double & alpha_, const trrom::Matrix<double> & input_, const double & beta_);
    /*!
     *
     * DGEMV  performs one of the matrix-vector operations
     *          y := alpha*A*x + beta*y,   or   y := alpha*A**T*x + beta*y,
     * where alpha and beta are scalars, x and y are vectors and A is an m by n matrix.
     *
     **/
    void gemv(bool transpose_,
              const double & alpha_,
              const trrom::Vector<double> & input_,
              const double & beta_,
              trrom::Vector<double> & output_) const;
    /*!
     *
     * GEMM  performs one of the matrix-matrix operations
     *          C := alpha*op( A )*op( B ) + beta*C,
     * where  op( X ) is one of op( X ) = X   or   op( X ) = X**T,
     * alpha and beta are scalars, and A, B and C are matrices, with op( A )
     * an m by k matrix,  op( B )  a  k by n matrix and  C an m by n matrix.
     *
     **/
    void gemm(const bool & transpose_A_,
              const bool & transpose_B_,
              const double & alpha_,
              const trrom::Matrix<double> & B_,
              const double & beta_,
              trrom::Matrix<double> & C_) const;
    void replaceGlobalValue(const int & global_row_index_, const int & global_column_index_, const double & value_);
    //! Operator overloads the parenthesis operator.
    double & operator ()(int my_row_index_, int my_column_index_);
    //! Operator overloads the constant parenthesis operator.
    const double & operator ()(int my_row_index_, int my_column_index_) const;
    //! Creates copy of this vector with user supplied dimensions */
    std::tr1::shared_ptr<trrom::Matrix<double> > create(int num_rows_ = 0, int num_cols_ = 0) const;

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

    // TODO: MAKE THIS FUNCTIONALITIES NON-VIRTUAL
    void insert(const trrom::Vector<double> & input_, int index_ = 0);
    const std::tr1::shared_ptr<trrom::Vector<double> > & vector(int index_) const;

private:
    //! Verifies input matrices' dimensions before performing general matrix-matrix multiplication.
    void verifyInputMatricesDimensions(const bool & transpose_A_,
                                       const bool & transpose_B_,
                                       const trrom::Matrix<double> & B_,
                                       const trrom::Matrix<double> & C_) const;

private:
    mxArray* m_Data;
    std::tr1::shared_ptr<trrom::Vector<double> > m_Vector;

private:
    MxMatrix(const trrom::MxMatrix &);
    trrom::MxMatrix & operator=(const trrom::MxMatrix & rhs_);
};

}

#endif /* TRROM_MXMATRIX_HPP_ */
