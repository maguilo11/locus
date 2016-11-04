/*
 * DOTk_UpperTriangularMatrix.hpp
 *
 *  Created on: Jul 2, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_UPPERTRIANGULARMATRIX_HPP_
#define DOTK_UPPERTRIANGULARMATRIX_HPP_

#include "matrix.hpp"

namespace dotk
{

namespace serial
{

template<class Type>
class vector;

template<class Type>
class DOTk_UpperTriangularMatrix : public dotk::matrix<Type>
{
public:
    DOTk_UpperTriangularMatrix(size_t num_rows_, Type value_ = 0.);
    virtual ~DOTk_UpperTriangularMatrix();

    // Returns the number of rows in the matrix.
    virtual size_t nrows() const;
    // Returns the number of columns in the matrix.
    virtual size_t ncols() const;
    // Returns the number of elements contained in the matrix object.
    virtual size_t size() const;
    // Copies row/column into matrix row/column defined by index.
    virtual void copy(const size_t & index_,
                      const dotk::vector<Type> & input_,
                      bool row_major_copy_ = true);
    // Returns the euclidean norm of a vector. Index indicates row/column to compute norm.
    virtual Type norm(const size_t & index_, bool row_major_norm_ = true) const;
    // Scales a vector by a real constant. Index indicates row/column to compute norm.
    virtual void scale(const size_t & index_, const Type & alpha_, bool row_major_scale_ = true);
    // Constant times a vector plus a vector. Index indicates row/column to modify.
    virtual void axpy(const size_t & index_,
                      const Type & alpha_,
                      const dotk::vector<Type> & input_,
                      bool row_major_axpy_ = true);
    // Returns the dot product of two vectors.
    virtual Type dot(const size_t & index_,
                     const dotk::vector<Type> & input_,
                     bool row_major_dot_ = true) const;
    // Returns the Frobenius norm of a matrix.
    virtual Type norm() const;
    // Matrix-vector multiplication.
    virtual void matVec(const dotk::vector<Type> & input_,
                        dotk::vector<Type> & output_,
                        bool transpose_ = false) const;
    // General matrix-vector multiplication.
    virtual void gemv(const Type & alpha_,
                      const dotk::vector<Type> & input_,
                      const Type & beta_,
                      dotk::vector<Type> & output_,
                      bool transpose_ = false) const;
    // General matrix-matrix multiplication.
    virtual void gemm(const bool & transpose_A_,
                      const bool & transpose_B_,
                      const Type & alpha_,
                      const dotk::matrix<Type> & B_,
                      const Type & beta_,
                      dotk::matrix<Type> & C_) const;
    // Scales all the elements by a constant.
    virtual void scale(const Type & alpha_);
    // Assigns new contents to the matrix, replacing its current contents, and not modifying its size.
    virtual void fill(const Type & value_);
    // Copies the elements in the range [first,last) into the range beginning at result.
    virtual void copy(const dotk::matrix<Type> & input_);
    // Copies the elements in the range [first,last) into the range beginning at result.
    virtual void copy(const size_t & num_inputs_, const Type* input_);
    // Gathers together matrix values from a group of processes
    virtual void gather(const size_t & dim_, Type* output_);
    // Sets matrix to identity
    virtual void identity();
    // Shifts matrix diagonal elements by a real constant.
    virtual void shift(const Type & alpha_);
    // Returns the elements on the main diagonal.
    virtual void diag(dotk::vector<Type> & input_) const;
    // Sets elements on the main diagonal to input data.
    virtual void setDiag(const dotk::vector<Type> & input_, bool zero_matrix_elements_ = false);
    // Scales the elements on the main diagonal by a real constant.
    virtual void scaleDiag(const Type & alpha_);
    // Returns the the sum of the elements on the main diagonal.
    virtual Type trace() const;
    // Sets matrix element to input value
    virtual void set(const size_t & row_index_, const size_t & column_index_, Type value_);
    // Operator overloads the parenthesis operator
    virtual Type & operator () (const size_t & row_index_, const size_t & column_index_);
    // Operator overloads the parenthesis operator
    virtual const Type & operator () (const size_t & row_index_, const size_t & column_index_) const;
    // Clones memory for an object of type dotk::matrix
    virtual std::tr1::shared_ptr< dotk::matrix<Type> > clone() const;
    // Returns dotk matrix type
    virtual dotk::types::matrix_t type() const;

private:
    void initialize();
    void clear();

private:
    Type m_Zero;
    Type* m_Data;
    size_t m_Size;
    size_t m_NumRows;
    size_t m_NumCols;
    size_t* m_Displacements;
    size_t* m_NumColumnCount;

private:
    DOTk_UpperTriangularMatrix(const dotk::serial::DOTk_UpperTriangularMatrix<Type> &);
    dotk::serial::DOTk_UpperTriangularMatrix<Type> & operator=(const dotk::serial::DOTk_UpperTriangularMatrix<Type> & rhs_);
};

}

}

#endif /* DOTK_UPPERTRIANGULARMATRIX_HPP_ */
