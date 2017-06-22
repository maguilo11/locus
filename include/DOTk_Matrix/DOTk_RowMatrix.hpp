/*
 * DOTk_RowMatrix.hpp
 *
 *  Created on: Jul 1, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_ROWMATRIX_HPP_
#define DOTK_ROWMATRIX_HPP_

#include "matrix.hpp"

namespace dotk
{

template<typename ScalarType>
class Vector;

namespace serial
{


template<typename ScalarType>
class DOTk_RowMatrix : public dotk::matrix<ScalarType>
{
public:
    DOTk_RowMatrix(const dotk::Vector<ScalarType> & row_, size_t storage_size_);
    virtual ~DOTk_RowMatrix();

    // Returns the number of rows in the matrix.
    virtual size_t nrows() const;
    // Returns the number of columns in the matrix.
    virtual size_t ncols() const;
    // Returns the number of elements contained in the matrix object.
    virtual size_t size() const;
    // Copies row/column into matrix row/column defined by index.
    virtual void copy(const size_t & index_,
                      const dotk::Vector<ScalarType> & input_,
                      bool row_major_copy_ = true);
    // Returns the dot product of two vectors. Index indicates row/column index.
    virtual ScalarType dot(const size_t & index_,
                     const dotk::Vector<ScalarType> & input_,
                     bool row_major_dot_product_ = true) const;
    // Returns the euclidean norm of a vector. Index indicates row/column index.
    virtual ScalarType norm(const size_t & index_, bool row_major_dot_product_ = true) const;
    // Scales a vector by a real constant. Index indicates row/column index.
    virtual void scale(const size_t & index_, const ScalarType & alpha_, bool row_major_dot_product_ = true);
    // Constant times a vector plus a vector. Index indicates row/column to modify.
    virtual void axpy(const size_t & index_,
                      const ScalarType & alpha_,
                      const dotk::Vector<ScalarType> & input_,
                      bool row_major_dot_ = true);
    // Returns the Frobenius norm of a matrix.
    virtual ScalarType norm() const;
    // Matrix-vector multiplication.
    virtual void matVec(const dotk::Vector<ScalarType> & input_,
                        dotk::Vector<ScalarType> & output_,
                        bool transpose_ = false) const;
    // General matrix-vector multiplication.
    virtual void gemv(const ScalarType & alpha_,
                      const dotk::Vector<ScalarType> & input_,
                      const ScalarType & beta_,
                      dotk::Vector<ScalarType> & output_,
                      bool transpose_ = false) const;
    // General matrix-matrix multiplication.
    virtual void gemm(const bool & transpose_A_,
                      const bool & transpose_B_,
                      const ScalarType & alpha_,
                      const dotk::matrix<ScalarType> & B_,
                      const ScalarType & beta_,
                      dotk::matrix<ScalarType> & C_) const;
    // Scales all the elements by a constant.
    virtual void scale(const ScalarType & alpha_);
    // Assigns new contents to the matrix, replacing its current contents, and not modifying its size.
    virtual void fill(const ScalarType & value_);
    // Copies the elements in the range [first,last) into the range beginning at result.
    virtual void copy(const dotk::matrix<ScalarType> & input_);
    // Copies the elements in the range [first,last) into the range beginning at result.
    virtual void copy(const size_t & num_inputs_, const ScalarType* input_);
    // Gathers together matrix values from a group of processes
    virtual void gather(const size_t & dim_, ScalarType* output_);
    // Sets matrix to identity
    virtual void identity();
    // Sets matrix element to input value
    virtual void set(const size_t & row_index_, const size_t & column_index_, ScalarType value_);
    // Returns the elements on the main diagonal.
    virtual void diag(dotk::Vector<ScalarType> & input_) const;
    // Sets elements on the main diagonal to input data.
    virtual void setDiag(const dotk::Vector<ScalarType> & input_, bool zero_matrix_elements_ = false);
    // Scales the elements on the main diagonal by a real constant.
    virtual void scaleDiag(const ScalarType & alpha_);
    // Returns the the sum of the elements on the main diagonal.
    virtual ScalarType trace() const;
    // Returns linearly independent spanning set dimension
    virtual size_t basisDimension() const;
    // Returns shared pointer to the first element of ith-row.
    virtual std::shared_ptr<dotk::Vector<ScalarType> > & basis(const size_t & index_);
    // Operator overloads the parenthesis operator
    virtual ScalarType & operator () (const size_t & row_index_, const size_t & column_index_);
    // Operator overloads the parenthesis operator
    virtual const ScalarType & operator () (const size_t & row_index_, const size_t & column_index_) const;
    // Clones memory for an object of type dotk::matrix
    virtual std::shared_ptr< dotk::matrix<ScalarType> > clone() const;
    // Returns dotk matrix type
    virtual dotk::types::matrix_t type() const;

private:
    void clear();
    void initialize(const dotk::Vector<ScalarType> & row_);

private:
    size_t m_Size;
    size_t m_NumRows;
    size_t m_NumColumns;
    std::shared_ptr<dotk::Vector<ScalarType> >* m_MatrixData;

private:
    DOTk_RowMatrix(const dotk::serial::DOTk_RowMatrix<ScalarType> &);
    dotk::serial::DOTk_RowMatrix<ScalarType> & operator=(const dotk::serial::DOTk_RowMatrix<ScalarType> & rhs_);
};

}

}

#endif /* DOTK_ROWMATRIX_HPP_ */
