/*
 * matrix.hpp
 *
 *  Created on: Jul 1, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef MATRIX_HPP_
#define MATRIX_HPP_

#include <limits>
#include <iostream>
#include <tr1/memory>

#include "DOTk_MatrixTypes.hpp"

namespace dotk
{

template<typename ScalarType>
class Vector;

template<typename ScalarType>
class matrix
{
public:
    matrix()
    {
    }
    virtual ~matrix()
    {
    }
    // Returns the number of rows in the matrix.
    virtual size_t nrows() const = 0;
    // Returns the number of columns in the matrix.
    virtual size_t ncols() const = 0;
    // Returns the number of elements contained in the matrix object.
    virtual size_t size() const = 0;
    // Copies row/column into matrix row/column defined by index.
    virtual void copy(const size_t & index_,
                      const dotk::Vector<ScalarType> & input_,
                      bool copy_this_1d_container_ = true) = 0;
    // Returns the euclidean norm of a vector. Index indicates row/column index.
    virtual ScalarType norm(const size_t & index_, bool apply_norm_to_this_1d_container_ = true) const = 0;
    // Scales a vector by a real constant. Index indicates row/column index.
    virtual void scale(const size_t & index_, const ScalarType & alpha_, bool scale_this_1d_container_ = true) = 0;
    // Constant times a vector plus a vector. Index indicates row/column index.
    virtual void axpy(const size_t & index_,
                      const ScalarType & value_,
                      const dotk::Vector<ScalarType> & input_,
                      bool apply_axpy_to_this_1d_container_ = true) = 0;
    // Returns the dot product of two vectors. Index indicates row/column index.
    virtual ScalarType dot(const size_t & index_,
                     const dotk::Vector<ScalarType> & input_,
                     bool apply_dot_to_this_1d_container_ = true) const = 0;
    // Returns the Frobenius norm of a matrix.
    virtual ScalarType norm() const = 0;
    // Matrix-vector multiplication.
    virtual void matVec(const dotk::Vector<ScalarType> & input_,
                        dotk::Vector<ScalarType> & output,
                        bool transpose_ = false) const = 0;
    // General matrix-vector multiplication.
    virtual void gemv(const ScalarType & alpha_,
                      const dotk::Vector<ScalarType> & input_,
                      const ScalarType & beta_,
                      dotk::Vector<ScalarType> & output_,
                      bool transpose_ = false) const = 0;
    // General matrix-matrix multiplication.
    virtual void gemm(const bool & transpose_A_,
                      const bool & transpose_B_,
                      const ScalarType & alpha_,
                      const dotk::matrix<ScalarType> & B_,
                      const ScalarType & beta_,
                      dotk::matrix<ScalarType> & C_) const = 0;
    // Scales all the elements by a constant.
    virtual void scale(const ScalarType & alpha_) = 0;
    // Assigns new contents to the matrix, replacing its current contents, and not modifying its size.
    virtual void fill(const ScalarType & value_) = 0;
    // Copies the elements in the range [first,last) into the range beginning at result.
    virtual void copy(const dotk::matrix<ScalarType> & input_) = 0;
    // Copies the elements in the range [first,last) into the range beginning at result.
    virtual void copy(const size_t & num_inputs_, const ScalarType* input_) = 0;
    // Gathers together matrix values from a group of processes
    virtual void gather(const size_t & dim_, ScalarType* output_) = 0;
    // Sets matrix to identity
    virtual void identity() = 0;
    // Sets matrix element to input value
    virtual void set(const size_t & row_index_, const size_t & column_index_, ScalarType value_) = 0;
    // Returns dotk matrix type
    virtual dotk::types::matrix_t type() const = 0;
    // Clones memory for an object of type dotk::matrix
    virtual std::tr1::shared_ptr< dotk::matrix<ScalarType> > clone() const = 0;
    // Operator overloads the parenthesis operator
    virtual ScalarType & operator () (const size_t & row_index_, const size_t & column_index_) = 0;
    // Operator overloads the parenthesis operator
    virtual const ScalarType & operator () (const size_t & row_index_, const size_t & column_index_) const = 0;
    // Returns linearly independent spanning set dimension
    virtual size_t basisDimension() const
    {
        std::string msg(" CALLING UNIMPLEMENTED dotk::matrix::basisDimension **** ");
        std::cerr << " **** ERROR MESSAGE: " << msg.c_str() << std::flush;
        return (0.);
    }
    // Returns a basis vector from the linearly independent spanning set
    virtual std::tr1::shared_ptr<dotk::Vector<ScalarType> > & basis(const size_t & index_)
    {
        std::string msg(" CALLING UNIMPLEMENTED dotk::matrix::basis **** ");
        std::cerr << " **** ERROR MESSAGE: " << msg.c_str() << std::flush;
        abort();
    }
    // Shifts matrix diagonal elements by a real constant.
    virtual void shift(const ScalarType & alpha_)
    {
        std::string msg(" CALLING UNIMPLEMENTED dotk::matrix::shift **** ");
        std::cerr << " **** ERROR MESSAGE: " << msg.c_str() << std::flush;
        abort();
    }
    // Returns the elements on the main diagonal.
    virtual void diag(dotk::Vector<ScalarType> & input_) const
    {
        std::string msg(" CALLING UNIMPLEMENTED dotk::matrix::diag **** ");
        std::cerr << " **** ERROR MESSAGE: " << msg.c_str() << std::flush;
        abort();
    }
    // Sets elements on the main diagonal to input data.
    virtual void setDiag(const dotk::Vector<ScalarType> & input_, bool zero_matrix_elements_ = false)
    {
        std::string msg(" CALLING UNIMPLEMENTED dotk::matrix::setDiag **** ");
        std::cerr << " **** ERROR MESSAGE: " << msg.c_str() << std::flush;
        abort();
    }
    // Scales the elements on the main diagonal by a real constant.
    virtual void scaleDiag(const ScalarType & alpha_)
    {
        std::string msg(" CALLING UNIMPLEMENTED dotk::matrix::scaleDiag **** ");
        std::cerr << " **** ERROR MESSAGE: " << msg.c_str() << std::flush;
        abort();
    }
    // Returns the the sum of the elements on the main diagonal.
    virtual ScalarType trace() const
    {
        std::string msg(" CALLING UNIMPLEMENTED dotk::matrix::trace **** ");
        std::cerr << " **** ERROR MESSAGE: " << msg.c_str() << std::flush;
        abort();
        return (std::numeric_limits<ScalarType>::quiet_NaN());
    }

private:
    matrix(const dotk::matrix<ScalarType> &);
    dotk::matrix<ScalarType> & operator=(const dotk::matrix<ScalarType> & rhs_);
};

}

#endif /* MATRIX_HPP_ */
