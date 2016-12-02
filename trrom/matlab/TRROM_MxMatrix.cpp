/*
 * TRROM_MxMatrix.cpp
 *
 *  Created on: Nov 28, 2016
 *      Author: maguilo
 */

#include <cmath>
#include <cassert>
#include <stddef.h>

#include <blas.h>

#include "TRROM_MxUtils.hpp"
#include "TRROM_MxVector.hpp"
#include "TRROM_MxMatrix.hpp"

namespace trrom
{

MxMatrix::MxMatrix(int num_rows_, int num_columns_, double initial_value_) :
        m_Data(mxCreateDoubleMatrix(num_rows_, num_columns_, mxREAL)),
        m_Vector()
{
    this->fill(initial_value_);
}

MxMatrix::MxMatrix(const mxArray* array_) :
        m_Data(mxDuplicateArray(array_)),
        m_Vector()
{
}

MxMatrix::~MxMatrix()
{
    mxDestroyArray(m_Data);
}

int MxMatrix::getNumRows() const
{
    return (mxGetM(m_Data));
}

int MxMatrix::getNumCols() const
{
    return (mxGetN(m_Data));
}

void MxMatrix::fill(double input_)
{
    double* my_data = mxGetPr(m_Data);
    int my_num_rows = this->getNumRows();
    int my_num_columns = this->getNumCols();
    for(int row = 0; row < my_num_rows; ++row)
    {
        for(int column = 0; column < my_num_columns; ++column)
        {
            int index = row + (my_num_rows * column);
            my_data[index] = input_;
        }
    }
}

void MxMatrix::scale(double input_)
{
    double* my_data = mxGetPr(m_Data);
    int my_num_rows = this->getNumRows();
    int my_num_columns = this->getNumCols();
    for(int row = 0; row < my_num_rows; ++row)
    {
        for(int column = 0; column < my_num_columns; ++column)
        {
            int index = row + (my_num_rows * column);
            my_data[index] = input_ * my_data[index];
        }
    }
}

void MxMatrix::update(const double & alpha_, const trrom::Matrix<double> & input_, const double & beta_)
{
    double* my_data = mxGetPr(m_Data);
    int my_num_rows = this->getNumRows();
    int my_num_columns = this->getNumCols();
    for(int row = 0; row < my_num_rows; ++row)
    {
        for(int column = 0; column < my_num_columns; ++column)
        {
            int index = row + (my_num_rows * column);
            my_data[index] = beta_ * my_data[index] + (alpha_ * input_(row, column));
        }
    }
}

void MxMatrix::gemv(bool transpose_,
                    const double & alpha_,
                    const trrom::Vector<double> & input_,
                    const double & beta_,
                    trrom::Vector<double> & output_) const
{
    if(transpose_ == false)
    {
        assert(input_.size() == this->getNumCols());
        assert(output_.size() == this->getNumRows());
    }
    else
    {
        assert(input_.size() == this->getNumRows());
        assert(output_.size() == this->getNumCols());
    }

    long int input_increment = 1;
    long int output_increment = 1;
    long int my_num_rows = this->getNumRows();
    long int my_num_columns = this->getNumCols();
    long int my_leading_dim = std::max(1, static_cast<int>(my_num_rows));

    std::vector<char> my_transpose = trrom::mx::transpose(transpose_);

    double* my_data = mxGetPr(m_Data);
    trrom::MxVector & output = dynamic_cast<trrom::MxVector &>(output_);
    const trrom::MxVector & input = dynamic_cast<const trrom::MxVector &>(input_);

    dgemv_(&my_transpose[0],
           &my_num_rows,
           &my_num_columns,
           &alpha_,
           my_data,
           &my_leading_dim,
           input.data(),
           &input_increment,
           &beta_,
           output.data(),
           &output_increment);
}

void MxMatrix::gemm(const bool & transpose_A_,
                    const bool & transpose_B_,
                    const double & alpha_,
                    const trrom::Matrix<double> & B_,
                    const double & beta_,
                    trrom::Matrix<double> & C_) const
{
    this->verifyInputMatricesDimensions(transpose_A_, transpose_B_, B_, C_);

    // **** Compute leading dimensions for each matrix (A, B, and C) ****
    long int dimension_m = -1;
    long int dimension_k = -1;
    long int my_leading_dim = -1;
    long int my_num_rows = this->getNumRows();
    long int my_num_columns = this->getNumCols();
    // Leading dimension for matrix A
    if(transpose_A_ == false)
    {
        // normal operation, i.e not transpose
        dimension_m = my_num_rows;
        dimension_k = my_num_columns;
        my_leading_dim = std::max(1, static_cast<int>(dimension_m));
    }
    else
    {
        // transpose operation
        dimension_k = my_num_rows;
        dimension_m = my_num_columns;
        my_leading_dim = std::max(1, static_cast<int>(dimension_k));
    }
    // Leading dimension for matrix B
    long int dimension_n = -1;
    long int B_leading_dim = -1;
    if(transpose_B_ == false)
    {
        // normal operation, i.e not transpose
        dimension_n = B_.getNumCols();
        B_leading_dim = std::max(1, static_cast<int>(dimension_k));
    }
    else
    {
        // transpose operation
        dimension_n = B_.getNumRows();
        B_leading_dim = std::max(1, static_cast<int>(dimension_n));
    }
    // Leading dimension for matrix C
    long int C_leading_dim = std::max(1, static_cast<int>(dimension_m));

    // **** Perform general matrix-matrix multiplication ****
    double* my_data = mxGetPr(m_Data);
    trrom::MxMatrix & C_matrix = dynamic_cast<trrom::MxMatrix &>(C_);
    const trrom::MxMatrix & B_matrix = dynamic_cast<const trrom::MxMatrix &>(B_);
    std::vector<char> transpose_B = trrom::mx::transpose(transpose_B_);
    std::vector<char> my_transpose = trrom::mx::transpose(transpose_A_);
    dgemm_(&my_transpose[0],
           &transpose_B[0],
           &dimension_m,
           &dimension_n,
           &dimension_k,
           &alpha_,
           my_data,
           &my_leading_dim,
           B_matrix.data(),
           &B_leading_dim,
           &beta_,
           C_matrix.data(),
           &C_leading_dim);
}

void MxMatrix::replaceGlobalValue(const int & global_row_index_,
                                  const int & global_column_index_,
                                  const double & value_)
{
    double* my_data = mxGetPr(m_Data);
    int my_num_rows = this->getNumRows();
    int index = global_row_index_ + (my_num_rows * global_column_index_);
    my_data[index] = value_;
}

double & MxMatrix::operator ()(int my_row_index_, int my_column_index_)
{
    double* my_data = mxGetPr(m_Data);
    int my_num_rows = this->getNumRows();
    int index = my_row_index_ + (my_num_rows * my_column_index_);
    return (my_data[index]);
}

const double & MxMatrix::operator ()(int my_row_index_, int my_column_index_) const
{
    double* my_data = mxGetPr(m_Data);
    int my_num_rows = this->getNumRows();
    int index = my_row_index_ + (my_num_rows * my_column_index_);
    return (my_data[index]);
}

std::tr1::shared_ptr<trrom::Matrix<double> > MxMatrix::create(int num_rows_, int num_cols_) const
{
    assert(num_cols_ >= 0);
    assert(num_rows_ >= 0);
    std::tr1::shared_ptr<trrom::MxMatrix> this_copy;
    if(num_cols_ > 0 && num_rows_ > 0)
    {
        this_copy.reset(new trrom::MxMatrix(num_rows_, num_cols_));
    }
    else
    {
        const int my_num_rows = this->getNumRows();
        const int my_num_columns = this->getNumCols();
        this_copy.reset(new trrom::MxMatrix(my_num_rows, my_num_columns));
    }
    return (this_copy);
}

double* MxMatrix::data()
{
    return (mxGetPr(m_Data));
}

const double* MxMatrix::data() const
{
    return (mxGetPr(m_Data));
}

mxArray* MxMatrix::array()
{
    return (m_Data);
}

const mxArray* MxMatrix::array() const
{
    return (m_Data);
}

void MxMatrix::setMxArray(const mxArray* input_)
{
    const int input_num_rows = mxGetM(input_);
    assert(input_num_rows == this->getNumRows());
    const int input_num_columns = mxGetN(input_);
    assert(input_num_columns == this->getNumCols());

    double* my_data = this->data();
    const double* input_data = mxGetPr(input_);
    for(int row = 0; row < input_num_rows; ++row)
    {
        for(int column = 0; column < input_num_columns; ++column)
        {
            int index = row + (input_num_columns * column);
            my_data[index] = input_data[index];
        }
    }
}

trrom::Vector<double> & MxMatrix::vector(int index_) const
{
    return (*m_Vector);
}

void MxMatrix::insert(const trrom::Vector<double> & input_)
{
}

void MxMatrix::verifyInputMatricesDimensions(const bool & transpose_A_,
                                             const bool & transpose_B_,
                                             const trrom::Matrix<double> & B_,
                                             const trrom::Matrix<double> & C_) const
{
    if(transpose_A_ == false && transpose_B_ == false)
    {
        assert(B_.getNumCols() == C_.getNumCols());
        assert(B_.getNumRows() == this->getNumCols());
        assert(C_.getNumRows() == this->getNumRows());
    }
    else if(transpose_A_ == true && transpose_B_ == false)
    {
        assert(B_.getNumCols() == C_.getNumCols());
        assert(B_.getNumRows() == this->getNumRows());
        assert(C_.getNumRows() == this->getNumCols());
    }
    else if(transpose_A_ == false && transpose_B_ == true)
    {
        assert(B_.getNumRows() == C_.getNumCols());
        assert(B_.getNumCols() == this->getNumCols());
        assert(C_.getNumRows() == this->getNumRows());
    }
    else
    {
        assert(B_.getNumRows() == C_.getNumCols());
        assert(B_.getNumCols() == this->getNumRows());
        assert(C_.getNumRows() == this->getNumCols());
    }
}

}
