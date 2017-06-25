/*
 * TRROM_TeuchosSerialDenseMatrix.hpp
 *
 *  Created on: Oct 19, 2016
 *      Author: maguilo
 */

#ifndef TRROM_TEUCHOSSERIALDENSEMATRIX_HPP_
#define TRROM_TEUCHOSSERIALDENSEMATRIX_HPP_

#include "TRROM_Matrix.hpp"
#include "TRROM_SerialArray.hpp"
#include "TRROM_TeuchosArray.hpp"

#include "Teuchos_SerialDenseMatrix.hpp"

namespace trrom
{

class TeuchosSerialDenseMatrix : public trrom::Matrix<double>
{
public:
    TeuchosSerialDenseMatrix() :
            mLength(1),
            mData(std::make_shared<Teuchos::SerialDenseMatrix<int, double>>()),
            mVector(std::make_shared<trrom::SerialArray<double>>(mLength))
    {
    }
    TeuchosSerialDenseMatrix(int num_rows_, int num_columns_) :
            mLength(1),
            mData(std::make_shared<Teuchos::SerialDenseMatrix<int, double>>(num_rows_, num_columns_)),
            mVector(std::make_shared<trrom::SerialArray<double>>(mLength))
    {
    }
    virtual ~TeuchosSerialDenseMatrix()
    {
    }

    void fill(double value_)
    {
        mData->putScalar(value_);
    }
    void scale(double value_)
    {
        mData->scale(value_);
    }
    void insert(const trrom::Vector<double> & input_, int index_ = 0)
    {
        // TODO: RECONSIDER PURE VIRTUAL FUNCTION
    }

    void update(const double & alpha_, const trrom::Matrix<double> & input_, const double & beta_)
    {
        int num_rows = input_.getNumRows();
        int num_columns = input_.getNumCols();
        assert(num_rows == this->getNumRows());
        assert(num_columns == this->getNumCols());

        for(int row_index = 0; row_index < num_rows; ++row_index)
        {
            for(int column_index = 0; column_index < num_columns; ++column_index)
            {
                (*mData)(row_index, column_index) =
                        beta_ * (*mData)(row_index, column_index) + alpha_ * input_(row_index, column_index);
            }
        }
    }
    void gemv(bool transpose_,
              const double & alpha_,
              const trrom::Vector<double> & input_,
              const double & beta_,
              trrom::Vector<double> & output_) const
    {
        int num_rows = this->getNumRows();
        int num_columns = this->getNumCols();
        int matrix_leading_dim = std::max(1, num_rows);
        trrom::TeuchosArray<double> input(num_columns);
        input.update(1., input_, 0.);
        int increments_for_in_vector = 1;
        trrom::TeuchosArray<double> output(num_rows);
        int increments_for_out_vector = 1;

        Teuchos::ETransp transpose = this->castTranspose(transpose_);

        mData->GEMV(transpose,
                     num_rows,
                     num_columns,
                     alpha_,
                     mData->values(),
                     matrix_leading_dim,
                     input.data()->getRawPtr(),
                     increments_for_in_vector,
                     beta_,
                     output.data()->getRawPtr(),
                     increments_for_out_vector);

        assert(output_.size() == output.size());
        output_.update(1., output, 0.);
    }
    void gemm(const bool & transpose_A_,
              const bool & transpose_B_,
              const double & alpha_,
              const trrom::Matrix<double> & B_,
              const double & beta_,
              trrom::Matrix<double> & C_) const
    {
        int num_rows = this->getNumRows();
        int num_columns = this->getNumCols();
        int matrix_A_leading_dim = std::max(1, num_rows);
        int matrix_B_leading_dim = num_columns;
        int matrix_C_leading_dim = std::max(1, num_rows);

        trrom::TeuchosSerialDenseMatrix B_matrix(B_.getNumRows(), B_.getNumCols());
        B_matrix.update(1., B_, 0.);
        trrom::TeuchosSerialDenseMatrix C_matrix(C_.getNumRows(), C_.getNumCols());
        C_matrix.update(1., C_, 0.);

        Teuchos::ETransp transpose_A = this->castTranspose(transpose_A_);
        Teuchos::ETransp transpose_B = this->castTranspose(transpose_B_);

        mData->GEMM(transpose_A,
                     transpose_B,
                     num_rows,
                     num_columns,
                     num_columns,
                     alpha_,
                     mData->values(),
                     matrix_A_leading_dim,
                     B_matrix.data()->values(),
                     matrix_B_leading_dim,
                     beta_,
                     C_matrix.data()->values(),
                     matrix_C_leading_dim);

        C_.update(1., C_matrix, 0.);
    }

    int getNumRows() const
    {
        int number_of_rows = mData->numRows();
        return (number_of_rows);
    }
    int getNumCols() const
    {
        int number_of_columns = mData->numCols();
        return (number_of_columns);
    }
    double & operator ()(int global_row_index_, int global_column_index_)
    {
        return (mData->operator ()(global_row_index_, global_column_index_));
    }
    const double & operator ()(int global_row_index_, int global_column_index_) const
    {
        return (mData->operator ()(global_row_index_, global_column_index_));
    }
    void replaceGlobalValue(const int & global_row_index_, const int & global_column_index_, const double & value_)
    {
        (*mData)(global_row_index_, global_column_index_) = value_;
    }

    const std::shared_ptr<trrom::Vector<double> > & vector(int index_) const
    {
        // TODO: RECONSIDER PURE VIRTUAL FUNCTION
        return (mVector);
    }
    std::shared_ptr<trrom::Matrix<double> > create(int nrows_ = 0, int ncols_ = 0) const
    {
        assert(nrows_ >= 0);
        assert(ncols_ >= 0);
        std::shared_ptr<trrom::TeuchosSerialDenseMatrix> this_copy;
        if((nrows_ > 0) && (ncols_ > 0))
        {
            this_copy = std::make_shared<trrom::TeuchosSerialDenseMatrix>(nrows_, ncols_);
        }
        else
        {
            int num_rows = this->getNumRows();
            int num_cols = this->getNumCols();
            this_copy = std::make_shared<trrom::TeuchosSerialDenseMatrix>(num_rows, num_cols);
        }
        return (this_copy);
    }
    std::shared_ptr<Teuchos::SerialDenseMatrix<int, double> > & data()
    {
        return (mData);
    }

private:
    Teuchos::ETransp castTranspose(bool transpose_) const
    {
        Teuchos::ETransp transpose;
        if(transpose_ == true)
        {
            transpose = Teuchos::ETransp::TRANS;
        }
        else
        {

            transpose = Teuchos::ETransp::NO_TRANS;
        }
        return (transpose);
    }

private:
    int mLength;
    std::shared_ptr< Teuchos::SerialDenseMatrix<int, double> > mData;
    std::shared_ptr< trrom::Vector<double> > mVector;

private:
    TeuchosSerialDenseMatrix(const trrom::TeuchosSerialDenseMatrix &);
    trrom::TeuchosSerialDenseMatrix & operator=(const trrom::TeuchosSerialDenseMatrix & rhs_);
};

}

#endif /* TRROM_TEUCHOSSERIALDENSEMATRIX_HPP_ */
