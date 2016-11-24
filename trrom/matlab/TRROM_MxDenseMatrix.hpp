/*
 * TRROM_MxDenseMatrix.hpp
 *
 *  Created on: Nov 21, 2016
 *      Author: maguilo
 */

#ifndef TRROM_MXDENSEMATRIX_HPP_
#define TRROM_MXDENSEMATRIX_HPP_

#include <mex.h>

#include "TRROM_Matrix.hpp"
#include "TRROM_MxUtils.hpp"
#include "TRROM_MxVector.hpp"
#include "/usr/local/matlab/8.6b/extern/include/blas.h"

namespace trrom
{

class MxDenseMatrix : public trrom::Matrix<double>
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
    MxDenseMatrix(int num_rows_, int num_columns_, double initial_value_ = 0) :
            m_Data(mxCreateDoubleMatrix(num_rows_, num_columns_, mxREAL)),
            m_Vector()
    {
        this->fill(initial_value_);
    }
    /*!
     * Creates a MxDenseMatrix object by making a deep copy of the input MEX array
     * Parameters:
     *    \param In
     *          array_: MEX array
     *
     * \return Reference to MxDenseMatrix.
     *
     **/
    explicit MxDenseMatrix(mxArray* array_) :
            m_Data(mxDuplicateArray(array_)),
            m_Vector()
    {
    }
    //! MxDenseMatrix destructor.
    virtual ~MxDenseMatrix()
    {
        mxDestroyArray(m_Data);
    }
    //@}

    //! Gets the number of rows.
    int getNumRows() const
    {
        return (mxGetM(m_Data));
    }
    //! Gets the number of columns.
    int getNumCols() const
    {
        return (mxGetN(m_Data));
    }
    void fill(double input_)
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
    void scale(double input_)
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
    void update(const double & alpha_, const trrom::Matrix<double> & input_, const double & beta_)
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
              trrom::Matrix<double> & C_) const
    {
        this->verifyInputsDimensions(transpose_A_, transpose_B_, B_, C_);

        long int B_num_columns = B_.getNumCols();
        long int my_num_rows = this->getNumRows();
        long int my_num_columns = this->getNumCols();

        long int my_leading_dim = 0;
        if(transpose_A_ == false)
        {
            my_leading_dim = std::max(1, static_cast<int>(my_num_rows));
        }
        else
        {
            my_leading_dim = std::max(1, static_cast<int>(my_num_columns));
        }

        long int B_leading_dim = 0;
        if(transpose_B_ == false)
        {
            B_leading_dim = std::max(1, static_cast<int>(my_num_columns));
        }
        else
        {
            B_leading_dim = std::max(1, static_cast<int>(B_num_columns));
        }

        long int C_leading_dim = std::max(1, static_cast<int>(my_leading_dim));

        double* my_data = mxGetPr(m_Data);
        trrom::MxDenseMatrix & C_matrix = dynamic_cast<trrom::MxDenseMatrix &>(C_);
        const trrom::MxDenseMatrix & B_matrix = dynamic_cast<const trrom::MxDenseMatrix &>(B_);

        std::vector<char> transpose_B = trrom::mx::transpose(transpose_B_);
        std::vector<char> my_transpose = trrom::mx::transpose(transpose_A_);

        dgemm_(&my_transpose[0],
               &transpose_B[0],
               &my_num_rows,
               &B_num_columns,
               &my_num_columns,
               &alpha_,
               my_data,
               &my_leading_dim,
               B_matrix.data(),
               &B_leading_dim,
               &beta_,
               C_matrix.data(),
               &C_leading_dim);
    }
    void replaceGlobalValue(const int & global_row_index_, const int & global_column_index_, const double & value_)
    {
        double* my_data = mxGetPr(m_Data);
        int my_num_rows = this->getNumRows();
        int index = global_row_index_ + (my_num_rows * global_column_index_);
        my_data[index] = value_;
    }
    double & operator ()(int my_row_index_, int my_column_index_)
    {
        double* my_data = mxGetPr(m_Data);
        int my_num_rows = this->getNumRows();
        int index = my_row_index_ + (my_num_rows * my_column_index_);
        return (my_data[index]);
    }
    const double & operator ()(int my_row_index_, int my_column_index_) const
    {
        double* my_data = mxGetPr(m_Data);
        int my_num_rows = this->getNumRows();
        int index = my_row_index_ + (my_num_rows * my_column_index_);
        return (my_data[index]);
    }
    //! Creates copy of this vector with user supplied dimensions */
    std::tr1::shared_ptr<trrom::Matrix<double> > create(int num_rows_ = 0, int num_cols_ = 0) const
    {
        assert(num_cols_ >= 0);
        assert(num_rows_ >= 0);
        std::tr1::shared_ptr<trrom::MxDenseMatrix> this_copy;
        if(num_cols_ > 0 && num_rows_ > 0)
        {
            this_copy.reset(new trrom::MxDenseMatrix(num_rows_, num_cols_));
        }
        else
        {
            const int my_num_rows = this->getNumRows();
            const int my_num_columns = this->getNumCols();
            this_copy.reset(new trrom::MxDenseMatrix(my_num_rows, my_num_columns));
        }
        return (this_copy);
    }

    trrom::Vector<double> & vector(int index_) const
    {
        return (*m_Vector);
    }
    void insert(const trrom::Vector<double> & input_)
    {
    }

    //! Get non-constant real numeric pointer for numeric array.
    double* data()
    {
        return (mxGetPr(m_Data));
    }
    //! Get constant real numeric pointer for numeric array.
    const double* data() const
    {
        return (mxGetPr(m_Data));
    }

private:
    void verifyInputsDimensions(const bool & transpose_A_,
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

private:
    mxArray* m_Data;
    std::tr1::shared_ptr<trrom::MxVector> m_Vector;

private:
    MxDenseMatrix(const trrom::MxDenseMatrix &);
    trrom::MxDenseMatrix & operator=(const trrom::MxDenseMatrix & rhs_);
};

}

#endif /* TRROM_MXDENSEMATRIX_HPP_ */
