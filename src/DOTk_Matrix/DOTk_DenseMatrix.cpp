/*
 * DOTk_DenseMatrix.cpp
 *
 *  Created on: Jul 14, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#include <cmath>
#include <cassert>

#include "vector.hpp"
#include "DOTk_DenseMatrix.hpp"

namespace dotk
{

namespace serial
{

template<class Type>
DOTk_DenseMatrix<Type>::DOTk_DenseMatrix(size_t nrows_, Type value_) :
        dotk::matrix<Type>(),
        m_MatrixData(new Type[nrows_ * nrows_]), m_Size(nrows_ * nrows_),
        m_NumRows(nrows_),
        m_NumCols(nrows_)
        {
            this->fill(value_);
        }

template<class Type>
DOTk_DenseMatrix<Type>::~DOTk_DenseMatrix()
{
    this->clear();
}

template<class Type>
size_t DOTk_DenseMatrix<Type>::nrows() const
{
    return (m_NumRows);
}

template<class Type>
size_t DOTk_DenseMatrix<Type>::ncols() const
{
    return (m_NumCols);
}

template<class Type>
size_t DOTk_DenseMatrix<Type>::size() const
{
    return (m_Size);
}

template<class Type>
void DOTk_DenseMatrix<Type>::copy(const size_t & index_, const dotk::vector<Type> & input_, bool row_major_copy_)
{
    if(row_major_copy_ == false)
    {
        assert(input_.size() == this->nrows());

        size_t index, row;
        size_t num_columns = this->ncols();

        for(row = 0; row < this->nrows(); ++row)
        {
            index = (num_columns * row) + index_;
            m_MatrixData[index] = input_[row];
        }
    }
    else
    {
        assert(input_.size() == this->ncols());

        size_t index, column;
        size_t num_columns = this->ncols();

        for(column = 0; column < num_columns; ++column)
        {
            index = (num_columns * index_) + column;
            m_MatrixData[index] = input_[column];
        }
    }
}

template<class Type>
Type DOTk_DenseMatrix<Type>::norm(const size_t & index_, bool row_major_norm_) const
{
    Type value = 0.;

    if(row_major_norm_ == false)
    {
        size_t index, row;
        size_t num_columns = this->ncols();

        for(row = 0; row < this->nrows(); ++row)
        {
            index = (num_columns * row) + index_;
            value += m_MatrixData[index] * m_MatrixData[index];
        }
    }
    else
    {
        size_t index, column;
        size_t num_columns = this->ncols();

        for(column = 0; column < num_columns; ++column)
        {
            index = (num_columns * index_) + column;
            value += m_MatrixData[index] * m_MatrixData[index];
        }
    }

    value = std::pow(value, 0.5);

    return (value);
}

template<class Type>
void DOTk_DenseMatrix<Type>::scale(const size_t & index_, const Type & alpha_, bool row_major_scale_)
{
    if(row_major_scale_ == false)
    {
        size_t index, row;
        size_t num_columns = this->ncols();

        for(row = 0; row < this->nrows(); ++row)
        {
            index = (num_columns * row) + index_;
            m_MatrixData[index] = alpha_ * m_MatrixData[index];
        }
    }
    else
    {
        size_t index, column;
        size_t num_columns = this->ncols();

        for(column = 0; column < num_columns; ++column)
        {
            index = (num_columns * index_) + column;
            m_MatrixData[index] = alpha_ * m_MatrixData[index];
        }
    }
}

template<class Type>
void DOTk_DenseMatrix<Type>::axpy(const size_t & index_,
                                  const Type & alpha_,
                                  const dotk::vector<Type> & input_,
                                  bool row_major_axpy_)
{
    if(row_major_axpy_ == false)
    {
        assert(input_.size() == this->nrows());

        size_t index, row;
        size_t num_columns = this->ncols();

        for(row = 0; row < this->nrows(); ++row)
        {
            index = (num_columns * row) + index_;
            m_MatrixData[index] = alpha_ * input_[row] + m_MatrixData[index];
        }
    }
    else
    {
        assert(input_.size() == this->ncols());

        size_t index, column;
        size_t num_columns = this->ncols();

        for(column = 0; column < num_columns; ++column)
        {
            index = (num_columns * index_) + column;
            m_MatrixData[index] = alpha_ * input_[column] + m_MatrixData[index];
        }
    }
}

template<class Type>
Type DOTk_DenseMatrix<Type>::dot(const size_t & index_,
                                 const dotk::vector<Type> & input_,
                                 bool row_major_dot_) const
{
    Type value = 0.;

    if(row_major_dot_ == false)
    {
        assert(input_.size() == this->nrows());

        size_t index, row;
        size_t num_columns = this->ncols();

        for(row = 0; row < this->nrows(); ++row)
        {
            index = (num_columns * row) + index_;
            value += m_MatrixData[index] * input_[row];
        }
    }
    else
    {
        assert(input_.size() == this->ncols());

        size_t index, column;
        size_t num_columns = this->ncols();

        for(column = 0; column < num_columns; ++column)
        {
            index = (num_columns * index_) + column;
            value += m_MatrixData[index] * input_[column];
        }
    }

    return (value);
}

template<class Type>
Type DOTk_DenseMatrix<Type>::norm() const
{
    Type value = 0.;

    for(size_t index = 0; index < this->size(); ++index)
    {
        value += m_MatrixData[index] * m_MatrixData[index];
    }

    value = std::pow(value, 0.5);

    return (value);
}

template<class Type>
Type DOTk_DenseMatrix<Type>::trace() const
{
    Type value = 0.;
    const size_t nrows = this->nrows();
    const size_t ncols = this->ncols();

    for(size_t row = 0; row < nrows; ++row)
    {
        const size_t index = (ncols * row) + row;
        value += m_MatrixData[index];
    }

    return (value);
}

template<class Type>
void DOTk_DenseMatrix<Type>::matVec(const dotk::vector<Type> & input_, dotk::vector<Type> & output_, bool transpose_) const
{
    output_.fill(0);
    const size_t num_rows = this->nrows();
    const size_t num_columns = this->ncols();

    if(transpose_ == false)
    {
        assert(num_columns == input_.size());
        assert(num_rows == output_.size());

        for(size_t row = 0; row < num_rows; ++row)
        {
            Type row_sum = 0.;
            for(size_t col = 0; col < num_columns; ++col)
            {
                size_t index = (num_columns * row) + col;
                const Type value = m_MatrixData[index] * input_[col];
                row_sum += value;
            }
            output_[row] = row_sum;
        }
    }
    else
    {
        assert(num_rows == input_.size());
        assert(num_columns == output_.size());

        for(size_t row = 0; row < num_rows; ++row)
        {
            for(size_t col = 0; col < num_columns; ++col)
            {
                size_t index = (num_columns * row) + col;
                const Type value = m_MatrixData[index] * input_[row];
                const Type value_to_add = output_[col] + value;
                output_[col] = value_to_add;
            }
        }
    }
}

template<class Type>
void DOTk_DenseMatrix<Type>::gemv(const Type & alpha_,
                                  const dotk::vector<Type> & input_,
                                  const Type & beta_,
                                  dotk::vector<Type> & output_,
                                  bool transpose_) const
{
    // Quick return if possible
    if(alpha_ == static_cast<Type>(0.))
    {
        if(beta_ == static_cast<Type>(0.))
        {
            output_.fill(0.);
        }
        else
        {
            output_.scale(beta_);
        }
        return;
    }
    // Start operations
    const size_t num_rows = this->nrows();
    const size_t num_cols = this->ncols();
    if(transpose_ == false)
    {
        assert(num_cols == input_.size());
        assert(num_rows == output_.size());

        for(size_t row = 0; row < num_rows; ++row)
        {
            Type scaled_row_sum = 0.;
            for(size_t col = 0; col < num_cols; ++col)
            {
                const size_t index = (num_cols * row) + col;
                const Type value = m_MatrixData[index] * input_[col];
                scaled_row_sum += alpha_ * value;
            }
            const Type beta_times_output = beta_ * output_[row];
            output_[row] = scaled_row_sum + beta_times_output;
        }
    }
    else
    {
        assert(num_rows == input_.size());
        assert(num_cols == output_.size());
        // Scale output vector if necessary
        if(beta_ == 0)
        {
            output_.fill(0.);
        }
        else
        {
            output_.scale(beta_);
        }
        // Start operations
        for(size_t row = 0; row < num_rows; ++row)
        {
            for(size_t col = 0; col < num_cols; ++col)
            {
                size_t index = (num_cols * row) + col;
                const Type value = alpha_ * m_MatrixData[index] * input_[row];
                const Type value_to_add = output_[col] + value;
                output_[col] = value_to_add;
            }
        }
    }
}

template<class Type>
void DOTk_DenseMatrix<Type>::gemm(const bool & transpose_A_,
                                  const bool & transpose_B_,
                                  const Type & alpha_,
                                  const dotk::matrix<Type> & B_,
                                  const Type & beta_,
                                  dotk::matrix<Type> & C_) const
{
    // Quick return if possible
    if(alpha_ == static_cast<Type>(0.))
    {
        if(beta_ == static_cast<Type>(0.))
        {
            C_.fill(0.);
        }
        else
        {
            C_.scale(beta_);
        }
        return;
    }
    // Scale Output Matrix if Necessary
    if(beta_ != static_cast<Type>(0.))
    {
        C_.scale(beta_);
    }
    else
    {
        C_.fill(0.);
    }
    // Start Operations
    if(transpose_B_ == false)
    {
        if(transpose_A_ == false)
        {
            // C = (A)(B)
            assert(this->ncols() == B_.nrows());
            assert(C_.nrows() == this->nrows());
            assert(C_.ncols() == B_.ncols());

            Type value = 0.;
            size_t i, j, k, index;
            size_t num_rows_A = this->nrows();
            size_t num_cols_A = this->ncols();
            size_t num_cols_B = B_.ncols();

            for(i = 0; i < num_rows_A; ++i)
            {
                for(k = 0; k < num_cols_A; ++k)
                {
                    for(j = 0; j < num_cols_B; ++j)
                    {
                        index = (num_cols_A * i) + k;
                        value = alpha_ * m_MatrixData[index] * B_(k, j);
                        C_(i, j) += value;
                    }
                }
            }
        }
        else
        {
            // C = (A^t)(B)
            assert(this->nrows() == B_.nrows());
            assert(C_.nrows() == this->ncols());
            assert(C_.ncols() == B_.ncols());

            Type value = 0.;
            size_t i, j, k, index;
            size_t num_rows_A = this->nrows();
            size_t num_cols_A = this->ncols();
            size_t num_cols_B = B_.ncols();

            for(k = 0; k < num_rows_A; ++k)
            {
                for(i = 0; i < num_cols_A; ++i)
                {
                    for(j = 0; j < num_cols_B; ++j)
                    {
                        index = (num_cols_A * k) + i;
                        value = alpha_ * m_MatrixData[index] * B_(k, j);
                        C_(i, j) += value;
                    }
                }
            }
        }
    }
    else
    {
        if(transpose_A_ == false)
        {
            // C = (A)(B^t)
            assert(this->ncols() == B_.ncols());
            assert(C_.nrows() == this->nrows());
            assert(C_.ncols() == B_.nrows());

            Type value = 0.;
            size_t i, j, k, index;
            size_t num_rows_A = this->nrows();
            size_t num_cols_A = this->ncols();
            size_t num_rows_B = B_.nrows();

            for(i = 0; i < num_rows_A; ++i)
            {
                for(j = 0; j < num_rows_B; ++j)
                {
                    for(k = 0; k < num_cols_A; ++k)
                    {
                        index = (num_cols_A * i) + k;
                        value = alpha_ * m_MatrixData[index] * B_(j, k);
                        C_(i, j) += value;
                    }
                }
            }
        }
        else
        {
            // C = (A^t)(B^t)
            assert(this->nrows() == B_.ncols());
            assert(C_.nrows() == this->ncols());
            assert(C_.ncols() == B_.nrows());

            Type value = 0.;
            size_t i, j, k, index;
            size_t num_rows_A = this->nrows();
            size_t num_cols_A = this->ncols();
            size_t num_rows_B = B_.nrows();

            for(j = 0; j < num_rows_B; ++j)
            {
                for(k = 0; k < num_rows_A; ++k)
                {
                    for(i = 0; i < num_cols_A; ++i)
                    {
                        index = (num_cols_A * k) + i;
                        value = alpha_ * m_MatrixData[index] * B_(j, k);
                        C_(i, j) += value;
                    }
                }
            }
        }
    }
}

template<class Type>
void DOTk_DenseMatrix<Type>::shift(const Type & alpha_)
{
    const size_t nrows = this->nrows();
    const size_t ncols = this->ncols();

    for(size_t row = 0; row < nrows; ++row)
    {
        size_t index = (ncols * row) + row;
        m_MatrixData[index] += alpha_;
    }
}

template<class Type>
void DOTk_DenseMatrix<Type>::scale(const Type & alpha_)
{
    const size_t nrows = this->nrows();
    const size_t ncols = this->ncols();
    const size_t dim = nrows * ncols;

    for(size_t index = 0; index < dim; ++index)
    {
        m_MatrixData[index] = alpha_ * m_MatrixData[index];
    }
}

template<class Type>
void DOTk_DenseMatrix<Type>::diag(dotk::vector<Type> & input_) const
{
    const size_t num_rows = this->nrows();
    const size_t num_cols = this->ncols();

    assert(num_rows == input_.size());
    assert(num_cols == input_.size());

    for(size_t row = 0; row < num_rows; ++row)
    {
        const size_t index = (num_cols * row) + row;
        input_[row] = m_MatrixData[index];
    }
}

template<class Type>
void DOTk_DenseMatrix<Type>::setDiag(const dotk::vector<Type> & input_, bool zero_elements_)
{
    if(zero_elements_ == true)
    {
        this->fill(0.);
    }

    const size_t num_rows = this->nrows();
    const size_t num_cols = this->ncols();

    assert(num_rows == input_.size());
    assert(num_cols == input_.size());

    for(size_t row = 0; row < num_rows; ++row)
    {
        const size_t index = (num_cols * row) + row;
        m_MatrixData[index] = input_[row];
    }
}

template<class Type>
void DOTk_DenseMatrix<Type>::scaleDiag(const Type & alpha_)
{
    const size_t nrows = this->nrows();
    const size_t ncols = this->ncols();

    for(size_t row = 0; row < nrows; ++row)
    {
        const size_t index = (ncols * row) + row;
        m_MatrixData[index] = alpha_ * m_MatrixData[index];
    }
}

template<class Type>
void DOTk_DenseMatrix<Type>::fill(const Type & value_)
{
    const size_t nrows = this->nrows();
    const size_t ncols = this->ncols();
    const size_t dim = nrows * ncols;

    for(size_t index = 0; index < dim; ++index)
    {
        m_MatrixData[index] = value_;
    }
}

template<class Type>
void DOTk_DenseMatrix<Type>::copy(const dotk::matrix<Type> & input_)
{
    assert(input_.size() == this->size());
    assert(input_.nrows() == this->nrows());
    assert(input_.ncols() == this->ncols());

    size_t index, row, column;
    size_t num_cols = this->ncols();
    size_t num_rows = this->nrows();

    for(row = 0; row < num_rows; ++row)
    {
        for(column = 0; column < num_cols; ++column)
        {
            index = (num_cols * row) + column;
            m_MatrixData[index] = input_(row, column);
        }
    }
}

template<class Type>
void DOTk_DenseMatrix<Type>::copy(const size_t & num_inputs_, const Type* input_)
{
    assert(num_inputs_ == this->size());

    for(size_t index = 0; index < num_inputs_; ++index)
    {
        m_MatrixData[index] = input_[index];
    }
}

template<class Type>
void DOTk_DenseMatrix<Type>::gather(const size_t & dim_, Type* output_)
{
    const size_t dim = this->nrows() * this->ncols();

    assert(dim == dim_);

    for(size_t index = 0; index < dim_; ++index)
    {
        output_[index] = m_MatrixData[index];
    }
}

template<class Type>
void DOTk_DenseMatrix<Type>::identity()
{
    size_t index, row;
    for(row = 0; row < this->nrows(); ++row)
    {
        size_t index = (m_NumCols * row) + row;
        m_MatrixData[index] = 1.;
    }
}

template<class Type>
void DOTk_DenseMatrix<Type>::set(const size_t & row_index_, const size_t & column_index_, Type value_)
{
    assert(row_index_ <= this->nrows() - 1u);
    assert(column_index_ <= this->ncols() - 1u);
    size_t index = (m_NumCols * row_index_) + column_index_;
    m_MatrixData[index] = value_;
}

template<class Type>
std::tr1::shared_ptr<dotk::matrix<Type> > DOTk_DenseMatrix<Type>::clone() const
{
    size_t num_rows = this->nrows();
    std::tr1::shared_ptr<dotk::serial::DOTk_DenseMatrix<Type> >
        matrix(new dotk::serial::DOTk_DenseMatrix<Type>(num_rows, 0.));
    return (matrix);
}

template<class Type>
Type & DOTk_DenseMatrix<Type>::operator()(const size_t & row_index_, const size_t & column_index_)
{
    assert(row_index_ <= this->nrows() - 1u);
    assert(column_index_ <= this->ncols() - 1u);
    size_t index = (m_NumCols * row_index_) + column_index_;
    return (m_MatrixData[index]);
}

template<class Type>
const Type & DOTk_DenseMatrix<Type>::operator()(const size_t & row_index_, const size_t & column_index_) const
{
    assert(row_index_ <= this->nrows() - 1u);
    assert(column_index_ <= this->ncols() - 1u);
    size_t index = (m_NumCols * row_index_) + column_index_;
    return (m_MatrixData[index]);
}

template<class Type>
dotk::types::matrix_t DOTk_DenseMatrix<Type>::type() const
{
    return (dotk::types::SERIAL_DENSE_MATRIX);
}

template<class Type>
void DOTk_DenseMatrix<Type>::clear()
{
    delete[] m_MatrixData;
    m_MatrixData = NULL;
}

}

}