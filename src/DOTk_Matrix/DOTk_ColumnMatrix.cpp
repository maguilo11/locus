/*
 * DOTk_ColumnMatrix.cpp
 *
 *  Created on: Jul 1, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#include "vector.hpp"
#include "DOTk_ColumnMatrix.hpp"

namespace dotk
{

namespace serial
{

template<class Type>
DOTk_ColumnMatrix<Type>::DOTk_ColumnMatrix(const dotk::vector<Type> & column_, size_t storage_size_) :
        dotk::matrix<Type>(),
        m_Size(column_.size() * storage_size_),
        m_NumRows(column_.size()),
        m_NumColumns(storage_size_),
        m_MatrixData(new std::tr1::shared_ptr<dotk::vector<Type> >[storage_size_])
{
    this->initialize(column_);
}

template<class Type>
DOTk_ColumnMatrix<Type>::~DOTk_ColumnMatrix()
{
    this->clear();
}

template<class Type>
size_t DOTk_ColumnMatrix<Type>::nrows() const
{
    return (m_NumRows);
}

template<class Type>
size_t DOTk_ColumnMatrix<Type>::ncols() const
{
    return (m_NumColumns);
}

template<class Type>
size_t DOTk_ColumnMatrix<Type>::size() const
{
    return (m_Size);
}

template<class Type>
void DOTk_ColumnMatrix<Type>::copy(const size_t & index_,
                                   const dotk::vector<Type> & input_,
                                   bool column_major_copy_)
{
    if(column_major_copy_ == false)
    {
        assert(input_.size() == this->ncols());

        size_t column;

        for(column = 0; column < input_.size(); ++column)
        {
            m_MatrixData[column]->operator[](index_) = input_.operator[](column);
        }
    }
    else
    {
        assert(input_.size() == this->nrows());
        m_MatrixData[index_]->copy(input_);
    }
}

template<class Type>
Type DOTk_ColumnMatrix<Type>::norm(const size_t & index_, bool column_major_norm_) const
{
    Type value = 0.;
    const size_t num_rows = this->nrows();
    const size_t num_columns = this->ncols();

    if(column_major_norm_ == false)
    {
        size_t data_index, column;

        for(column = 0; column < num_columns; ++ column)
        {
            data_index = (num_rows * column) + index_;
            value += m_MatrixData[column]->operator[](index_) * m_MatrixData[column]->operator[](index_);
        }

        value = std::pow(value, 0.5);
    }
    else
    {
        value = m_MatrixData[index_]->norm();
    }

    return (value);
}

template<class Type>
void DOTk_ColumnMatrix<Type>::scale(const size_t & index_, const Type & alpha_, bool column_major_scale_)
{
    Type value = 0.;

    if(column_major_scale_ == false)
    {
        size_t column;

        for(column = 0; column < this->ncols(); ++ column)
        {
            value = m_MatrixData[column]->operator[](index_) * alpha_;
            m_MatrixData[column]->operator[](index_) = value;
        }
    }
    else
    {
        m_MatrixData[index_]->scale(alpha_);
    }
}

template<class Type>
void DOTk_ColumnMatrix<Type>::axpy(const size_t & index_,
                                    const Type & alpha_,
                                    const dotk::vector<Type> & input_,
                                    bool column_major_axpy_)
{
    Type value, scaled_value;

    if(column_major_axpy_ == false)
    {
        const size_t num_columns = this->ncols();
        assert(num_columns == input_.size());

        size_t column;

        for(column = 0; column < input_.size(); ++ column)
        {
            scaled_value = alpha_ * input_.operator[](column);
            value = m_MatrixData[column]->operator[](index_) + scaled_value;
            m_MatrixData[column]->operator[](index_) = value;
        }
    }
    else
    {
        assert(this->nrows() == input_.size());
        m_MatrixData[index_]->axpy(alpha_, input_);
    }
}

template<class Type>
Type DOTk_ColumnMatrix<Type>::dot(const size_t & index_,
                                  const dotk::vector<Type> & input_,
                                  bool column_major_dot_) const
{
    Type result = 0.;

    if(column_major_dot_ == false)
    {
        assert(this->ncols() == input_.size());

        size_t column;

        for(column = 0; column < input_.size(); ++ column)
        {
            result += m_MatrixData[column]->operator[](index_) * input_.operator[](column);
        }
    }
    else
    {
        assert(this->nrows() == input_.size());
        result = m_MatrixData[index_]->dot(input_);
    }

    return (result);
}

template<class Type>
Type DOTk_ColumnMatrix<Type>::norm() const
{
    Type value = 0.;

    for(size_t column = 0; column < this->ncols(); ++ column)
    {
        value += m_MatrixData[column]->dot(*m_MatrixData[column]);
    }

    value = std::pow(value, 0.5);

    return (value);
}

template<class Type>
void DOTk_ColumnMatrix<Type>::matVec(const dotk::vector<Type> & input_,
                                     dotk::vector<Type> & output_,
                                     bool transpose_) const
{
    output_.fill(0);
    if(transpose_ == false)
    {
        assert(this->ncols() == input_.size());
        assert(this->nrows() == output_.size());

        size_t row, column;
        Type value, value_to_add;

        for(column = 0; column < input_.size(); ++column)
        {
            for(row = 0; row < output_.size(); ++row)
            {
                value = m_MatrixData[column]->operator[](row) * input_.operator[](column);
                value_to_add = output_.operator[](row) + value;
                output_.operator[](row) = value_to_add;
            }
        }
    }
    else
    {
        assert(this->nrows() == input_.size());
        assert(this->ncols() == output_.size());

        Type sum, value;
        size_t row, column;

        for(column = 0; column < output_.size(); ++ column)
        {
            sum = 0.;
            for(row = 0; row < input_.size(); ++ row)
            {
                value = m_MatrixData[column]->operator[](row) * input_.operator[](row);
                sum += value;
            }
            output_.operator[](column) = sum;
        }
    }
}

template<class Type>
void DOTk_ColumnMatrix<Type>::gemv(const Type & alpha_,
                                   const dotk::vector<Type> & input_,
                                   const Type & beta_,
                                   dotk::vector<Type> & output_,
                                   bool transpose_) const
{
    if(transpose_ == false)
    {
        assert(this->ncols() == input_.size());
        assert(this->nrows() == output_.size());

        size_t i;
        for(i = 0; i < output_.size(); ++ i)
        {
            output_[i] = beta_ * output_[i];
        }

        size_t row, column;
        Type value, value_to_add;

        for(column = 0; column < input_.size(); ++ column)
        {
            Type scaled_row_sum = 0.;
            for(row = 0; row < output_.size(); ++ row)
            {
                value = alpha_ * m_MatrixData[column]->operator[](row) * input_.operator[](column);
                value_to_add = output_.operator[](row) + value;
                output_.operator[](row) = value_to_add;
            }
        }
    }
    else
    {
        assert(this->nrows() == input_.size());
        assert(this->ncols() == output_.size());

        size_t row, column;
        Type value, sum, beta_times_output;

        for(column = 0; column < output_.size(); ++ column)
        {
            sum = 0.;
            for(row = 0; row < input_.size(); ++ row)
            {
                value = m_MatrixData[column]->operator[](row) * input_.operator[](row);
                sum += alpha_ * value;
            }
            beta_times_output = beta_ * output_.operator[](column);
            output_.operator[](column) = sum + beta_times_output;
        }
    }
}

template<class Type>
void DOTk_ColumnMatrix<Type>::gemm(const bool & transpose_A_,
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
            size_t i, j, k;
            size_t num_rows_A = this->nrows();
            size_t num_cols_A = this->ncols();
            size_t num_cols_B = B_.ncols();

            for(j = 0; j < num_cols_B; ++j)
            {
                for(k = 0; k < num_cols_A; ++k)
                {
                    for(i = 0; i < num_rows_A; ++i)
                    {
                        value = alpha_ * m_MatrixData[k]->operator[](i) * B_(k, j);
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
            size_t i, j, k;
            size_t num_rows_A = this->nrows();
            size_t num_cols_A = this->ncols();
            size_t num_cols_B = B_.ncols();

            for(j = 0; j < num_cols_B; ++j)
            {
                for(i = 0; i < num_cols_A; ++i)
                {
                    for(k = 0; k < num_rows_A; ++k)
                    {
                        value = alpha_ * m_MatrixData[i]->operator[](k) * B_(k, j);
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
            size_t i, j, k;
            size_t num_rows_A = this->nrows();
            size_t num_cols_A = this->ncols();
            size_t num_rows_B = B_.nrows();

            for(k = 0; k < num_cols_A; ++k)
            {
                for(j = 0; j < num_rows_B; ++j)
                {
                    for(i = 0; i < num_rows_A; ++i)
                    {
                        value = alpha_ * m_MatrixData[k]->operator[](i) * B_(j, k);
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
            size_t i, j, k;
            size_t num_rows_A = this->nrows();
            size_t num_cols_A = this->ncols();
            size_t num_rows_B = B_.nrows();

            for(i = 0; i < num_cols_A; ++i)
            {
                for(k = 0; k < num_rows_A; ++k)
                {
                    for(j = 0; j < num_rows_B; ++j)
                    {
                        value = alpha_ * m_MatrixData[i]->operator[](k) * B_(j, k);
                        C_(i, j) += value;
                    }
                }
            }
        }
    }
}

template<class Type>
void DOTk_ColumnMatrix<Type>::scale(const Type & alpha_)
{
    for(size_t column = 0; column < this->ncols(); ++ column)
    {
        m_MatrixData[column]->scale(alpha_);
    }
}

template<class Type>
void DOTk_ColumnMatrix<Type>::fill(const Type & value_)
{
    for(size_t column = 0; column < this->ncols(); ++ column)
    {
        m_MatrixData[column]->fill(value_);
    }
}

template<class Type>
void DOTk_ColumnMatrix<Type>::copy(const dotk::matrix<Type> & input_)
{
    assert(input_.size() == this->size());
    assert(input_.type() == this->type());

    size_t num_cols = this->ncols();
    size_t num_rows = this->nrows();

    for(size_t column = 0; column < num_cols; ++ column)
    {
        for(size_t row = 0; row < num_rows; ++ row)
        {
            m_MatrixData[column]->operator[](row) = input_(row, column);
        }
    }
}

template<class Type>
void DOTk_ColumnMatrix<Type>::copy(const size_t & num_inputs_, const Type* input_)
{
    assert(num_inputs_ == this->size());

    size_t num_cols = this->ncols();
    size_t num_rows = this->nrows();

    for(size_t column = 0; column < num_cols; ++column)
    {
        for(size_t row = 0; row < num_rows; ++row)
        {
            size_t index = (column * num_rows) + row;
            m_MatrixData[column]->operator[](row) = input_[index];
        }
    }
}

template<class Type>
void DOTk_ColumnMatrix<Type>::gather(const size_t & dim_, Type* output_)
{
    assert(dim_ == this->size());

    size_t num_cols = this->ncols();
    size_t num_rows = this->nrows();

    for(size_t column = 0; column < num_cols; ++ column)
    {
        for(size_t row = 0; row < num_rows; ++ row)
        {
            size_t index = (column * num_rows) + row;
            output_[index] = m_MatrixData[column]->operator[](row);
        }
    }
}

template<class Type>
void DOTk_ColumnMatrix<Type>::set(const size_t & row_index_, const size_t & column_index_, Type value_)
{
    assert(row_index_ <= this->nrows() - 1u);
    assert(column_index_ <= this->ncols() - 1u);
    m_MatrixData[column_index_]->operator[](row_index_) = value_;
}

template<class Type>
void DOTk_ColumnMatrix<Type>::identity()
{
    size_t diagonal_dim;
    if(this->ncols() <= this->nrows())
    {
        diagonal_dim = this->ncols();
    }
    else
    {
        diagonal_dim = this->nrows();
    }

    this->fill(0.);
    for(size_t index = 0; index < diagonal_dim; ++index)
    {
        m_MatrixData[index]->operator[](index) = 1.;
    }
}

template<class Type>
void DOTk_ColumnMatrix<Type>::diag(dotk::vector<Type> & input_) const
{

    size_t diagonal_dim;
    if(this->ncols() <= this->nrows())
    {
        diagonal_dim = this->ncols();
    }
    else
    {
        diagonal_dim = this->nrows();
    }

    assert(diagonal_dim == input_.size());

    for(size_t index = 0; index < diagonal_dim; ++index)
    {
        input_[index] = m_MatrixData[index]->operator[](index);
    }
}

template<class Type>
void DOTk_ColumnMatrix<Type>::setDiag(const dotk::vector<Type> & input_, bool zero_matrix_entries_)
{
    size_t diagonal_dim;
    if(this->ncols() <= this->nrows())
    {
        diagonal_dim = this->ncols();
    }
    else
    {
        diagonal_dim = this->nrows();
    }

    assert(diagonal_dim == input_.size());

    if(zero_matrix_entries_ == true)
    {
        this->fill(0.);
    }

    for(size_t index = 0; index < diagonal_dim; ++index)
    {
         m_MatrixData[index]->operator[](index) = input_[index];
    }
}

template<class Type>
void DOTk_ColumnMatrix<Type>::scaleDiag(const Type & alpha_)
{
    size_t index, diagonal_dim;
    if(this->ncols() <= this->nrows())
    {
        diagonal_dim = this->ncols();
    }
    else
    {
        diagonal_dim = this->nrows();
    }

    Type scaled_value = 0.;
    for(index = 0; index < diagonal_dim; ++index)
    {
        scaled_value = alpha_ * m_MatrixData[index]->operator[](index);
        m_MatrixData[index]->operator[](index) = scaled_value;
    }
}

template<class Type>
Type DOTk_ColumnMatrix<Type>::trace() const
{
    size_t index, diagonal_dim;
    if(this->ncols() <= this->nrows())
    {
        diagonal_dim = this->ncols();
    }
    else
    {
        diagonal_dim = this->nrows();
    }

    Type value = 0.;
    for(index = 0; index < diagonal_dim; ++index)
    {
        value += m_MatrixData[index]->operator[](index);
    }
    return (value);
}

template<class Type>
size_t DOTk_ColumnMatrix<Type>::basisDimension() const
{
    return (m_NumColumns);
}

template<class Type>
std::tr1::shared_ptr<dotk::vector<Type> > & DOTk_ColumnMatrix<Type>::basis(const size_t & index_)
{
    return (m_MatrixData[index_]);
}

template<class Type>
Type & DOTk_ColumnMatrix<Type>::operator ()(const size_t & row_index_, const size_t & column_index_)
{
    assert(row_index_ <= this->nrows() - 1u);
    assert(column_index_ <= this->ncols() - 1u);
    return (m_MatrixData[column_index_]->operator[](row_index_));
}

template<class Type>
const Type & DOTk_ColumnMatrix<Type>::operator ()(const size_t & row_index_, const size_t & column_index_) const
{
    assert(row_index_ <= this->nrows() - 1u);
    assert(column_index_ <= this->ncols() - 1u);
    return (m_MatrixData[column_index_]->operator[](row_index_));
}

template<class Type>
std::tr1::shared_ptr<dotk::matrix<Type> > DOTk_ColumnMatrix<Type>::clone() const
{

    size_t basis_dim = this->basisDimension();
    std::tr1::shared_ptr<dotk::vector<Type> > & basis_vector = m_MatrixData[0];
    std::tr1::shared_ptr<dotk::serial::DOTk_ColumnMatrix<Type> >
        matrix(new dotk::serial::DOTk_ColumnMatrix<Type>(*basis_vector, basis_dim));

    return (matrix);
}

template<class Type>
dotk::types::matrix_t DOTk_ColumnMatrix<Type>::type() const
{
    return (dotk::types::SERIAL_COLUMN_MATRIX);
}

template<class Type>
void DOTk_ColumnMatrix<Type>::initialize(const dotk::vector<Type> & column_)
{
    for(size_t index = 0; index < this->ncols(); ++ index)
    {
        m_MatrixData[index] = column_.clone();
    }
}

template<class Type>
void DOTk_ColumnMatrix<Type>::clear()
{
    delete[] m_MatrixData;
    m_MatrixData = NULL;
}

}

}
