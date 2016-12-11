/*
 * DOTk_RowMatrix.cpp
 *
 *  Created on: Jul 1, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#include <cmath>
#include <cassert>

#include "vector.hpp"
#include "DOTk_RowMatrix.hpp"

namespace dotk
{

namespace serial
{

template<typename ScalarType>
DOTk_RowMatrix<ScalarType>::DOTk_RowMatrix(const dotk::Vector<ScalarType> & row_, size_t num_rows_) :
        dotk::matrix<ScalarType>(),
        m_Size(row_.size() * num_rows_),
        m_NumRows(num_rows_),
        m_NumColumns(row_.size()),
        m_MatrixData(new std::tr1::shared_ptr<dotk::Vector<ScalarType> >[num_rows_])
{
    this->initialize(row_);
}

template<typename ScalarType>
DOTk_RowMatrix<ScalarType>::~DOTk_RowMatrix()
{
    this->clear();
}

template<typename ScalarType>
size_t DOTk_RowMatrix<ScalarType>::nrows() const
{
    return (m_NumRows);
}

template<typename ScalarType>
size_t DOTk_RowMatrix<ScalarType>::ncols() const
{
    return (m_NumColumns);
}

template<typename ScalarType>
size_t DOTk_RowMatrix<ScalarType>::size() const
{
    return (m_Size);
}

template<typename ScalarType>
void DOTk_RowMatrix<ScalarType>::copy(const size_t & index_, const dotk::Vector<ScalarType> & input_, bool row_major_copy_)
{
    if(row_major_copy_ == false)
    {
        assert(input_.size() == this->nrows());

        size_t row;

        for(row = 0; row < input_.size(); ++row)
        {
            m_MatrixData[row]->operator[](index_) = input_.operator[](row);
        }
    }
    else
    {
        assert(input_.size() == this->ncols());
        m_MatrixData[index_]->update(1., input_, 0.);
    }
}

template<typename ScalarType>
ScalarType DOTk_RowMatrix<ScalarType>::dot(const size_t & index_,
                               const dotk::Vector<ScalarType> & input_,
                               bool row_major_dot_) const
{
    ScalarType result = 0.;

    if(row_major_dot_ == false)
    {
        assert(this->nrows() == input_.size());

        size_t row;

        for(row = 0; row < input_.size(); ++row)
        {
            result += m_MatrixData[row]->operator[](index_) * input_.operator[](row);
        }
    }
    else
    {
        assert(this->ncols() == input_.size());
        result = m_MatrixData[index_]->dot(input_);
    }

    return (result);
}

template<typename ScalarType>
ScalarType DOTk_RowMatrix<ScalarType>::norm(const size_t & index_, bool row_major_norm_) const
{
    ScalarType value = 0.;
    const size_t num_rows = this->nrows();

    if(row_major_norm_ == false)
    {
        ScalarType data;
        size_t row;

        for(row = 0; row < num_rows; ++row)
        {
            data = m_MatrixData[row]->operator[](index_);
            value += data * data;
        }

        value = std::sqrt(value);
    }
    else
    {
        value = m_MatrixData[index_]->norm();
    }

    return (value);
}

template<typename ScalarType>
void DOTk_RowMatrix<ScalarType>::scale(const size_t & index_, const ScalarType & alpha_, bool row_major_scale_)
{
    ScalarType value = 0.;

    if(row_major_scale_ == false)
    {
        size_t row;

        for(row = 0; row < this->nrows(); ++row)
        {
            value = m_MatrixData[row]->operator[](index_) * alpha_;
            m_MatrixData[row]->operator[](index_) = value;
        }
    }
    else
    {
        m_MatrixData[index_]->scale(alpha_);
    }
}

template<typename ScalarType>
void DOTk_RowMatrix<ScalarType>::axpy(const size_t & index_,
                                      const ScalarType & alpha_,
                                      const dotk::Vector<ScalarType> & input_,
                                      bool row_major_axpy_)
{
    ScalarType value, scaled_value;

    if(row_major_axpy_ == false)
    {
        assert(this->nrows() == input_.size());

        size_t row;

        for(row = 0; row < input_.size(); ++row)
        {
            scaled_value = alpha_ * input_.operator[](row);
            value = m_MatrixData[row]->operator[](index_) + scaled_value;
            m_MatrixData[row]->operator[](index_) = value;
        }
    }
    else
    {
        assert(this->ncols() == input_.size());
        m_MatrixData[index_]->update(alpha_, input_, 1.);
    }
}

template<typename ScalarType>
ScalarType DOTk_RowMatrix<ScalarType>::norm() const
{
    ScalarType value = 0.;

    for(size_t column = 0; column < this->ncols(); ++ column)
    {
        value += m_MatrixData[column]->dot(*m_MatrixData[column]);
    }

    value = std::pow(value, 0.5);

    return (value);
}

template<typename ScalarType>
void DOTk_RowMatrix<ScalarType>::matVec(const dotk::Vector<ScalarType> & input_,
                                  dotk::Vector<ScalarType> & output_,
                                  bool transpose_) const
{
    output_.fill(0.);
    if(transpose_ == false)
    {
        assert(this->ncols() == input_.size());
        assert(this->nrows() == output_.size());

        ScalarType sum, value;
        size_t column, row;

        for(row = 0; row < output_.size(); ++ row)
        {
            sum = 0.;
            for(column = 0; column < input_.size(); ++column)
            {
                value = m_MatrixData[row]->operator[](column) * input_.operator[](column);
                sum += value;
            }
            output_.operator[](row) = sum;
        }
    }
    else
    {
        assert(this->nrows() == input_.size());
        assert(this->ncols() == output_.size());

        size_t column, row;
        ScalarType value, value_to_add;

        for(row = 0; row < input_.size(); ++ row)
        {
            for(column = 0; column < output_.size(); ++column)
            {
                value = m_MatrixData[row]->operator[](column) * input_.operator[](row);
                value_to_add = output_.operator[](column) + value;
                output_.operator[](column) = value_to_add;
            }
        }
    }
}

template<typename ScalarType>
void DOTk_RowMatrix<ScalarType>::gemv(const ScalarType & alpha_,
                                const dotk::Vector<ScalarType> & input_,
                                const ScalarType & beta_,
                                dotk::Vector<ScalarType> & output_,
                                bool transpose_) const
{
    if(transpose_ == false)
    {
        assert(this->ncols() == input_.size());
        assert(this->nrows() == output_.size());

        size_t column, row;
        ScalarType value, scaled_row_sum, beta_times_output;

        for(row = 0; row < output_.size(); ++ row)
        {
            scaled_row_sum = 0.;
            for(column = 0; column < input_.size(); ++ column)
            {
                value = m_MatrixData[row]->operator[](column) * input_.operator[](column);
                scaled_row_sum += alpha_ * value;
            }
            beta_times_output = beta_ * output_.operator[](row);
            output_.operator[](row) = scaled_row_sum + beta_times_output;
        }
    }
    else
    {
        assert(this->nrows() == input_.size());
        assert(this->ncols() == output_.size());

        for(size_t index = 0; index < output_.size(); ++ index)
        {
            output_[index] = beta_ * output_[index];
        }

        ScalarType value;
        size_t column, row;

        for(row = 0; row < input_.size(); ++row)
        {
            for(column = 0; column < output_.size(); ++column)
            {
                value = alpha_ * m_MatrixData[row]->operator[](column) * input_.operator[](row);
                output_.operator[](column) = output_.operator[](column) + value;
            }
        }
    }
}

template<typename ScalarType>
void DOTk_RowMatrix<ScalarType>::gemm(const bool & transpose_A_,
                                const bool & transpose_B_,
                                const ScalarType & alpha_,
                                const dotk::matrix<ScalarType> & B_,
                                const ScalarType & beta_,
                                dotk::matrix<ScalarType> & C_) const
{
    // Quick return if possible
    if(alpha_ == static_cast<ScalarType>(0.))
    {
        if(beta_ == static_cast<ScalarType>(0.))
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
    if(beta_ != static_cast<ScalarType>(0.))
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

            ScalarType value = 0.;
            size_t i, j, k;
            size_t num_rows_A = this->nrows();
            size_t num_cols_A = this->ncols();
            size_t num_cols_B = B_.ncols();

            for(i = 0; i < num_rows_A; ++i)
            {
                for(k = 0; k < num_cols_A; ++k)
                {
                    for(j = 0; j < num_cols_B; ++j)
                    {
                        value = alpha_ * m_MatrixData[i]->operator[](k) * B_(k, j);
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

            ScalarType value = 0.;
            size_t i, j, k;
            size_t num_rows_A = this->nrows();
            size_t num_cols_A = this->ncols();
            size_t num_cols_B = B_.ncols();

            for(k = 0; k < num_rows_A; ++k)
            {
                for(i = 0; i < num_cols_A; ++i)
                {
                    for(j = 0; j < num_cols_B; ++j)
                    {
                        value = alpha_ * m_MatrixData[k]->operator[](i) * B_(k, j);
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

            ScalarType value = 0.;
            size_t i, j, k;
            size_t num_rows_A = this->nrows();
            size_t num_cols_A = this->ncols();
            size_t num_rows_B = B_.nrows();

            for(i = 0; i < num_rows_A; ++i)
            {
                for(j = 0; j < num_rows_B; ++j)
                {
                    for(k = 0; k < num_cols_A; ++k)
                    {
                        value = alpha_ * m_MatrixData[i]->operator[](k) * B_(j, k);
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

            ScalarType value = 0.;
            size_t i, j, k;
            size_t num_rows_A = this->nrows();
            size_t num_cols_A = this->ncols();
            size_t num_rows_B = B_.nrows();

            for(j = 0; j < num_rows_B; ++j)
            {
                for(k = 0; k < num_rows_A; ++k)
                {
                    for(i = 0; i < num_cols_A; ++i)
                    {
                        value = alpha_ * m_MatrixData[k]->operator[](i) * B_(j, k);
                        C_(i, j) += value;
                    }
                }
            }
        }
    }
}

template<typename ScalarType>
void DOTk_RowMatrix<ScalarType>::scale(const ScalarType & alpha_)
{
    for(size_t row = 0; row < this->nrows(); ++row)
    {
        m_MatrixData[row]->scale(alpha_);
    }
}

template<typename ScalarType>
void DOTk_RowMatrix<ScalarType>::fill(const ScalarType & value_)
{
    for(size_t row = 0; row < this->nrows(); ++row)
    {
        m_MatrixData[row]->fill(value_);
    }
}

template<typename ScalarType>
void DOTk_RowMatrix<ScalarType>::copy(const dotk::matrix<ScalarType> & input_)
{
    assert(input_.size() == this->size());
    assert(input_.type() == this->type());

    for(size_t row = 0; row < this->nrows(); ++ row)
    {
        for(size_t column = 0; column < this->ncols(); ++ column)
        {
            m_MatrixData[row]->operator[](column) = input_(row, column);
        }
    }
}

template<typename ScalarType>
void DOTk_RowMatrix<ScalarType>::copy(const size_t & num_inputs_, const ScalarType* input_)
{
    assert(num_inputs_ == this->size());

    size_t row, column, index;
    size_t num_cols = this->ncols();
    size_t num_rows = this->nrows();

    for(row = 0; row < num_rows; ++row)
    {
        for(column = 0; column < num_cols; ++column)
        {
            index = (row * num_cols) + column;
            m_MatrixData[row]->operator[](column) = input_[index];
        }
    }
}

template<typename ScalarType>
void DOTk_RowMatrix<ScalarType>::gather(const size_t & dim_, ScalarType* output_)
{
    assert(dim_ == this->size());

    size_t num_cols = this->ncols();
    size_t num_rows = this->nrows();

    for(size_t row = 0; row < num_rows; ++row)
    {
        for(size_t column = 0; column < num_cols; ++column)
        {
            size_t index = (row * num_cols) + column;
            output_[index] = m_MatrixData[row]->operator[](column);
        }
    }
}

template<typename ScalarType>
void DOTk_RowMatrix<ScalarType>::set(const size_t & row_index_, const size_t & column_index_, ScalarType value_)
{
    assert(row_index_ <= this->nrows() - 1u);
    assert(column_index_ <= this->ncols() - 1u);
    m_MatrixData[row_index_]->operator [](column_index_) = value_;
}

template<typename ScalarType>
void DOTk_RowMatrix<ScalarType>::identity()
{
    size_t diagonal_dim;
    if(this->nrows() <= this->ncols())
    {
        diagonal_dim = this->nrows();
    }
    else
    {
        diagonal_dim = this->ncols();
    }

    this->fill(0.);
    for(size_t index = 0; index < diagonal_dim; ++index)
    {
        m_MatrixData[index]->operator[](index) = 1.;
    }
}

template<typename ScalarType>
void DOTk_RowMatrix<ScalarType>::diag(dotk::Vector<ScalarType> & input_) const
{
    size_t diagonal_dim;
    if(this->nrows() <= this->ncols())
    {
        diagonal_dim = this->nrows();
    }
    else
    {
        diagonal_dim = this->ncols();
    }

    assert(diagonal_dim == input_.size());

    for(size_t index = 0; index < diagonal_dim; ++index)
    {
        input_[index] = m_MatrixData[index]->operator[](index);
    }
}

template<typename ScalarType>
void DOTk_RowMatrix<ScalarType>::setDiag(const dotk::Vector<ScalarType> & input_, bool zero_matrix_entries_)
{
    size_t diagonal_dim;
    if(this->nrows() <= this->ncols())
    {
        diagonal_dim = this->nrows();
    }
    else
    {
        diagonal_dim = this->ncols();
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

template<typename ScalarType>
void DOTk_RowMatrix<ScalarType>::scaleDiag(const ScalarType & alpha_)
{
    size_t diagonal_dim;
    if(this->nrows() <= this->ncols())
    {
        diagonal_dim = this->nrows();
    }
    else
    {
        diagonal_dim = this->ncols();
    }

    ScalarType scaled_value = 0.;
    for(size_t index = 0; index < diagonal_dim; ++index)
    {
        scaled_value = alpha_ * m_MatrixData[index]->operator[](index);
        m_MatrixData[index]->operator[](index) = scaled_value;
    }
}

template<typename ScalarType>
ScalarType DOTk_RowMatrix<ScalarType>::trace() const
{
    size_t diagonal_dim;
    if(this->nrows() <= this->ncols())
    {
        diagonal_dim = this->nrows();
    }
    else
    {
        diagonal_dim = this->ncols();
    }

    ScalarType value = 0.;
    for(size_t index = 0; index < diagonal_dim; ++ index)
    {
        value += m_MatrixData[index]->operator[](index);
    }
    return (value);
}

template<typename ScalarType>
size_t DOTk_RowMatrix<ScalarType>::basisDimension() const
{
    return (m_NumRows);
}

template<typename ScalarType>
std::tr1::shared_ptr<dotk::Vector<ScalarType> > & DOTk_RowMatrix<ScalarType>::basis(const size_t & index_)
{
    assert(index_ < this->nrows());

    return (m_MatrixData[index_]);
}

template<typename ScalarType>
ScalarType & DOTk_RowMatrix<ScalarType>::operator ()(const size_t & row_index_, const size_t & column_index_)
{
    assert(row_index_ <= this->nrows() - 1u);
    assert(column_index_ <= this->ncols() - 1u);
    return (m_MatrixData[row_index_]->operator [](column_index_));
}

template<typename ScalarType>
const ScalarType & DOTk_RowMatrix<ScalarType>::operator ()(const size_t & row_index_, const size_t & column_index_) const
{
    assert(row_index_ <= this->nrows() - 1u);
    assert(column_index_ <= this->ncols() - 1u);
    return (m_MatrixData[row_index_]->operator [](column_index_));
}

template<typename ScalarType>
std::tr1::shared_ptr<dotk::matrix<ScalarType> > DOTk_RowMatrix<ScalarType>::clone() const
{

    size_t basis_dim = this->basisDimension();
    std::tr1::shared_ptr<dotk::Vector<ScalarType> > & basis_vector = m_MatrixData[0];
    std::tr1::shared_ptr<dotk::serial::DOTk_RowMatrix<ScalarType> >
        matrix(new dotk::serial::DOTk_RowMatrix<ScalarType>(*basis_vector, basis_dim));

    return (matrix);
}

template<typename ScalarType>
dotk::types::matrix_t DOTk_RowMatrix<ScalarType>::type() const
{
    return (dotk::types::SERIAL_ROW_MATRIX);
}

template<typename ScalarType>
void DOTk_RowMatrix<ScalarType>::initialize(const dotk::Vector<ScalarType> & row_)
{
    for(size_t index = 0; index < this->nrows(); ++ index)
    {
        m_MatrixData[index] = row_.clone();
    }
}

template<typename ScalarType>
void DOTk_RowMatrix<ScalarType>::clear()
{
    delete[] m_MatrixData;
    m_MatrixData = NULL;
}

}

}
