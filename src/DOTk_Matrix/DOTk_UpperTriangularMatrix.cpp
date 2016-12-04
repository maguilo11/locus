/*
 * DOTk_UpperTriangularMatrix.cpp
 *
 *  Created on: Jul 2, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#include <cmath>
#include <cassert>

#include "vector.hpp"
#include "DOTk_UpperTriangularMatrix.hpp"

namespace dotk
{

namespace serial
{

template<class Type>
DOTk_UpperTriangularMatrix<Type>::DOTk_UpperTriangularMatrix(size_t num_rows_, Type value_) :
        dotk::matrix<Type>(),
        m_Zero(0),
        m_Data(NULL),
        m_Size(0),
        m_NumRows(num_rows_),
        m_NumCols(num_rows_),
        m_Displacements(new size_t[num_rows_]),
        m_NumColumnCount(new size_t[num_rows_])
{
    this->initialize();
    this->fill(value_);
}

template<class Type>
DOTk_UpperTriangularMatrix<Type>::~DOTk_UpperTriangularMatrix()
{
    this->clear();
}

template<class Type>
size_t DOTk_UpperTriangularMatrix<Type>::nrows() const
{
    return (m_NumRows);
}

template<class Type>
size_t DOTk_UpperTriangularMatrix<Type>::ncols() const
{
    return (m_NumCols);
}

template<class Type>
size_t DOTk_UpperTriangularMatrix<Type>::size() const
{
    return (m_Size);
}

template<class Type>
void DOTk_UpperTriangularMatrix<Type>::copy(const size_t & index_,
                                            const dotk::vector<Type> & input_,
                                            bool row_major_copy_)
{
    if(row_major_copy_ == false)
    {
        assert(this->nrows() == input_.size());

        size_t data_index, row;
        size_t total_num_rows = index_ + 1;

        for(row = 0; row < total_num_rows; ++ row)
        {
            data_index = m_Displacements[row] + index_ - row;
            m_Data[data_index] = input_[row];
        }
    }
    else
    {
        assert(this->ncols() == input_.size());

        size_t input_index, data_index, column;

        for(column = 0; column < m_NumColumnCount[index_]; ++ column)
        {
            input_index = index_ + column;
            data_index = m_Displacements[index_] + column;
            m_Data[data_index] = input_[input_index];
        }
    }
}

template<class Type>
Type DOTk_UpperTriangularMatrix<Type>::norm(const size_t & index_, bool row_major_norm_) const
{
    Type value = 0.;

    if(row_major_norm_ == false)
    {
        size_t num_rows = index_ + 1;
        size_t data_index, row_index;

        for(row_index = 0; row_index < num_rows; ++ row_index)
        {
            data_index = m_Displacements[row_index] + index_ - row_index;
            value += m_Data[data_index] * m_Data[data_index];
        }
    }
    else
    {
        size_t data_index, col_index;

        for(col_index = 0; col_index < m_NumColumnCount[index_]; ++ col_index)
        {
            data_index = m_Displacements[index_] + col_index;
            value += m_Data[data_index] * m_Data[data_index];
        }
    }

    value = std::pow(value, 0.5);

    return (value);
}

template<class Type>
void DOTk_UpperTriangularMatrix<Type>::scale(const size_t & index_, const Type & alpha_, bool row_major_scale_)
{
    if(row_major_scale_ == false)
    {
        Type scaled_value;
        size_t num_rows = index_ + 1;
        size_t data_index, row_index;

        for(row_index = 0; row_index < num_rows; ++ row_index)
        {
            data_index = m_Displacements[row_index] + index_ - row_index;
            scaled_value = alpha_ * m_Data[data_index];
            m_Data[data_index] = scaled_value;
        }
    }
    else
    {
        Type scaled_value;
        size_t data_index, col_index;

        for(col_index = 0; col_index < m_NumColumnCount[index_]; ++ col_index)
        {
            data_index = m_Displacements[index_] + col_index;
            scaled_value = alpha_ * m_Data[data_index];
            m_Data[data_index] = scaled_value;
        }
    }
}

template<class Type>
void DOTk_UpperTriangularMatrix<Type>::axpy(const size_t & index_,
                                            const Type & alpha_,
                                            const dotk::vector<Type> & input_,
                                            bool row_major_axpy_)
{
    if(row_major_axpy_ == false)
    {
        assert(this->nrows() == input_.size());

        Type scaled_value, value;
        size_t num_rows = index_ + 1;
        size_t data_index, row_index;

        for(row_index = 0; row_index < num_rows; ++row_index)
        {
            scaled_value = alpha_ * input_[row_index];
            data_index = m_Displacements[row_index] + index_ - row_index;
            value = m_Data[data_index] + scaled_value;
            m_Data[data_index] = value;
        }
    }
    else
    {
        const size_t num_columns = this->ncols();
        assert(num_columns == input_.size());

        Type scaled_value, value;
        size_t input_index, data_index, col_index;

        for(col_index = 0; col_index < m_NumColumnCount[index_]; ++col_index)
        {
            input_index = index_ + col_index;
            data_index = m_Displacements[index_] + col_index;
            scaled_value = alpha_ * input_[input_index];
            value = m_Data[data_index] + scaled_value;
            m_Data[data_index] = value;
        }
    }
}

template<class Type>
Type DOTk_UpperTriangularMatrix<Type>::dot(const size_t & index_,
                                           const dotk::vector<Type> & input_,
                                           bool row_major_dot_) const
{
    Type value = 0.;

    if(row_major_dot_ == false)
    {
        assert(this->nrows() == input_.size());

        size_t num_rows = index_ + 1;
        size_t data_index, row_index;

        for(row_index = 0; row_index < num_rows; ++ row_index)
        {
            data_index = m_Displacements[row_index] + index_ - row_index;
            value += m_Data[data_index] * input_[row_index];
        }
    }
    else
    {
        assert(this->ncols() == input_.size());

        size_t input_index, data_index, col_index;

        for(col_index = 0; col_index < m_NumColumnCount[index_]; ++ col_index)
        {
            input_index = index_ + col_index;
            data_index = m_Displacements[index_] + col_index;
            value += m_Data[data_index] * input_[input_index];
        }
    }

    return (value);
}

template<class Type>
Type DOTk_UpperTriangularMatrix<Type>::norm() const
{
    Type value = 0.;
    const size_t dim = this->size();

    for(size_t index = 0; index < dim; ++ index)
    {
        value += (m_Data + index)[0] * (m_Data + index)[0];
    }

    value = std::pow(value, 0.5);

    return (value);
}

template<class Type>
void DOTk_UpperTriangularMatrix<Type>::matVec(const dotk::vector<Type> & input_,
                                              dotk::vector<Type> & output_,
                                              bool transpose_) const
{
    output_.fill(0);
    const size_t nrows = this->nrows();
    const size_t ncols = this->ncols();

    if(transpose_ == false)
    {
        assert(ncols == input_.size());
        assert(nrows == output_.size());

        Type sum, value;
        size_t displacement, row_index, col_index, data_index, input_index;

        for(row_index = 0; row_index < output_.size(); ++ row_index)
        {
            sum = 0.;
            displacement = m_Displacements[row_index];
            for(col_index = 0; col_index < m_NumColumnCount[row_index]; ++col_index)
            {
                input_index = row_index + col_index;
                data_index = displacement + col_index;
                value = m_Data[data_index] * input_[input_index];
                sum += value;
            }
            output_[row_index] = sum;
        }
    }
    else
    {
        assert(nrows == input_.size());
        assert(ncols == output_.size());

        Type input_value, value, value_to_add;
        size_t row_index, col_index, displacement, data_index, output_index;

        for(row_index = 0; row_index < input_.size(); ++ row_index)
        {
            input_value = input_[row_index];
            displacement = m_Displacements[row_index];
            for(col_index = 0; col_index < m_NumColumnCount[row_index]; ++col_index)
            {
                data_index = displacement + col_index;
                value = m_Data[data_index] * input_value;
                output_index = col_index + row_index;
                value_to_add = output_[output_index] + value;
                output_[output_index] = value_to_add;
            }
        }
    }
}

template<class Type>
void DOTk_UpperTriangularMatrix<Type>::gemv(const Type & alpha_,
                                            const dotk::vector<Type> & input_,
                                            const Type & beta_,
                                            dotk::vector<Type> & output_,
                                            bool transpose_) const
{
    const size_t nrows = this->nrows();
    const size_t ncols = this->ncols();

    if(transpose_ == false)
    {
        assert(ncols == input_.size());
        assert(nrows == output_.size());

        Type sum, value, beta_times_output;
        size_t displacement, row_index, col_index, data_index, input_index;

        for(row_index = 0; row_index < output_.size(); ++ row_index)
        {
            sum = 0.;
            displacement = m_Displacements[row_index];
            for(col_index = 0; col_index < m_NumColumnCount[row_index]; ++ col_index)
            {
                input_index = row_index + col_index;
                data_index = displacement + col_index;
                value = m_Data[data_index] * input_[input_index];
                sum += alpha_ * value;
            }
            beta_times_output = beta_ * output_[row_index];
            output_[row_index] = sum + beta_times_output;
        }
    }
    else
    {
        assert(nrows == input_.size());
        assert(ncols == output_.size());

        Type scaled_value;

        for(size_t i = 0; i < output_.size(); ++ i)
        {
            scaled_value = beta_ * output_[i];
            output_[i] = scaled_value;
        }

        Type input_value, value, value_to_add;
        size_t displacement, row_index, col_index, data_index, output_index;

        for(row_index = 0; row_index < input_.size(); ++ row_index)
        {
            input_value = input_[row_index];
            displacement = m_Displacements[row_index];
            for(col_index = 0; col_index < m_NumColumnCount[row_index]; ++ col_index)
            {
                data_index = displacement + col_index;
                value = alpha_ * m_Data[data_index] * input_value;
                output_index = col_index + row_index;
                value_to_add = output_[output_index] + value;
                output_[output_index] = value_to_add;
            }
        }
    }
}

template<class Type>
void DOTk_UpperTriangularMatrix<Type>::gemm(const bool & transpose_A_,
                                            const bool & transpose_B_,
                                            const Type & alpha_,
                                            const dotk::matrix<Type> & B_,
                                            const Type & beta_,
                                            dotk::matrix<Type> & C_) const
{
    // TODO: IMPLEMENT GENERAL MATRIX-MATRIX PRODUCT FOR UPPER TRIANGULAR MATRIX
}

template<class Type>
void DOTk_UpperTriangularMatrix<Type>::scale(const Type & alpha_)
{
    const size_t dim = this->size();

    for(size_t index = 0; index < dim; ++ index)
    {
        (m_Data + index)[0] = alpha_ * (m_Data + index)[0];
    }
}

template<class Type>
void DOTk_UpperTriangularMatrix<Type>::fill(const Type & value_)
{
    const size_t dim = this->size();

    for(size_t index = 0; index < dim; ++ index)
    {
        (m_Data + index)[0] = value_;
    }
}

template<class Type>
void DOTk_UpperTriangularMatrix<Type>::copy(const dotk::matrix<Type> & input_)
{
    assert(input_.size() == this->size());
    assert(input_.type() == this->type());

    size_t row, column, index;

    for(row = 0; row < this->nrows(); ++row)
    {
        for(column = 0; column < this->ncols(); ++ column)
        {
            index = m_Displacements[row] + column - row;
            m_Data[index] = input_(row, column);
        }
    }
}

template<class Type>
void DOTk_UpperTriangularMatrix<Type>::copy(const size_t & num_inputs_, const Type* input_)
{
    assert(num_inputs_ == this->size());

    for(size_t index = 0; index < num_inputs_; ++index)
    {
        (m_Data + index)[0] = (input_ + index)[0];
    }
}

template<class Type>
void DOTk_UpperTriangularMatrix<Type>::gather(const size_t & dim_, Type* output_)
{
    const size_t dim = this->size();
    assert(dim == dim_);

    for(size_t index = 0; index < dim_; ++ index)
    {
        (output_ + index)[0] = (m_Data + index)[0];
    }
}

template<class Type>
void DOTk_UpperTriangularMatrix<Type>::identity()
{
    this->fill(0.);
    size_t data_index, row;
    size_t num_rows = this->nrows();

    for(row = 0; row < num_rows; ++row)
    {
        data_index = m_Displacements[row];
        m_Data[data_index] = 1.;
    }
}

template<class Type>
void DOTk_UpperTriangularMatrix<Type>::shift(const Type & alpha_)
{
    Type shifted_value = 0.;
    size_t data_index, row;
    size_t num_rows = this->nrows();

    for(row = 0; row < num_rows; ++ row)
    {
        data_index = m_Displacements[row];
        shifted_value = alpha_ + m_Data[data_index];
        m_Data[data_index] = shifted_value;
    }
}

template<class Type>
void DOTk_UpperTriangularMatrix<Type>::diag(dotk::vector<Type> & input_) const
{
    size_t num_rows = this->nrows();

    assert(input_.size() == num_rows);

    size_t data_index, row;

    for(row = 0; row < num_rows; ++row)
    {
        data_index = m_Displacements[row];
        input_[row] = m_Data[data_index];
    }
}

template<class Type>
void DOTk_UpperTriangularMatrix<Type>::setDiag(const dotk::vector<Type> & input_, bool zero_elements_)
{
    size_t num_rows = this->nrows();

    assert(input_.size() == num_rows);

    if(zero_elements_ == true)
    {
        this->fill(0.);
    }

    size_t data_index, row;

    for(row = 0; row < num_rows; ++ row)
    {
        data_index = m_Displacements[row];
        m_Data[data_index] = input_[row];
    }
}

template<class Type>
void DOTk_UpperTriangularMatrix<Type>::scaleDiag(const Type & alpha_)
{
    Type scaled_value = 0.;
    size_t data_index, row;
    size_t num_rows = this->nrows();

    for(row = 0; row < num_rows; ++ row)
    {
        data_index = m_Displacements[row];
        scaled_value = alpha_ * m_Data[data_index];
        m_Data[data_index] = scaled_value;
    }
}

template<class Type>
Type DOTk_UpperTriangularMatrix<Type>::trace() const
{
    Type value = 0.;
    size_t data_index, row;
    size_t num_rows = this->nrows();

    for(row = 0; row < num_rows; ++ row)
    {
        data_index = m_Displacements[row];
        value += m_Data[data_index];
    }

    return (value);
}

template<class Type>
void DOTk_UpperTriangularMatrix<Type>::set(const size_t & row_index_, const size_t & column_index_, Type value_)
{
    assert(column_index_ >= row_index_);
    size_t index = m_Displacements[row_index_] + column_index_ - row_index_;
    m_Data[index] = value_;
}

template<class Type>
Type & DOTk_UpperTriangularMatrix<Type>::operator ()(const size_t & row_index_, const size_t & column_index_)
{
    assert(row_index_ < this->nrows());
    assert(column_index_ < this->ncols());
    if(column_index_ >= row_index_)
    {
        size_t index = m_Displacements[row_index_] + column_index_ - row_index_;
        return (m_Data[index]);
    }
    else
    {
        return(m_Zero);
    }
}

template<class Type>
const Type & DOTk_UpperTriangularMatrix<Type>::operator ()(const size_t & row_index_,
                                                           const size_t & column_index_) const
{
    assert(row_index_ < this->nrows());
    assert(column_index_ < this->ncols());
    if(column_index_ >= row_index_)
    {
        size_t index = m_Displacements[row_index_] + column_index_ - row_index_;
        return (m_Data[index]);
    }
    else
    {
        return(m_Zero);
    }
}

template<class Type>
std::tr1::shared_ptr<dotk::matrix<Type> > DOTk_UpperTriangularMatrix<Type>::clone() const
{

    size_t num_rows = this->nrows();
    std::tr1::shared_ptr<dotk::serial::DOTk_UpperTriangularMatrix<Type> >
        matrix(new dotk::serial::DOTk_UpperTriangularMatrix<Type>(num_rows));

    return (matrix);
}

template<class Type>
dotk::types::matrix_t DOTk_UpperTriangularMatrix<Type>::type() const
{
    return (dotk::types::SERIAL_UPPER_TRI_MATRIX);
}

template<class Type>
void DOTk_UpperTriangularMatrix<Type>::initialize()
{
    size_t num_rows = this->nrows();
    size_t num_columns = this->ncols();
    assert(num_rows == num_columns);

    for(size_t row = 0; row < num_rows; ++ row)
    {
        size_t num_cols_in_row = num_columns - row;
        m_NumColumnCount[row] = num_cols_in_row;
        m_Displacements[row] = m_Size;
        m_Size += num_cols_in_row;
    }
    m_Data = new Type[m_Size];
}

template<class Type>
void DOTk_UpperTriangularMatrix<Type>::clear()
{
    delete[] m_Data;
    m_Data = NULL;
    delete[] m_Displacements;
    m_Displacements = NULL;
    delete[] m_NumColumnCount;
    m_NumColumnCount = NULL;
}

}

}
