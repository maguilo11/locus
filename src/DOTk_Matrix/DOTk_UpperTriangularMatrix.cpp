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

template<typename ScalarType>
DOTk_UpperTriangularMatrix<ScalarType>::DOTk_UpperTriangularMatrix(size_t num_rows_, ScalarType value_) :
        dotk::matrix<ScalarType>(),
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

template<typename ScalarType>
DOTk_UpperTriangularMatrix<ScalarType>::~DOTk_UpperTriangularMatrix()
{
    this->clear();
}

template<typename ScalarType>
size_t DOTk_UpperTriangularMatrix<ScalarType>::nrows() const
{
    return (m_NumRows);
}

template<typename ScalarType>
size_t DOTk_UpperTriangularMatrix<ScalarType>::ncols() const
{
    return (m_NumCols);
}

template<typename ScalarType>
size_t DOTk_UpperTriangularMatrix<ScalarType>::size() const
{
    return (m_Size);
}

template<typename ScalarType>
void DOTk_UpperTriangularMatrix<ScalarType>::copy(const size_t & index_,
                                            const dotk::Vector<ScalarType> & input_,
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

template<typename ScalarType>
ScalarType DOTk_UpperTriangularMatrix<ScalarType>::norm(const size_t & index_, bool row_major_norm_) const
{
    ScalarType value = 0.;

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

template<typename ScalarType>
void DOTk_UpperTriangularMatrix<ScalarType>::scale(const size_t & index_, const ScalarType & alpha_, bool row_major_scale_)
{
    if(row_major_scale_ == false)
    {
        ScalarType scaled_value;
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
        ScalarType scaled_value;
        size_t data_index, col_index;

        for(col_index = 0; col_index < m_NumColumnCount[index_]; ++ col_index)
        {
            data_index = m_Displacements[index_] + col_index;
            scaled_value = alpha_ * m_Data[data_index];
            m_Data[data_index] = scaled_value;
        }
    }
}

template<typename ScalarType>
void DOTk_UpperTriangularMatrix<ScalarType>::axpy(const size_t & index_,
                                            const ScalarType & alpha_,
                                            const dotk::Vector<ScalarType> & input_,
                                            bool row_major_axpy_)
{
    if(row_major_axpy_ == false)
    {
        assert(this->nrows() == input_.size());

        ScalarType scaled_value, value;
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

        ScalarType scaled_value, value;
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

template<typename ScalarType>
ScalarType DOTk_UpperTriangularMatrix<ScalarType>::dot(const size_t & index_,
                                           const dotk::Vector<ScalarType> & input_,
                                           bool row_major_dot_) const
{
    ScalarType value = 0.;

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

template<typename ScalarType>
ScalarType DOTk_UpperTriangularMatrix<ScalarType>::norm() const
{
    ScalarType value = 0.;
    const size_t dim = this->size();

    for(size_t index = 0; index < dim; ++ index)
    {
        value += (m_Data + index)[0] * (m_Data + index)[0];
    }

    value = std::pow(value, 0.5);

    return (value);
}

template<typename ScalarType>
void DOTk_UpperTriangularMatrix<ScalarType>::matVec(const dotk::Vector<ScalarType> & input_,
                                              dotk::Vector<ScalarType> & output_,
                                              bool transpose_) const
{
    output_.fill(0);
    const size_t nrows = this->nrows();
    const size_t ncols = this->ncols();

    if(transpose_ == false)
    {
        assert(ncols == input_.size());
        assert(nrows == output_.size());

        ScalarType sum, value;
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

        ScalarType input_value, value, value_to_add;
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

template<typename ScalarType>
void DOTk_UpperTriangularMatrix<ScalarType>::gemv(const ScalarType & alpha_,
                                            const dotk::Vector<ScalarType> & input_,
                                            const ScalarType & beta_,
                                            dotk::Vector<ScalarType> & output_,
                                            bool transpose_) const
{
    const size_t nrows = this->nrows();
    const size_t ncols = this->ncols();

    if(transpose_ == false)
    {
        assert(ncols == input_.size());
        assert(nrows == output_.size());

        ScalarType sum, value, beta_times_output;
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

        ScalarType scaled_value;

        for(size_t i = 0; i < output_.size(); ++ i)
        {
            scaled_value = beta_ * output_[i];
            output_[i] = scaled_value;
        }

        ScalarType input_value, value, value_to_add;
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

template<typename ScalarType>
void DOTk_UpperTriangularMatrix<ScalarType>::gemm(const bool & transpose_A_,
                                            const bool & transpose_B_,
                                            const ScalarType & alpha_,
                                            const dotk::matrix<ScalarType> & B_,
                                            const ScalarType & beta_,
                                            dotk::matrix<ScalarType> & C_) const
{
    // TODO: IMPLEMENT GENERAL MATRIX-MATRIX PRODUCT FOR UPPER TRIANGULAR MATRIX
}

template<typename ScalarType>
void DOTk_UpperTriangularMatrix<ScalarType>::scale(const ScalarType & alpha_)
{
    const size_t dim = this->size();

    for(size_t index = 0; index < dim; ++ index)
    {
        (m_Data + index)[0] = alpha_ * (m_Data + index)[0];
    }
}

template<typename ScalarType>
void DOTk_UpperTriangularMatrix<ScalarType>::fill(const ScalarType & value_)
{
    const size_t dim = this->size();

    for(size_t index = 0; index < dim; ++ index)
    {
        (m_Data + index)[0] = value_;
    }
}

template<typename ScalarType>
void DOTk_UpperTriangularMatrix<ScalarType>::copy(const dotk::matrix<ScalarType> & input_)
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

template<typename ScalarType>
void DOTk_UpperTriangularMatrix<ScalarType>::copy(const size_t & num_inputs_, const ScalarType* input_)
{
    assert(num_inputs_ == this->size());

    for(size_t index = 0; index < num_inputs_; ++index)
    {
        (m_Data + index)[0] = (input_ + index)[0];
    }
}

template<typename ScalarType>
void DOTk_UpperTriangularMatrix<ScalarType>::gather(const size_t & dim_, ScalarType* output_)
{
    const size_t dim = this->size();
    assert(dim == dim_);

    for(size_t index = 0; index < dim_; ++ index)
    {
        (output_ + index)[0] = (m_Data + index)[0];
    }
}

template<typename ScalarType>
void DOTk_UpperTriangularMatrix<ScalarType>::identity()
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

template<typename ScalarType>
void DOTk_UpperTriangularMatrix<ScalarType>::shift(const ScalarType & alpha_)
{
    ScalarType shifted_value = 0.;
    size_t data_index, row;
    size_t num_rows = this->nrows();

    for(row = 0; row < num_rows; ++ row)
    {
        data_index = m_Displacements[row];
        shifted_value = alpha_ + m_Data[data_index];
        m_Data[data_index] = shifted_value;
    }
}

template<typename ScalarType>
void DOTk_UpperTriangularMatrix<ScalarType>::diag(dotk::Vector<ScalarType> & input_) const
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

template<typename ScalarType>
void DOTk_UpperTriangularMatrix<ScalarType>::setDiag(const dotk::Vector<ScalarType> & input_, bool zero_elements_)
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

template<typename ScalarType>
void DOTk_UpperTriangularMatrix<ScalarType>::scaleDiag(const ScalarType & alpha_)
{
    ScalarType scaled_value = 0.;
    size_t data_index, row;
    size_t num_rows = this->nrows();

    for(row = 0; row < num_rows; ++ row)
    {
        data_index = m_Displacements[row];
        scaled_value = alpha_ * m_Data[data_index];
        m_Data[data_index] = scaled_value;
    }
}

template<typename ScalarType>
ScalarType DOTk_UpperTriangularMatrix<ScalarType>::trace() const
{
    ScalarType value = 0.;
    size_t data_index, row;
    size_t num_rows = this->nrows();

    for(row = 0; row < num_rows; ++ row)
    {
        data_index = m_Displacements[row];
        value += m_Data[data_index];
    }

    return (value);
}

template<typename ScalarType>
void DOTk_UpperTriangularMatrix<ScalarType>::set(const size_t & row_index_, const size_t & column_index_, ScalarType value_)
{
    assert(column_index_ >= row_index_);
    size_t index = m_Displacements[row_index_] + column_index_ - row_index_;
    m_Data[index] = value_;
}

template<typename ScalarType>
ScalarType & DOTk_UpperTriangularMatrix<ScalarType>::operator ()(const size_t & row_index_, const size_t & column_index_)
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

template<typename ScalarType>
const ScalarType & DOTk_UpperTriangularMatrix<ScalarType>::operator ()(const size_t & row_index_,
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

template<typename ScalarType>
std::tr1::shared_ptr<dotk::matrix<ScalarType> > DOTk_UpperTriangularMatrix<ScalarType>::clone() const
{

    size_t num_rows = this->nrows();
    std::tr1::shared_ptr<dotk::serial::DOTk_UpperTriangularMatrix<ScalarType> >
        matrix(new dotk::serial::DOTk_UpperTriangularMatrix<ScalarType>(num_rows));

    return (matrix);
}

template<typename ScalarType>
dotk::types::matrix_t DOTk_UpperTriangularMatrix<ScalarType>::type() const
{
    return (dotk::types::SERIAL_UPPER_TRI_MATRIX);
}

template<typename ScalarType>
void DOTk_UpperTriangularMatrix<ScalarType>::initialize()
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
    m_Data = new ScalarType[m_Size];
}

template<typename ScalarType>
void DOTk_UpperTriangularMatrix<ScalarType>::clear()
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
