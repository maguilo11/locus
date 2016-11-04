/*
 * DOTk_SerialArray.cpp
 *
 *  Created on: Apr 4, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#include <algorithm>
#include <tr1/memory>
#include "DOTk_SerialArray.hpp"

namespace dotk
{

namespace serial
{

template<class Type>
array<Type>::array(int dim_, Type value_) :
        m_LocalDim(dim_),
        m_Data(new Type[dim_])
{
    this->fill(value_);
}

template<class Type>
array<Type>::~array()
{
    delete[] m_Data;
    m_Data = NULL;
}

template<class Type>
void array<Type>::scale(const Type & alpha_)
{
    size_t dim = this->size();
    for(size_t i = 0; i < dim; ++i)
    {
        m_Data[i] = alpha_ * m_Data[i];
    }
}

template<class Type>
void array<Type>::cwiseProd(const dotk::vector<Type> & input_)
{
    size_t dim = this->size();
    assert(dim == input_.size());
    for(size_t i = 0; i < dim; ++i)
    {
        m_Data[i] = m_Data[i] * input_[i];
    }
}

template<class Type>
void array<Type>::axpy(const Type & alpha_, const dotk::vector<Type> & input_)
{
    size_t dim = this->size();
    assert(dim == input_.size());
    for(size_t i = 0; i < dim; ++i)
    {
        m_Data[i] = alpha_ * input_[i] + m_Data[i];
    }
}

template<class Type>
Type array<Type>::max() const
{
    size_t dim = this->size();

    Type result = *std::max_element(m_Data, m_Data + dim);

    return (result);
}

template<class Type>
Type array<Type>::min() const
{
    size_t dim = this->size();

    Type result = *std::min_element(m_Data, m_Data + dim);

    return (result);
}

template<class Type>
void array<Type>::abs()
{
    size_t dim = this->size();
    for(size_t index = 0; index < dim; ++index)
    {
        m_Data[index] = m_Data[index] < static_cast<Type>(0.) ? -(m_Data[index]) : m_Data[index];
    }
}

template<class Type>
Type array<Type>::sum() const
{
    Type result = 0.;
    size_t dim = this->size();

    for(size_t i = 0; i < dim; ++i)
    {
        result += m_Data[i];
    }

    return (result);
}

template<class Type>
Type array<Type>::dot(const dotk::vector<Type> & input_) const
{
    Type result = 0;
    size_t dim = this->size();
    assert(dim == input_.size());

    for(size_t i = 0; i < dim; ++i)
    {
        result += m_Data[i] * input_[i];
    }

    return (result);
}

template<class Type>
Type array<Type>::norm() const
{
    Type value = 0.;
    size_t dim = this->size();

    for(size_t i = 0; i < dim; ++i)
    {
        value += m_Data[i] * m_Data[i];
    }

    Type result = sqrt(value);

    return (result);
}

template<class Type>
void array<Type>::fill(const Type & value_)
{
    size_t dim = this->size();
    for(size_t i = 0; i < dim; ++i)
    {
        m_Data[i] = value_;
    }
}

template<class Type>
void array<Type>::copy(const dotk::vector<Type> & input_)
{
    size_t dim = this->size();
    assert(dim == input_.size());
    for(size_t i = 0; i < dim; ++i)
    {
        m_Data[i] = input_[i];
    }
}

template<class Type>
void array<Type>::gather(Type* input_) const
{
    size_t dim = this->size();
    for(size_t i = 0; i < dim; ++i)
    {
        input_[i] = m_Data[i];
    }
}

template<class Type>
size_t array<Type>::size() const
{
    return (m_LocalDim);
}

template<class Type>
std::tr1::shared_ptr<dotk::vector<Type> > array<Type>::clone() const
{
    std::tr1::shared_ptr<dotk::serial::array<Type> > vec(new dotk::serial::array<Type>(m_LocalDim));
    return (vec);
}

template<class Type>
Type & array<Type>::operator [](size_t index_)
{
    return (m_Data[index_]);
}

template<class Type>
const Type & array<Type>::operator [](size_t index_) const
{
    return (m_Data[index_]);
}

template<class Type>
dotk::types::container_t array<Type>::type() const
{
    return (dotk::types::SERIAL_ARRAY);
}

template<class Type>
size_t array<Type>::rank() const
{
    return (0);
}

}

}
