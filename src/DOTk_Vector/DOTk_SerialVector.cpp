/*
 * DOTk_SerialVector.cpp
 *
 *  Created on: Apr 4, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#include <vector>
#include <algorithm>
#include <tr1/memory>
#include "DOTk_SerialVector.hpp"

namespace dotk
{

namespace serial
{

template<class Type>
vector<Type>::vector(std::vector<Type> & data_) :
        m_Data(data_)
{
}

template<class Type>
vector<Type>::vector(int dim_, Type value_) :
        m_Data(std::vector<Type>(dim_, value_))
{
}

template<class Type>
vector<Type>::~vector()
{
}

template<class Type>
void vector<Type>::scale(const Type & alpha_)
{
    size_t dim = this->size();
    for(size_t i = 0; i < dim; ++i)
    {
        m_Data[i] = alpha_ * m_Data[i];
    }
}

template<class Type>
void vector<Type>::cwiseProd(const dotk::vector<Type> & input_)
{
    size_t dim = this->size();
    assert(dim == input_.size());
    for(size_t i = 0; i < dim; ++i)
    {
        m_Data[i] = m_Data[i] * input_[i];
    }
}

template<class Type>
void vector<Type>::axpy(const Type & alpha_, const dotk::vector<Type> & input_)
{
    size_t dim = this->size();
    assert(dim == input_.size());
    for(size_t i = 0; i < dim; ++i)
    {
        m_Data[i] = alpha_ * input_[i] + m_Data[i];
    }
}

template<class Type>
Type vector<Type>::max() const
{
    Type result = *std::max_element(m_Data.begin(), m_Data.end());

    return (result);
}

template<class Type>
Type vector<Type>::min() const
{
    Type result = *std::min_element(m_Data.begin(), m_Data.end());

    return (result);
}

template<class Type>
void vector<Type>::abs()
{
    size_t dim = this->size();
    for(size_t index = 0; index < dim; ++index)
    {
        m_Data.data()[index] =
                m_Data.data()[index] < static_cast<Type>(0.) ? -(m_Data.data()[index]) : m_Data.data()[index];
    }
}

template<class Type>
Type vector<Type>::sum() const
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
Type vector<Type>::dot(const dotk::vector<Type> & input_) const
{
    size_t dim = this->size();
    assert(dim == input_.size());

    Type result = 0.;
    for(size_t i = 0; i < dim; ++i)
    {
        result += m_Data[i] * input_[i];
    }

    return (result);
}

template<class Type>
Type vector<Type>::norm() const
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
void vector<Type>::fill(const Type & value_)
{
    size_t dim = this->size();
    for(size_t i = 0; i < dim; ++i)
    {
        m_Data[i] = value_;
    }
}

template<class Type>
void vector<Type>::copy(const dotk::vector<Type> & input_)
{
    size_t dim = this->size();
    assert(dim == input_.size());
    for(size_t i = 0; i < dim; ++i)
    {
        m_Data[i] = input_[i];
    }
}

template<class Type>
void vector<Type>::gather(Type* input_) const
{
    size_t dim = this->size();
    for(size_t i = 0; i < dim; ++i)
    {
        input_[i] = m_Data[i];
    }
}

template<class Type>
size_t vector<Type>::size() const
{
    size_t dim = m_Data.size();
    return (dim);
}

template<class Type>
std::tr1::shared_ptr< dotk::vector<Type> > vector<Type>::clone() const
{
    size_t dim = this->size();
    std::tr1::shared_ptr<dotk::serial::vector<Type> > vec(new dotk::serial::vector<Type>(dim, 0.));
    return (vec);
}

template<class Type>
Type & vector<Type>::operator [](size_t index_)
{
    return (m_Data.operator [](index_));
}

template<class Type>
const Type & vector<Type>::operator [](size_t index_) const
{
    return (m_Data.operator [](index_));
}

template<class Type>
dotk::types::container_t vector<Type>::type() const
{
    return (dotk::types::SERIAL_VECTOR);
}

template<class Type>
size_t vector<Type>::rank() const
{
    return (0);
}

}

}
