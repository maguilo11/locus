/*
 * DOTk_OmpArray.cpp
 *
 *  Created on: Apr 4, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#include <tr1/memory>
#include "DOTk_OmpArray.hpp"

namespace dotk
{

namespace omp
{

template<class Type>
array<Type>::array(int dim_, int num_threads_, Type value_) :
        m_NumDim(dim_),
        m_NumThreads(num_threads_),
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
    size_t i;
    size_t dim = this->size();

    int thread_count = this->threads();
# pragma omp parallel num_threads(thread_count) \
default( none ) \
shared ( dim, alpha_ ) \
private ( i )

# pragma omp for
    for(i = 0; i < dim; ++i)
    {
        m_Data[i] = alpha_ * m_Data[i];
    }
}

template<class Type>
void array<Type>::cwiseProd(const dotk::vector<Type> & input_)
{
    size_t i;
    size_t dim = this->size();
    assert(dim == input_.size());

    int thread_count = this->threads();
# pragma omp parallel num_threads(thread_count) \
default( none ) \
shared ( dim, input_ ) \
private ( i )

# pragma omp for
    for(i = 0; i < dim; ++i)
    {
        m_Data[i] = m_Data[i] * input_[i];
    }
}

template<class Type>
void array<Type>::axpy(const Type & alpha_, const dotk::vector<Type> & input_)
{
    size_t i;
    size_t dim = this->size();
    assert(dim == input_.size());

    int thread_count = this->threads();
# pragma omp parallel num_threads(thread_count) \
default( none ) \
shared ( dim, input_, alpha_ ) \
private ( i )

# pragma omp for
    for(i = 0; i < dim; ++i)
    {
        m_Data[i] = alpha_ * input_[i] + m_Data[i];
    }
}

template<class Type>
Type array<Type>::max() const
{
    size_t i;
    Type max_value = 0;
    size_t dim = this->size();

    int thread_count = this->threads();
# pragma omp parallel num_threads(thread_count) \
default( none ) \
shared ( dim, max_value ) \
private ( i )

# pragma omp for reduction( max : max_value )
    for(i = 0; i < dim; ++i)
    {
        if(m_Data[i] > max_value)
        {
            max_value = m_Data[i];
        }
    }

    return (max_value);
}

template<class Type>
Type array<Type>::min() const
{
    size_t i;
    Type min_value = 0.;
    size_t dim = this->size();

    int thread_count = this->threads();
# pragma omp parallel num_threads(thread_count) \
default( none ) \
shared ( dim, min_value ) \
private ( i )

# pragma omp for reduction( min : min_value )
    for(i = 0; i < dim; ++i)
    {
        if(m_Data[i] < min_value)
        {
            min_value = m_Data[i];
        }
    }

    return (min_value);
}

template<class Type>
void array<Type>::abs()
{
    size_t index;
    size_t dim = this->size();
    int thread_count = this->threads();

# pragma omp parallel num_threads(thread_count) \
default(none) \
shared ( dim ) \
private ( index )

# pragma omp for
    for(size_t index = 0; index < dim; ++index)
    {
        m_Data[index] = m_Data[index] < static_cast<Type>(0.) ? -(m_Data[index]) : m_Data[index];
    }
}

template<class Type>
Type array<Type>::sum() const
{
    size_t i;
    Type result = 0;
    size_t dim = this->size();

    int thread_count = this->threads();
# pragma omp parallel num_threads(thread_count) \
default(none) \
shared ( dim, result ) \
private ( i )

# pragma omp for reduction(+:result)
    for(i = 0; i < dim; ++i)
    {
        result += m_Data[i];
    }

    return (result);
}

template<class Type>
Type array<Type>::dot(const dotk::vector<Type> & input_) const
{
    size_t i;
    size_t dim = this->size();
    assert(dim == input_.size());
    Type result = 0.;

    int thread_count = this->threads();
# pragma omp parallel num_threads(thread_count) \
default( none ) \
shared ( dim, result, input_ ) \
private ( i )

# pragma omp for reduction(+:result)
    for(i = 0; i < dim; ++i)
    {
        result += (m_Data[i] * input_[i]);
    }

    return (result);
}

template<class Type>
Type array<Type>::norm() const
{
    size_t i;
    size_t dim = this->size();
    Type value = 0.;
    Type result = 0.;

    int thread_count = this->threads();
# pragma omp parallel num_threads(thread_count) \
default( none ) \
shared ( dim, value ) \
private ( i )

# pragma omp for reduction(+:value)
    for(i = 0; i < dim; ++i)
    {
        value += (m_Data[i] * m_Data[i]);
    }

    result = sqrt(value);

    return (result);
}

template<class Type>
void array<Type>::fill(const Type & value_)
{
    size_t i;
    size_t dim = this->size();

    int thread_count = this->threads();
# pragma omp parallel num_threads(thread_count) \
default( none ) \
shared ( dim, value_ ) \
private ( i )

# pragma omp for
    for(i = 0; i < dim; ++i)
    {
        m_Data[i] = value_;
    }
}

template<class Type>
void array<Type>::copy(const dotk::vector<Type> & input_)
{
    size_t i;
    size_t dim = this->size();
    assert(dim == input_.size());

    int thread_count = this->threads();
# pragma omp parallel num_threads(thread_count) \
default( none ) \
shared ( dim, input_ ) \
private ( i )

# pragma omp for
    for(i = 0; i < dim; ++i)
    {
        m_Data[i] = input_[i];
    }
}

template<class Type>
void array<Type>::gather(Type* input_) const
{
    size_t i;
    size_t dim = this->size();

    int thread_count = this->threads();
# pragma omp parallel num_threads(thread_count) \
default( none ) \
shared ( dim, input_ ) \
private ( i )

# pragma omp for
    for(i = 0; i < dim; ++i)
    {
        input_[i] = m_Data[i];
    }
}

template<class Type>
size_t array<Type>::size() const
{
    return (m_NumDim);
}

template<class Type>
std::tr1::shared_ptr<dotk::vector<Type> > array<Type>::clone() const
{
    size_t dim = this->size();
    int thread_count = this->threads();
    std::tr1::shared_ptr<dotk::omp::array<Type> > vec(new dotk::omp::array<Type>(dim, thread_count, 0.));
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
    return (dotk::types::OMP_ARRAY);
}

template<class Type>
size_t array<Type>::rank() const
{
    return (0);
}

template<class Type>
int array<Type>::threads() const
{
    return (m_NumThreads);
}

}

}
