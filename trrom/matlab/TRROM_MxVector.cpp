/*
 * TRROM_MxVector.cpp
 *
 *  Created on: Nov 21, 2016
 *      Author: maguilo
 */

#include <cmath>
#include <limits>
#include <cassert>
#include "TRROM_MxVector.hpp"

namespace trrom
{

MxVector::MxVector(int a_length, double a_initial_value) :
        m_Data(mxCreateDoubleMatrix_700(1, a_length, mxREAL))
{
    this->fill(a_initial_value);
}

MxVector::MxVector(const mxArray* array_) :
        m_Data(mxDuplicateArray(array_))
{
}

MxVector::~MxVector()
{
    mxDestroyArray(m_Data);
}

void MxVector::scale(const double & a_alpha)
{
    double* data = mxGetPr(m_Data);
    int my_length = mxGetN(m_Data);
    for(int index = 0; index < my_length; ++index)
    {
        data[index] = a_alpha * data[index];
    }
}

void MxVector::elementWiseMultiplication(const trrom::Vector<double> & a_input)
{
    assert(a_input.size() == this->size());

    int my_length = mxGetN(m_Data);
    double* my_data = mxGetPr(m_Data);
    for(int index = 0; index < my_length; ++index)
    {
        my_data[index] = a_input[index] * my_data[index];
    }
}

void MxVector::update(const double & a_alpha, const trrom::Vector<double> & a_input, const double & a_beta)
{
    assert(a_input.size() == this->size());

    int my_length = mxGetN(m_Data);
    double* my_data = mxGetPr(m_Data);
    for(int index = 0; index < my_length; ++index)
    {
        my_data[index] = a_beta * my_data[index] + a_alpha * a_input[index];
    }
}

double MxVector::max(int & a_index) const
{
    double max_value = std::numeric_limits<double>::min();
    int my_length = this->size();
    double* my_data = mxGetPr(m_Data);
    for(int index = 0; index < my_length; ++index)
    {
        if(my_data[index] > max_value)
        {
            max_value = my_data[index];
            a_index = index;
        }
    }
    return (max_value);
}

double MxVector::min(int & a_index) const
{
    double min_value = std::numeric_limits<double>::max();
    int my_length = this->size();
    double* my_data = mxGetPr(m_Data);
    for(int index = 0; index < my_length; ++index)
    {
        if(my_data[index] < min_value)
        {
            min_value = my_data[index];
            a_index = index;
        }
    }
    return (min_value);
}

void MxVector::modulus()
{
    int my_length = this->size();
    double* my_data = mxGetPr(m_Data);
    for(int index = 0; index < my_length; ++index)
    {
        my_data[index] = my_data[index] < static_cast<double>(0.) ? -(my_data[index]) : my_data[index];
    }
}

double MxVector::sum() const
{
    double my_sum = 0.;
    int my_length = this->size();
    double* my_data = mxGetPr(m_Data);
    for(int index = 0; index < my_length; ++index)
    {
        my_sum += my_data[index];
    }
    return (my_sum);
}

double MxVector::dot(const trrom::Vector<double> & a_input) const
{
    assert(a_input.size() == this->size());

    double my_inner_product = 0.;
    int my_length = this->size();
    double* my_data = mxGetPr(m_Data);
    for(int index = 0; index < my_length; ++index)
    {
        my_inner_product += my_data[index] * a_input[index];
    }
    return (my_inner_product);
}

double MxVector::norm() const
{
    double my_norm = this->dot(*this);
    my_norm = std::sqrt(my_norm);
    return (my_norm);
}

void MxVector::fill(const double & a_input)
{
    int my_length = this->size();
    double* my_data = mxGetPr(m_Data);
    for(int index = 0; index < my_length; ++index)
    {
        my_data[index] = a_input;
    }
}

int MxVector::size() const
{
    return (mxGetN(m_Data));
}

std::tr1::shared_ptr<trrom::Vector<double> > MxVector::create(int a_length) const
{
    std::tr1::shared_ptr<trrom::MxVector> this_copy;
    if(a_length == 0)
    {
        int this_length = this->size();
        this_copy.reset(new trrom::MxVector(this_length));
    }
    else
    {
        this_copy.reset(new trrom::MxVector(a_length));
    }
    return (this_copy);
}

double & MxVector::operator [](int a_index)
{
    double* my_data = mxGetPr(m_Data);
    return (my_data[a_index]);
}

const double & MxVector::operator [](int a_index) const
{
    double* my_data = mxGetPr(m_Data);
    return (my_data[a_index]);
}

double* MxVector::data()
{
    return (mxGetPr(m_Data));
}

const double* MxVector::data() const
{
    return (mxGetPr(m_Data));
}

mxArray* MxVector::array()
{
    return (m_Data);
}

const mxArray* MxVector::array() const
{
    return (m_Data);
}

void MxVector::setMxArray(const mxArray* input_)
{
    const int input_length = mxGetN(input_);
    assert(input_length == this->size());

    double* my_data = this->data();
    const double* input_data = mxGetPr(input_);
    for(int index = 0; index < input_length; ++index)
    {
        my_data[index] = input_data[index];
    }
}

}
