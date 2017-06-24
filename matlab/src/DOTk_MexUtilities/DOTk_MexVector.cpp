/*
 * DOTk_MexVector.cpp
 *
 *  Created on: Dec 23, 2016
 *      Author: Miguel A. Aguilo Valentin
 */

#include <cmath>
#include <limits>
#include <cassert>
#include <sstream>
#include <iostream>

#include "DOTk_MexVector.hpp"

namespace dotk
{

MexVector::MexVector(size_t length_, double a_initial_value) :
        m_Data(mxCreateDoubleMatrix(1u, length_, mxREAL))
{
    this->fill(a_initial_value);
}

MexVector::MexVector(const mxArray* array_) :
        m_Data(mxDuplicateArray(array_))
{
}

MexVector::~MexVector()
{
    if(m_Data != nullptr)
    {
        mxDestroyArray(m_Data);
    }
    m_Data = nullptr;
}

void MexVector::scale(const double & input_)
{
    double* my_data = this->data();
    size_t my_num_elements = mxGetNumberOfElements(m_Data);
    for(size_t index = 0; index < my_num_elements; ++index)
    {
        my_data[index] = input_ * my_data[index];
    }
}

void MexVector::elementWiseMultiplication(const dotk::Vector<double> & input_)
{
    assert(input_.size() == this->size());

    double* my_data = this->data();
    size_t my_num_elements = mxGetNumberOfElements(m_Data);
    for(size_t index = 0; index < my_num_elements; ++index)
    {
        my_data[index] = input_[index] * my_data[index];
    }
}

void MexVector::update(const double & alpha__, const dotk::Vector<double> & input_, const double & beta_)
{
    assert(input_.size() == this->size());

    double* my_data = mxGetPr(m_Data);
    size_t my_num_elements = mxGetNumberOfElements(m_Data);
    for(size_t index = 0; index < my_num_elements; ++index)
    {
        my_data[index] = beta_ * my_data[index] + alpha__ * input_[index];
    }
}

double MexVector::max() const
{
    double max_value = std::numeric_limits<double>::min();
    size_t my_length = this->size();
    double* my_data = mxGetPr(m_Data);
    for(size_t index = 0; index < my_length; ++index)
    {
        if(my_data[index] > max_value)
        {
            max_value = my_data[index];
        }
    }
    return (max_value);
}

double MexVector::min() const
{
    double min_value = std::numeric_limits<double>::max();
    size_t my_length = this->size();
    double* my_data = mxGetPr(m_Data);
    for(size_t index = 0; index < my_length; ++index)
    {
        if(my_data[index] < min_value)
        {
            min_value = my_data[index];
        }
    }
    return (min_value);
}

void MexVector::abs()
{
    size_t my_length = this->size();
    double* my_data = mxGetPr(m_Data);
    for(size_t index = 0; index < my_length; ++index)
    {
        my_data[index] = my_data[index] < static_cast<double>(0.) ? -(my_data[index]) : my_data[index];
    }
}

double MexVector::sum() const
{
    double my_sum = 0.;
    size_t my_length = this->size();
    double* my_data = mxGetPr(m_Data);
    for(size_t index = 0; index < my_length; ++index)
    {
        my_sum += my_data[index];
    }
    return (my_sum);
}

double MexVector::dot(const dotk::Vector<double> & input_) const
{
    assert(input_.size() == this->size());

    double my_inner_product = 0.;
    size_t my_length = this->size();
    double* my_data = mxGetPr(m_Data);
    for(size_t index = 0; index < my_length; ++index)
    {
        my_inner_product += my_data[index] * input_[index];
    }
    return (my_inner_product);
}

double MexVector::norm() const
{
    double my_norm = this->dot(*this);
    my_norm = std::sqrt(my_norm);
    return (my_norm);
}

void MexVector::fill(const double & input_)
{
    size_t my_length = this->size();
    double* my_data = mxGetPr(m_Data);
    for(size_t index = 0; index < my_length; ++index)
    {
        my_data[index] = input_;
    }
}

size_t MexVector::size() const
{
    return (mxGetNumberOfElements(m_Data));
}

std::shared_ptr<dotk::Vector<double> > MexVector::clone() const
{
    double tValue = 0;
    size_t tSize = this->size();
    std::shared_ptr<dotk::MexVector> tVector = std::make_shared<dotk::MexVector>(tSize, tValue);
    return (tVector);
}

double & MexVector::operator [](size_t index_)
{
    assert(index_ < this->size());

    double* my_data = mxGetPr(m_Data);
    return (my_data[index_]);
}

const double & MexVector::operator [](size_t index_) const
{
    assert(index_ < this->size());

    double* my_data = mxGetPr(m_Data);
    return (my_data[index_]);
}

double* MexVector::data()
{
    return (mxGetPr(m_Data));
}

const double* MexVector::data() const
{
    return (mxGetPr(m_Data));
}

mxArray* MexVector::array()
{
    return (m_Data);
}

const mxArray* MexVector::array() const
{
    return (m_Data);
}

void MexVector::setMxArray(const mxArray* input_)
{
    try
    {
        size_t input_length = mxGetNumberOfElements(input_);
        if(input_length <= 0)
        {
            std::ostringstream error;
            error << "\nERROR IN: " << __FILE__ << ", LINE: " << __LINE__ << " -> Input input array has size <= 0.\n";
            throw error.str().c_str();
        }

        assert(input_length == this->size());

        double* my_data = this->data();
        const int my_length = this->size();
        const double* input_data = mxGetPr(input_);
        for(int index = 0; index < my_length; ++index)
        {
            my_data[index] = input_data[index];
        }
    }
    catch(const char *error_msg)
    {
        std::cout << error_msg << std::flush;
    }
}

}
