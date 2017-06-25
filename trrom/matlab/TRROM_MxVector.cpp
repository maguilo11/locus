/*
 * TRROM_MxVector.cpp
 *
 *  Created on: Nov 21, 2016
 *      Author: maguilo
 */

#include <cmath>
#include <limits>
#include <cassert>
#include <sstream>
#include "TRROM_MxVector.hpp"

namespace trrom
{

MxVector::MxVector(int length_, double a_initial_value) :
        m_Data(mxCreateDoubleMatrix_700(1, length_, mxREAL))
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

void MxVector::scale(const double & input_)
{
    double* my_data = this->data();
    int my_num_elements = mxGetNumberOfElements(m_Data);
    for(int index = 0; index < my_num_elements; ++index)
    {
        my_data[index] = input_ * my_data[index];
    }
}

void MxVector::elementWiseMultiplication(const trrom::Vector<double> & input_)
{
    assert(input_.size() == this->size());

    double* my_data = this->data();
    int my_num_elements = mxGetNumberOfElements(m_Data);
    for(int index = 0; index < my_num_elements; ++index)
    {
        my_data[index] = input_[index] * my_data[index];
    }
}

void MxVector::update(const double & alpha__, const trrom::Vector<double> & input_, const double & beta_)
{
    assert(input_.size() == this->size());

    double* my_data = mxGetPr(m_Data);
    int my_num_elements = mxGetNumberOfElements(m_Data);
    for(int index = 0; index < my_num_elements; ++index)
    {
        my_data[index] = beta_ * my_data[index] + alpha__ * input_[index];
    }
}

double MxVector::max(int & index_) const
{
    double max_value = std::numeric_limits<double>::min();
    int my_length = this->size();
    double* my_data = mxGetPr(m_Data);
    for(int index = 0; index < my_length; ++index)
    {
        if(my_data[index] > max_value)
        {
            max_value = my_data[index];
            index_ = index;
        }
    }
    return (max_value);
}

double MxVector::min(int & index_) const
{
    double min_value = std::numeric_limits<double>::max();
    int my_length = this->size();
    double* my_data = mxGetPr(m_Data);
    for(int index = 0; index < my_length; ++index)
    {
        if(my_data[index] < min_value)
        {
            min_value = my_data[index];
            index_ = index;
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

double MxVector::dot(const trrom::Vector<double> & input_) const
{
    assert(input_.size() == this->size());

    double my_inner_product = 0.;
    int my_length = this->size();
    double* my_data = mxGetPr(m_Data);
    for(int index = 0; index < my_length; ++index)
    {
        my_inner_product += my_data[index] * input_[index];
    }
    return (my_inner_product);
}

double MxVector::norm() const
{
    double my_norm = this->dot(*this);
    my_norm = std::sqrt(my_norm);
    return (my_norm);
}

void MxVector::fill(const double & input_)
{
    int my_length = this->size();
    double* my_data = mxGetPr(m_Data);
    for(int index = 0; index < my_length; ++index)
    {
        my_data[index] = input_;
    }
}

int MxVector::size() const
{
    return (mxGetNumberOfElements(m_Data));
}

std::shared_ptr<trrom::Vector<double> > MxVector::create(int length_) const
{
    std::shared_ptr<trrom::MxVector> this_copy;
    if(length_ == 0)
    {
        int this_length = this->size();
        this_copy = std::make_shared<trrom::MxVector>(this_length);
    }
    else
    {
        this_copy = std::make_shared<trrom::MxVector>(length_);
    }
    return (this_copy);
}

double & MxVector::operator [](int index_)
{
    assert(index_ < this->size());

    double* my_data = mxGetPr(m_Data);
    return (my_data[index_]);
}

const double & MxVector::operator [](int index_) const
{
    assert(index_ < this->size());

    double* my_data = mxGetPr(m_Data);
    return (my_data[index_]);
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
    try
    {
        int input_length = mxGetNumberOfElements(input_);
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
