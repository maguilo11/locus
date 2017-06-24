/*
 * DOTk_Variable.cpp
 *
 *  Created on: Feb 18, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#include <sstream>
#include <iostream>

#include "vector.hpp"
#include "DOTk_Variable.hpp"
#include "DOTk_SerialArray.hpp"
#include "DOTk_SerialVector.hpp"

namespace dotk
{

DOTk_Variable::DOTk_Variable(dotk::types::variable_t type_) :
        m_Type(type_),
        m_Data(),
        m_LowerBound(),
        m_UpperBound()
{
}

DOTk_Variable::DOTk_Variable(dotk::types::variable_t type_, const dotk::Vector<Real> & data_) :
        m_Type(type_),
        m_Data(data_.clone()),
        m_LowerBound(),
        m_UpperBound()
{
    this->initialize(data_);
}

DOTk_Variable::DOTk_Variable(dotk::types::variable_t type_,
                             const dotk::Vector<Real> & data_,
                             const dotk::Vector<Real> & lower_bound_,
                             const dotk::Vector<Real> & upper_bound_) :
        m_Type(type_),
        m_Data(data_.clone()),
        m_LowerBound(lower_bound_.clone()),
        m_UpperBound(upper_bound_.clone())
{
    this->initialize(data_, lower_bound_, upper_bound_);
}

DOTk_Variable::~DOTk_Variable()
{
}

size_t DOTk_Variable::size() const
{
    return (m_Data->size());
}

dotk::types::variable_t DOTk_Variable::type() const
{
    return (m_Type);
}

const std::shared_ptr<dotk::Vector<Real> > & DOTk_Variable::data() const
{
    return (m_Data);
}

void DOTk_Variable::setLowerBound(Real value_)
{
    this->checkData();
    if(m_LowerBound.use_count() == 0)
    {
        m_LowerBound.reset();
        m_LowerBound = m_Data->clone();
    }
    m_LowerBound->fill(value_);
}

void DOTk_Variable::setLowerBound(const dotk::Vector<Real> & lower_bound_)
{
    this->checkData();
    if(m_LowerBound.use_count() == 0)
    {
        m_LowerBound.reset();
        m_LowerBound = m_Data->clone();
    }
    m_LowerBound->update(1., lower_bound_, 0.);
}

const std::shared_ptr<dotk::Vector<Real> > & DOTk_Variable::lowerBound() const
{
    return (m_LowerBound);
}

void DOTk_Variable::setUpperBound(Real value_)
{
    this->checkData();
    if(m_UpperBound.use_count() == 0)
    {
        m_UpperBound.reset();
        m_UpperBound = m_Data->clone();
    }
    m_UpperBound->fill(value_);
}

void DOTk_Variable::setUpperBound(const dotk::Vector<Real> & upper_bound_)
{
    this->checkData();
    if(m_UpperBound.use_count() == 0)
    {
        m_UpperBound.reset();
        m_UpperBound = m_Data->clone();
    }
    m_UpperBound->update(1., upper_bound_, 0.);
}

const std::shared_ptr<dotk::Vector<Real> > & DOTk_Variable::upperBound() const
{
    return (m_UpperBound);
}

void DOTk_Variable::allocate(const dotk::Vector<Real> & input_)
{
    m_Data = input_.clone();
}

void DOTk_Variable::allocateSerialArray(size_t size_, Real value_)
{
    m_Data = std::make_shared<dotk::StdArray<Real>>(size_, value_);
}

void DOTk_Variable::allocateSerialVector(size_t size_, Real value_)
{
    m_Data = std::make_shared<dotk::StdVector<Real>>(size_, value_);
}

void DOTk_Variable::checkData()
{
    if(m_Data.use_count() < 1)
    {
        std::ostringstream msg;
        msg << "\n**** ERROR IN: " << __FILE__ << ", LINE: " << __LINE__ << ", -> DATA is NULL, ABORT. ****\n";
        std::perror(msg.str().c_str());
        std::abort();
    }
}

void DOTk_Variable::initialize(const dotk::Vector<Real> & data_)
{
    m_Data->update(1., data_, 0.);
}

void DOTk_Variable::initialize(const dotk::Vector<Real> & data_,
                               const dotk::Vector<Real> & lower_bound_,
                               const dotk::Vector<Real> & upper_bound_)
{
    try
    {
        size_t data_size = data_.size();
        size_t lower_bound_size = lower_bound_.size();
        if(lower_bound_size != data_size)
        {
            std::ostringstream msg;
            msg << "ERROR IN: " << __FILE__ << ", LINE: " << __LINE__
                    << ", -> DIMENSION MISMATCH BETWEEN LOWER BOUND AND DATA CONTAINERS."
                    << " DATA CONTAINER DIMENSION IS EQUAL TO " << data_size
                    << " AND LOWER BOUND CONTAINER DIMENSION IS EQUAL TO " << lower_bound_size << ": ABORT\n\n";
            throw msg.str().c_str();
        }

        size_t upper_bound_size = upper_bound_.size();
        if(upper_bound_size != data_size)
        {
            std::ostringstream msg;
            msg << "ERROR IN: " << __FILE__ << ", LINE: " << __LINE__
                    << ", -> DIMENSION MISMATCH BETWEEN UPPER BOUND AND DATA CONTAINERS."
                    << " DATA CONTAINER DIMENSION IS EQUAL TO " << data_size
                    << " AND UPPER BOUND CONTAINER DIMENSION IS EQUAL TO " << upper_bound_size << ": ABORT\n\n";
            throw msg.str().c_str();
        }

        for(size_t i = 0; i < lower_bound_.size(); ++ i)
        {
            if(lower_bound_[i] > upper_bound_[i])
            {
                std::ostringstream msg;
                msg << "ERROR IN: " << __FILE__ << ", LINE:" << __LINE__ << ", -> LOWER BOUND AT INDEX " << i
                        << " EXCEEDS UPPER BOUND WITH VALUE " << lower_bound_[i] << ". UPPER BOUND AT INDEX " << i
                        << " HAS A VALUE OF " << upper_bound_[i] << ": ABORT\n\n";
                throw msg.str().c_str();
            }
        }

        m_Data->update(1., data_, 0.);
        m_LowerBound->update(1., lower_bound_, 0.);
        m_UpperBound->update(1., upper_bound_, 0.);
    }
    catch(const char *error_msg)
    {
        std::cout << error_msg << std::flush;
    }
}

}

