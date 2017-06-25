/*
 * TRROM_Variable.cpp
 *
 *  Created on: Feb 18, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#include <sstream>

#include "TRROM_Vector.hpp"
#include "TRROM_Variable.hpp"

namespace trrom
{

Variable::Variable(trrom::types::variable_t type_) :
        m_Type(type_),
        m_Data(),
        m_LowerBound(),
        m_UpperBound()
{
}

Variable::Variable(trrom::types::variable_t type_, const trrom::Vector<double> & data_) :
        m_Type(type_),
        m_Data(data_.create()),
        m_LowerBound(),
        m_UpperBound()
{
    this->initialize(data_);
}

Variable::Variable(trrom::types::variable_t type_,
                   const trrom::Vector<double> & data_,
                   const trrom::Vector<double> & lower_bound_,
                   const trrom::Vector<double> & upper_bound_) :
        m_Type(type_),
        m_Data(data_.create()),
        m_LowerBound(lower_bound_.create()),
        m_UpperBound(upper_bound_.create())
{
    this->initialize(data_, lower_bound_, upper_bound_);
}

Variable::~Variable()
{
}

int Variable::size() const
{
    return (m_Data->size());
}

trrom::types::variable_t Variable::type() const
{
    return (m_Type);
}

const std::shared_ptr<trrom::Vector<double> > & Variable::data() const
{
    return (m_Data);
}

void Variable::setLowerBound(double value_)
{
    this->checkData();
    if(m_LowerBound.use_count() == 0)
    {
        m_LowerBound.reset();
        m_LowerBound = m_Data->create();
    }
    m_LowerBound->fill(value_);
}

void Variable::setLowerBound(const trrom::Vector<double> & lower_bound_)
{
    this->checkData();
    if(m_LowerBound.use_count() == 0)
    {
        m_LowerBound.reset();
        m_LowerBound = m_Data->create();
    }
    m_LowerBound->update(1., lower_bound_, 0.);
}

const std::shared_ptr<trrom::Vector<double> > & Variable::lowerBound() const
{
    return (m_LowerBound);
}

void Variable::setUpperBound(double value_)
{
    this->checkData();
    if(m_UpperBound.use_count() == 0)
    {
        m_UpperBound.reset();
        m_UpperBound = m_Data->create();
    }
    m_UpperBound->fill(value_);
}

void Variable::setUpperBound(const trrom::Vector<double> & upper_bound_)
{
    this->checkData();
    if(m_UpperBound.use_count() == 0)
    {
        m_UpperBound.reset();
        m_UpperBound = m_Data->create();
    }
    m_UpperBound->update(1., upper_bound_, 0.);
}

const std::shared_ptr<trrom::Vector<double> > & Variable::upperBound() const
{
    return (m_UpperBound);
}

void Variable::checkData()
{
    try
    {
        if(m_Data.use_count() <= 0)
        {
            std::ostringstream error;
            error << "\nERROR IN: " << __FILE__ << ", LINE: " << __LINE__ << ", FUNCTION: " << __FUNCTION__
                    << ", MESSAGE: Data structure is not allocated\n";
            throw error.str().c_str();
        }
    }
    catch(const char *error_msg)
    {
        std::cout << error_msg << std::flush;
    }
}

void Variable::initialize(const trrom::Vector<double> & data_)
{
    m_Data->update(1., data_, 0.);
}

void Variable::initialize(const trrom::Vector<double> & data_,
                          const trrom::Vector<double> & lower_bound_,
                          const trrom::Vector<double> & upper_bound_)
{
    try
    {
        int data_size = data_.size();
        int lower_bound_size = lower_bound_.size();
        if(lower_bound_size != data_size)
        {
            std::ostringstream msg;
            msg << "\n\n**** ERROR IN: " << __FILE__ << ", LINE: " << __LINE__ << ", FUNCTION: " << __FUNCTION__
                    << ", MESSAGE: DIMENSION MISMATCH BETWEEN LOWER BOUND AND DATA CONTAINERS."
                    << " DATA CONTAINER DIMENSION IS EQUAL TO " << data_size
                    << " AND LOWER BOUND CONTAINER DIMENSION IS EQUAL TO " << lower_bound_size << ": ABORT ****\n\n";
            throw msg.str().c_str();
        }

        int upper_bound_size = upper_bound_.size();
        if(upper_bound_size != data_size)
        {
            std::ostringstream msg;
            msg << "\n\n**** ERROR IN: " << __FILE__ << ", LINE: " << __LINE__ << ", FUNCTION: " << __FUNCTION__
                    << ", MESSAGE: DIMENSION MISMATCH BETWEEN UPPER BOUND AND DATA CONTAINERS."
                    << " DATA CONTAINER DIMENSION IS EQUAL TO " << data_size
                    << " AND UPPER BOUND CONTAINER DIMENSION IS EQUAL TO " << upper_bound_size << ": ABORT ****\n\n";
            throw msg.str().c_str();
        }

        for(int index = 0; index < lower_bound_.size(); ++ index)
        {
            if(lower_bound_[index] > upper_bound_[index])
            {
                std::ostringstream msg;
                msg << "\n\n**** ERROR IN: " << __FILE__ << ", LINE: " << __LINE__ << ", FUNCTION: " << __FUNCTION__
                        << ", MESSAGE: LOWER BOUND AT INDEX " << index << " EXCEEDS UPPER BOUND WITH VALUE "
                        << lower_bound_[index] << ". UPPER BOUND AT INDEX " << index << " HAS A VALUE OF "
                        << upper_bound_[index] << ": ABORT ****\n\n";
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

