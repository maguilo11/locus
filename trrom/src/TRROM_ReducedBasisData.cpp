/*
 * TRROM_ReducedBasisData.cpp
 *
 *  Created on: Aug 17, 2016
 */

#include <sstream>

#include "TRROM_Vector.hpp"
#include "TRROM_ReducedBasisData.hpp"

namespace trrom
{

ReducedBasisData::ReducedBasisData() :
        trrom::Data(),
        m_Fidelity(trrom::types::HIGH_FIDELITY),
        m_LeftHandSideSnapshot(),
        m_RightHandSideSnapshot(),
        m_LeftHandSideActiveIndices()
{
}

ReducedBasisData::~ReducedBasisData()
{
}

void ReducedBasisData::allocateLeftHandSideSnapshot(const trrom::Vector<double> & input_)
{
    try
    {
        if(input_.size() <= 0)
        {
            std::ostringstream error;
            error << "\nERROR IN: " << __FILE__ << ", LINE: " << __LINE__ << ", FUNCTION:" << __FUNCTION__
                    << " -> Input left hand side vector's length <= 0.\n";
            throw error.str().c_str();
        }
        m_LeftHandSideSnapshot = input_.create();
    }
    catch(const char *error_msg)
    {
        std::cout << error_msg << std::flush;
    }
}

void ReducedBasisData::allocateRightHandSideSnapshot(const trrom::Vector<double> & input_)
{
    try
    {
        if(input_.size() <= 0)
        {
            std::ostringstream error;
            error << "\nERROR IN: " << __FILE__ << ", LINE: " << __LINE__ << ", FUNCTION:" << __FUNCTION__
                    << " -> Input right hand side vector's length <= 0.\n";
            throw error.str().c_str();
        }
        m_RightHandSideSnapshot = input_.create();
    }
    catch(const char *error_msg)
    {
        std::cout << error_msg << std::flush;
    }
}

trrom::types::fidelity_t ReducedBasisData::fidelity() const
{
    return (m_Fidelity);
}

void ReducedBasisData::fidelity(trrom::types::fidelity_t input_)
{
    m_Fidelity = input_;
}

const std::shared_ptr<trrom::Vector<double> > & ReducedBasisData::getLeftHandSideSnapshot() const
{
    return (m_LeftHandSideSnapshot);
}

void ReducedBasisData::setLeftHandSideSnapshot(const trrom::Vector<double> & input_)
{
    try
    {
        if(input_.size() <= 0)
        {
            std::ostringstream error;
            error << "\nERROR IN: " << __FILE__ << ", LINE: " << __LINE__ << ", FUNCTION:" << __FUNCTION__
                    << " -> Input left hand side snapshot has size <= 0.\n";
            throw error.str().c_str();
        }

        if(m_LeftHandSideSnapshot.use_count() <= 0)
        {
            m_LeftHandSideSnapshot = input_.create();
            m_LeftHandSideSnapshot->update(1., input_, 0.);
        }
        else
        {
            m_LeftHandSideSnapshot->update(1., input_, 0.);
        }
    }
    catch(const char *error_msg)
    {
        std::cout << error_msg << std::flush;
    }
}

const std::shared_ptr<trrom::Vector<double> > & ReducedBasisData::getRightHandSideSnapshot() const
{
    return (m_RightHandSideSnapshot);
}

void ReducedBasisData::setRightHandSideSnapshot(const trrom::Vector<double> & input_)
{
    try
    {
        if(input_.size() <= 0)
        {
            std::ostringstream error;
            error << "\nERROR IN: " << __FILE__ << ", LINE: " << ", FUNCTION:" << __FUNCTION__
                  << __LINE__ << " -> Input right hand side snapshot has size <= 0.\n";
            throw error.str().c_str();
        }

        if(m_RightHandSideSnapshot.use_count() <= 0)
        {
            m_RightHandSideSnapshot = input_.create();
            m_RightHandSideSnapshot->update(1., input_, 0.);
        }
        else
        {
            m_RightHandSideSnapshot->update(1., input_, 0.);
        }
    }
    catch(const char *error_msg)
    {
        std::cout << error_msg << std::flush;
    }
}

const std::shared_ptr<trrom::Vector<double> > & ReducedBasisData::getLeftHandSideActiveIndices() const
{
    return (m_LeftHandSideActiveIndices);
}

void ReducedBasisData::setLeftHandSideActiveIndices(const trrom::Vector<double> & input_)
{
    try
    {
        if(input_.size() <= 0)
        {
            std::ostringstream error;
            error << "\nERROR IN: " << __FILE__ << ", LINE: " << __LINE__ << ", FUNCTION:" << __FUNCTION__
                    << ", MESSAGE: Input left hand side active indices array has size <= 0.\n";
            throw error.str().c_str();
        }
        m_LeftHandSideActiveIndices = input_.create();
        m_LeftHandSideActiveIndices->update(1., input_, 0.);
    }
    catch(const char *error_msg)
    {
        std::cout << error_msg << std::flush;
    }
}

}
