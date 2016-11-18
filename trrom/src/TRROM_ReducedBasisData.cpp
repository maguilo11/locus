/*
 * TRROM_ReducedBasisData.cpp
 *
 *  Created on: Aug 17, 2016
 */

#include <sstream>

#include "TRROM_Matrix.hpp"
#include "TRROM_Vector.hpp"
#include "TRROM_ReducedBasisData.hpp"

namespace trrom
{

ReducedBasisData::ReducedBasisData() :
        trrom::Data(),
        m_Fidelity(trrom::types::HIGH_FIDELITY),
        m_ReducedDualSolution(),
        m_DualSnapshotEnsemble(),
        m_DualOrthonormalBasis(),
        m_ReducedDualLeftHandSide(),
        m_ReducedDualRightHandSide(),
        m_ReducedStateSolution(),
        m_StateSnapshotEnsemble(),
        m_StateOrthonormalBasis(),
        m_ReducedStateLeftHandSide(),
        m_ReducedStateRightHandSide(),
        m_LeftHandSideSnapshot(),
        m_RightHandSideSnapshot(),
        m_LeftHandSideActiveIndices(),
        m_LeftHandSideSnapshotEnsemble(),
        m_LeftHandSideOrthonormalBasis(),
        m_LeftHandSideDeimCoefficients()
{
}

ReducedBasisData::~ReducedBasisData()
{
}

void ReducedBasisData::allocateReducedDualSolution(const trrom::Vector<double> & input_)
{
    m_ReducedDualSolution = input_.create();
}

void ReducedBasisData::allocateDualSnapshotEnsemble(const trrom::Matrix<double> & input_)
{
    m_DualSnapshotEnsemble = input_.create();
}

void ReducedBasisData::allocateDualOrthonormalBasis(const trrom::Matrix<double> & input_)
{
    m_DualOrthonormalBasis = input_.create();
}

void ReducedBasisData::allocateReducedDualLeftHandSide(const trrom::Matrix<double> & input_)
{
    m_ReducedDualLeftHandSide = input_.create();
}

void ReducedBasisData::allocateReducedDualRightHandSide(const trrom::Vector<double> & input_)
{
    m_ReducedDualRightHandSide = input_.create();
}

void ReducedBasisData::allocateReducedStateSolution(const trrom::Vector<double> & input_)
{
    m_ReducedStateSolution = input_.create();
}

void ReducedBasisData::allocateStateSnapshotEnsemble(const trrom::Matrix<double> & input_)
{
    m_StateSnapshotEnsemble = input_.create();
}

void ReducedBasisData::allocateStateOrthonormalBasis(const trrom::Matrix<double> & input_)
{
    m_StateOrthonormalBasis = input_.create();
}

void ReducedBasisData::allocateReducedStateLeftHandSide(const trrom::Matrix<double> & input_)
{
    m_ReducedStateLeftHandSide = input_.create();
}

void ReducedBasisData::allocateReducedStateRightHandSide(const trrom::Vector<double> & input_)
{
    m_ReducedStateRightHandSide = input_.create();
}

void ReducedBasisData::allocateLeftHandSideSnapshot(const trrom::Vector<double> & input_)
{
    m_LeftHandSideSnapshot = input_.create();
}

void ReducedBasisData::allocateRightHandSideSnapshot(const trrom::Vector<double> & input_)
{
    m_RightHandSideSnapshot = input_.create();
}

void ReducedBasisData::allocateLeftHandSideSnapshotEnsemble(const trrom::Matrix<double> & input_)
{
    m_LeftHandSideSnapshotEnsemble = input_.create();
}

void ReducedBasisData::allocateLeftHandSideOrthonormalBasis(const trrom::Matrix<double> & input_)
{
    m_LeftHandSideOrthonormalBasis = input_.create();
}

void ReducedBasisData::allocateLeftHandSideDeimCoefficients(const trrom::Vector<double> & input_)
{
    m_LeftHandSideDeimCoefficients = input_.create();
}

trrom::types::fidelity_t ReducedBasisData::fidelity() const
{
    return (m_Fidelity);
}

void ReducedBasisData::fidelity(trrom::types::fidelity_t input_)
{
    m_Fidelity = input_;
}

const trrom::Vector<double> & ReducedBasisData::getLeftHandSideSnapshot() const
{
    return (*m_LeftHandSideSnapshot);
}

void ReducedBasisData::setLeftHandSideSnapshot(const trrom::Vector<double> & input_)
{
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

const trrom::Vector<double> & ReducedBasisData::getRightHandSideSnapshot() const
{
    return (*m_RightHandSideSnapshot);
}

void ReducedBasisData::setRightHandSideSnapshot(const trrom::Vector<double> & input_)
{
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

const trrom::Vector<double> & ReducedBasisData::getLeftHandSideActiveIndices() const
{
    return (*m_LeftHandSideActiveIndices);
}

void ReducedBasisData::setLeftHandSideActiveIndices(const trrom::Vector<double> & input_)
{
    m_LeftHandSideActiveIndices = input_.create();
    m_LeftHandSideActiveIndices->update(1., input_, 0.);
}

std::tr1::shared_ptr<trrom::Vector<double> > ReducedBasisData::createReducedDualSolutionCopy(int global_dim_) const
{
    try
    {
        if(m_ReducedDualSolution.use_count() <= 0)
        {
            std::ostringstream error;
            error << "\nERROR IN: " << __FILE__ << ", LINE: "
                  << __LINE__ << " -> Reduced dual solution data structure is not allocated\n";
            throw error.str().c_str();
        }
    }
    catch(const char *error_msg)
    {
        std::cout << error_msg << std::flush;
    }
    return (m_ReducedDualSolution->create(global_dim_));
}

std::tr1::shared_ptr<trrom::Matrix<double> > ReducedBasisData::createDualOrthonormalBasisCopy(int global_num_rows_,
                                                                                              int global_num_cols_) const
{
    try
    {
        if(m_DualOrthonormalBasis.use_count() <= 0)
        {
            std::ostringstream error;
            error << "\nERROR IN: " << __FILE__ << ", LINE: "
                  << __LINE__ << " -> Dual orthonormal basis data structure is not allocated\n";
            throw error.str().c_str();
        }
    }
    catch(const char *error_msg)
    {
        std::cout << error_msg << std::flush;
    }
    return (m_DualOrthonormalBasis->create(global_num_rows_, global_num_cols_));
}

std::tr1::shared_ptr<trrom::Matrix<double> > ReducedBasisData::createReducedDualLeftHandSideCopy(int global_num_rows_,
                                                                                                 int global_num_cols_) const
{
    try
    {
        if(m_ReducedDualLeftHandSide.use_count() <= 0)
        {
            std::ostringstream error;
            error << "\nERROR IN: " << __FILE__ << ", LINE: "
                  << __LINE__ << " -> Reduced dual left hand side matrix data structure is not allocated\n";
            throw error.str().c_str();
        }
    }
    catch(const char *error_msg)
    {
        std::cout << error_msg << std::flush;
    }
    return (m_ReducedDualLeftHandSide->create(global_num_rows_, global_num_cols_));
}

std::tr1::shared_ptr<trrom::Vector<double> > ReducedBasisData::createReducedDualRightHandSideCopy(int global_dim_) const
{
    try
    {
        if(m_ReducedDualRightHandSide.use_count() <= 0)
        {
            std::ostringstream error;
            error << "\nERROR IN: " << __FILE__ << ", LINE: "
                  << __LINE__ << " -> Reduced dual right hand side data structure is not allocated\n";
            throw error.str().c_str();
        }
    }
    catch(const char *error_msg)
    {
        std::cout << error_msg << std::flush;
    }
    return (m_ReducedDualRightHandSide->create(global_dim_));
}

std::tr1::shared_ptr<trrom::Vector<double> > ReducedBasisData::createReducedStateSolutionCopy(int global_dim_) const
{
    try
    {
        if(m_ReducedStateSolution.use_count() <= 0)
        {
            std::ostringstream error;
            error << "\nERROR IN: " << __FILE__ << ", LINE: "
                  << __LINE__ << " -> Reduced state solution data structure is not allocated\n";
            throw error.str().c_str();
        }
    }
    catch(const char *error_msg)
    {
        std::cout << error_msg << std::flush;
    }
    return (m_ReducedStateSolution->create(global_dim_));
}

std::tr1::shared_ptr<trrom::Matrix<double> > ReducedBasisData::createStateOrthonormalBasisCopy(int global_num_rows_,
                                                                                               int global_num_cols_) const
{
    try
    {
        if(m_StateOrthonormalBasis.use_count() <= 0)
        {
            std::ostringstream error;
            error << "\nERROR IN: " << __FILE__ << ", LINE: "
                  << __LINE__ << " -> State orthonormal basis data structure is not allocated\n";
            throw error.str().c_str();
        }
    }
    catch(const char *error_msg)
    {
        std::cout << error_msg << std::flush;
    }
    return (m_StateOrthonormalBasis->create(global_num_rows_, global_num_cols_));
}

std::tr1::shared_ptr<trrom::Matrix<double> > ReducedBasisData::createReducedStateLeftHandSideCopy(int global_num_rows_,
                                                                                                  int global_num_cols_) const
{
    try
    {
        if(m_ReducedStateLeftHandSide.use_count() <= 0)
        {
            std::ostringstream error;
            error << "\nERROR IN: " << __FILE__ << ", LINE: "
                  << __LINE__ << " -> Reduced state left hand side matrix data structure is not allocated\n";
            throw error.str().c_str();
        }
    }
    catch(const char *error_msg)
    {
        std::cout << error_msg << std::flush;
    }
    return (m_ReducedStateLeftHandSide->create(global_num_rows_, global_num_cols_));
}

std::tr1::shared_ptr<trrom::Vector<double> > ReducedBasisData::createReducedStateRightHandSideCopy(int global_dim_) const
{
    try
    {
        if(m_ReducedStateRightHandSide.use_count() <= 0)
        {
            std::ostringstream error;
            error << "\nERROR IN: " << __FILE__ << ", LINE: "
                  << __LINE__ << " -> Reduced state right hand side data structure is not allocated\n";
            throw error.str().c_str();
        }
    }
    catch(const char *error_msg)
    {
        std::cout << error_msg << std::flush;
    }
    return (m_ReducedStateRightHandSide->create(global_dim_));
}

std::tr1::shared_ptr<trrom::Vector<double> > ReducedBasisData::createLeftHandSideSnapshotCopy(int global_dim_) const
{
    try
    {
        if(m_LeftHandSideSnapshot.use_count() <= 0)
        {
            std::ostringstream error;
            error << "\nERROR IN: " << __FILE__ << ", LINE: "
                  << __LINE__ << " -> Left hand side snapshot data structure is not allocated\n";
            throw error.str().c_str();
        }
    }
    catch(const char *error_msg)
    {
        std::cout << error_msg << std::flush;
    }
    return (m_LeftHandSideSnapshot->create(global_dim_));
}

std::tr1::shared_ptr<trrom::Vector<double> > ReducedBasisData::createRightHandSideSnapshotCopy(int global_dim_) const
{
    try
    {
        if(m_RightHandSideSnapshot.use_count() <= 0)
        {
            std::ostringstream error;
            error << "\nERROR IN: " << __FILE__ << ", LINE: "
                  << __LINE__ << " -> Right hand side snapshot data structure is not allocated\n";
            throw error.str().c_str();
        }
    }
    catch(const char *error_msg)
    {
        std::cout << error_msg << std::flush;
    }
    return (m_RightHandSideSnapshot->create(global_dim_));
}

std::tr1::shared_ptr<trrom::Matrix<double> > ReducedBasisData::createLeftHandSideOrthonormalBasisCopy(int global_num_rows_,
                                                                                                      int global_num_cols_) const
{
    try
    {
        if(m_LeftHandSideOrthonormalBasis.use_count() <= 0)
        {
            std::ostringstream error;
            error << "\nERROR IN: " << __FILE__ << ", LINE: "
                  << __LINE__ << " -> Left hand side orthonormal basis data structure is not allocated\n";
            throw error.str().c_str();
        }
    }
    catch(const char *error_msg)
    {
        std::cout << error_msg << std::flush;
    }
    return (m_LeftHandSideOrthonormalBasis->create(global_num_rows_, global_num_cols_));
}

std::tr1::shared_ptr<trrom::Vector<double> > ReducedBasisData::createLeftHandSideDeimCoefficientsCopy(int global_dim_) const
{
    try
    {
        if(m_LeftHandSideDeimCoefficients.use_count() <= 0)
        {
            std::ostringstream error;
            error << "\nERROR IN: " << __FILE__ << ", LINE: "
                  << __LINE__ << " -> Left hand side DEIM coefficients data structure is not allocated\n";
            throw error.str().c_str();
        }
    }
    catch(const char *error_msg)
    {
        std::cout << error_msg << std::flush;
    }
    return (m_LeftHandSideDeimCoefficients->create(global_dim_));
}

}
