/*
 * DOTk_KrylovSolverDataMng.cpp
 *
 *  Created on: Nov 3, 2014
 *      Author: Miguel A. Aguilo Valentin
 */

#include "DOTk_Primal.hpp"
#include "DOTk_MultiVector.hpp"
#include "DOTk_MultiVector.cpp"
#include "DOTk_OptimizationDataMng.hpp"
#include "DOTk_OrthogonalProjection.hpp"
#include "DOTk_LeftPreconditioner.hpp"
#include "DOTk_RightPreconditioner.hpp"
#include "DOTk_KrylovSolverDataMng.hpp"

namespace dotk
{

DOTk_KrylovSolverDataMng::DOTk_KrylovSolverDataMng
(const std::shared_ptr<dotk::DOTk_Primal> & aPrimal,
 const std::shared_ptr<dotk::DOTk_LinearOperator> & aLinearOperator) :
        m_MaxNumSolverItr(200),
        m_SolverType(dotk::types::KRYLOV_SOLVER_DISABLED),
        m_Solution(),
        m_Residual(),
        m_FirstSolution(),
        m_PreviousSolution(),
        m_MatrixTimesVector(),
        m_LinearOperator(aLinearOperator)
{
    this->initialize(aPrimal);
}

DOTk_KrylovSolverDataMng::~DOTk_KrylovSolverDataMng()
{
}

void DOTk_KrylovSolverDataMng::setSolverType(dotk::types::krylov_solver_t aType)
{
    m_SolverType = aType;
}

dotk::types::krylov_solver_t DOTk_KrylovSolverDataMng::getSolverType() const
{
    return (m_SolverType);
}

void DOTk_KrylovSolverDataMng::setMaxNumSolverItr(size_t aMaxNumIterations)
{
    m_MaxNumSolverItr = aMaxNumIterations;
}

size_t DOTk_KrylovSolverDataMng::getMaxNumSolverItr() const
{
    return (m_MaxNumSolverItr);
}

const std::shared_ptr<dotk::DOTk_LinearOperator> & DOTk_KrylovSolverDataMng::getLinearOperator() const
{
    return (m_LinearOperator);
}

const std::shared_ptr<dotk::Vector<Real> > & DOTk_KrylovSolverDataMng::getSolution() const
{
    return (m_Solution);
}

const std::shared_ptr<dotk::Vector<Real> > & DOTk_KrylovSolverDataMng::getFirstSolution() const
{
    return (m_FirstSolution);
}

const std::shared_ptr<dotk::Vector<Real> > & DOTk_KrylovSolverDataMng::getPreviousSolution() const
{
    return (m_PreviousSolution);
}

void DOTk_KrylovSolverDataMng::setResidual (const std::shared_ptr<dotk::Vector<Real> > & aVector)
{
    m_Residual->update(1., *aVector, 0.);
}

const std::shared_ptr<dotk::Vector<Real> > & DOTk_KrylovSolverDataMng::getResidual() const
{
    return (m_Residual);
}

const std::shared_ptr<dotk::Vector<Real> > & DOTk_KrylovSolverDataMng::getResidual(size_t aIndex) const
{
    std::perror("\n**** Unimplemented Function DOTk_KrylovSolverDataMng::getResidual(index). ABORT. ****\n");
    std::abort();
}

const std::shared_ptr<dotk::Vector<Real> > & DOTk_KrylovSolverDataMng::getMatrixTimesVector() const
{
    return (m_MatrixTimesVector);
}

const std::shared_ptr<dotk::Vector<Real> > & DOTk_KrylovSolverDataMng::getLeftPrecTimesVector() const
{
    std::perror("\n**** Unimplemented Function DOTk_KrylovSolverDataMng::getLeftPrecTimesVector. ABORT. ****\n");
    std::abort();
}

const std::shared_ptr<dotk::Vector<Real> > & DOTk_KrylovSolverDataMng::getLeftPrecTimesVector(size_t aIndex) const
{
    std::perror("\n**** Unimplemented Function DOTk_KrylovSolverDataMng::getLeftPrecTimesVector(index). ABORT. ****\n");
    std::abort();
}

const std::shared_ptr<dotk::Vector<Real> > & DOTk_KrylovSolverDataMng::getRightPrecTimesVector() const
{
    std::perror("\n**** Unimplemented Function DOTk_KrylovSolverDataMng::getRightPrecTimesVector. ABORT. ****\n");
    std::abort();
}

void DOTk_KrylovSolverDataMng::setProjection
(const std::shared_ptr<dotk::DOTk_OrthogonalProjection> & aProjection)
{
    std::perror("\n**** Unimplemented Function DOTk_KrylovSolverDataMng::setProjection. ABORT. ****\n");
    std::abort();
}

const std::shared_ptr<dotk::DOTk_OrthogonalProjection> &
DOTk_KrylovSolverDataMng::getProjection() const
{
    std::perror("\n**** Unimplemented Function DOTk_KrylovSolverDataMng::getProjection. ABORT. ****\n");
    std::abort();
}

void DOTk_KrylovSolverDataMng::setLeftPrec(const std::shared_ptr<dotk::DOTk_LeftPreconditioner> & aPreconditioner)
{
    std::perror("\n**** Unimplemented Function DOTk_KrylovSolverDataMng::setLeftPrec. ABORT. ****\n");
    std::abort();
}

const std::shared_ptr<dotk::DOTk_LeftPreconditioner> &
DOTk_KrylovSolverDataMng::getLeftPrec() const
{
    std::perror("\n**** Unimplemented Function DOTk_KrylovSolverDataMng::getLeftPrec. ABORT. ****\n");
    std::abort();
}

void DOTk_KrylovSolverDataMng::setRightPrec
(const std::shared_ptr<dotk::DOTk_RightPreconditioner> & aPreconditioner)
{
    std::perror("\n**** Unimplemented Function DOTk_KrylovSolverDataMng::setRightPrec. ABORT. ****\n");
    std::abort();
}

const std::shared_ptr<dotk::DOTk_RightPreconditioner> &
DOTk_KrylovSolverDataMng::getRightPrec() const
{
    std::perror("\n**** Unimplemented Function DOTk_KrylovSolverDataMng::getRightPrec. ABORT. ****\n");
    std::abort();
}

void DOTk_KrylovSolverDataMng::initialize(const std::shared_ptr<dotk::DOTk_Primal> & aPrimal)
{
    bool is_dual_allocated = aPrimal->dual().use_count() > 0;
    bool is_state_allocated = aPrimal->state().use_count() > 0;
    bool is_control_allocated = aPrimal->control().use_count() > 0;

    if( (is_dual_allocated == false) && (is_state_allocated == false) && (is_control_allocated == true) )
    {
        m_Solution = aPrimal->control()->clone();
        m_Residual = aPrimal->control()->clone();
        m_FirstSolution = aPrimal->control()->clone();
        m_PreviousSolution = aPrimal->control()->clone();
        m_MatrixTimesVector = aPrimal->control()->clone();
    }
    else
    {
        m_Solution = std::make_shared<dotk::DOTk_MultiVector<Real>>(*aPrimal);
        m_Solution->fill(0);
        m_Residual = std::make_shared<dotk::DOTk_MultiVector<Real>>(*aPrimal);
        m_Residual->fill(0);
        m_FirstSolution = std::make_shared<dotk::DOTk_MultiVector<Real>>(*aPrimal);
        m_FirstSolution->fill(0);
        m_PreviousSolution = std::make_shared<dotk::DOTk_MultiVector<Real>>(*aPrimal);
        m_PreviousSolution->fill(0);
        m_MatrixTimesVector = std::make_shared<dotk::DOTk_MultiVector<Real>>(*aPrimal);
        m_MatrixTimesVector->fill(0);
    }

    if( (is_dual_allocated == false) && (is_state_allocated == false) && (is_control_allocated == false) )
    {
        std::perror("\n**** DOTk ERROR in DOTk_KrylovSolverDataMng::initialize. User did not allocate data. ABORT. ****\n");
        std::abort();
    }
}

}
