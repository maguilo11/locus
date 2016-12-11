/*
 * DOTk_SecondOrderOperator.cpp
 *
 *  Created on: Oct 2, 2014
 *      Author: Miguel A. Aguilo Valentin
 */

#include <cmath>

#include "vector.hpp"
#include "matrix.hpp"
#include "DOTk_OptimizationDataMng.hpp"
#include "DOTk_SecondOrderOperator.hpp"

using namespace dotk;

DOTk_SecondOrderOperator::DOTk_SecondOrderOperator(size_t max_secant_storage_) :
        m_NumUpdatesStored(0),
        m_MaxNumSecantStorages(max_secant_storage_),
        m_NumOptimizationItrDone(0),
        m_DiagonalScaleFactor(0.1),
        m_LowerBoundOnDiagonalScaleFactor(1e-8),
        m_IsSecantStorageFull(false),
        m_UpdateSecondOrderOperator(true),
        m_HessianType(dotk::types::HESSIAN_DISABLED),
        m_InvHessianType(dotk::types::INV_HESS_DISABLED)
{
}

DOTk_SecondOrderOperator::~DOTk_SecondOrderOperator()
{
}

Int DOTk_SecondOrderOperator::getNumUpdatesStored() const
{
    return (m_NumUpdatesStored);
}

void DOTk_SecondOrderOperator::setNumUpdatesStored(size_t value_)
{
    m_NumUpdatesStored = value_;
}

Int DOTk_SecondOrderOperator::getMaxNumSecantStorage() const
{
    return (m_MaxNumSecantStorages);
}

void DOTk_SecondOrderOperator::setMaxNumSecantStorages(size_t value_)
{
    m_MaxNumSecantStorages = value_;
}

Int DOTk_SecondOrderOperator::getNumOptimizationItrDone() const
{
    return (m_NumOptimizationItrDone);
}

void DOTk_SecondOrderOperator::setNumOptimizationItrDone(size_t itr_)
{
    m_NumOptimizationItrDone = itr_;
}

Real DOTk_SecondOrderOperator::getDiagonalScaleFactor() const
{
    return (m_DiagonalScaleFactor);
}

void DOTk_SecondOrderOperator::setDiagonalScaleFactor(Real value_)
{
    m_DiagonalScaleFactor = value_;
}

Real DOTk_SecondOrderOperator::getLowerBoundOnDiagonalScaleFactor() const
{
    return (m_LowerBoundOnDiagonalScaleFactor);
}

void DOTk_SecondOrderOperator::setLowerBoundOnDiagonalScaleFactor(Real value_)
{
    m_LowerBoundOnDiagonalScaleFactor = value_;
}

void DOTk_SecondOrderOperator::setUpdateSecondOrderOperator(bool update_second_order_operator_)
{
    m_UpdateSecondOrderOperator = update_second_order_operator_;
}

bool DOTk_SecondOrderOperator::updateSecondOrderOperator() const
{
    return (m_UpdateSecondOrderOperator);
}

void DOTk_SecondOrderOperator::setSecantStorageFullFlag(bool is_secant_storage_full_)
{
    m_IsSecantStorageFull = is_secant_storage_full_;
}

bool DOTk_SecondOrderOperator::IsSecantStorageFull() const
{
    return (m_IsSecantStorageFull);
}

dotk::types::hessian_t DOTk_SecondOrderOperator::getHessianType() const
{
    return (m_HessianType);
}

void DOTk_SecondOrderOperator::setHessianType(dotk::types::hessian_t type_)
{
    m_HessianType = type_;
}

dotk::types::invhessian_t DOTk_SecondOrderOperator::getInvHessianType() const
{
    return (m_InvHessianType);
}

void DOTk_SecondOrderOperator::setInvHessianType(dotk::types::invhessian_t type_)
{
    m_InvHessianType = type_;
}

Real DOTk_SecondOrderOperator::getBarzilaiBorweinStep(const std::tr1::shared_ptr<dotk::Vector<Real> > & dprimal_,
                                                      const std::tr1::shared_ptr<dotk::Vector<Real> > & dgrad_)
{
    Real step = 0.;
    if(this->getNumOptimizationItrDone() % 2 == 0)
    {
        // even m_OptimizationItr
        step = dprimal_->dot(*dgrad_) / dgrad_->dot(*dgrad_);
    }
    else
    {
        // odd m_OptimizationItr
        step = dprimal_->dot(*dprimal_) / dprimal_->dot(*dgrad_);
    }
    return (step);
}

void DOTk_SecondOrderOperator::computeDeltaPrimal(const std::tr1::shared_ptr<dotk::Vector<Real> > & new_primal_,
                                                  const std::tr1::shared_ptr<dotk::Vector<Real> > & old_primal_,
                                                  std::tr1::shared_ptr<dotk::Vector<Real> > & delta_primal_)
{
    if(this->updateSecondOrderOperator() == true)
    {
        delta_primal_->update(1., *new_primal_, 0.);
        delta_primal_->update(-1., *old_primal_, 1.);
    }
}

void DOTk_SecondOrderOperator::computeDeltaGradient(const std::tr1::shared_ptr<dotk::Vector<Real> > & new_gradient_,
                                                    const std::tr1::shared_ptr<dotk::Vector<Real> > & old_gradient_,
                                                    std::tr1::shared_ptr<dotk::Vector<Real> > & delta_gradient_)
{
    if(this->updateSecondOrderOperator() == true)
    {
        delta_gradient_->update(1., *new_gradient_, 0.);
        delta_gradient_->update(-1., *old_gradient_, 1.);
    }
}

void DOTk_SecondOrderOperator::updateSecantStorage(const std::tr1::shared_ptr<dotk::Vector<Real> > & dprimal_,
                                                   const std::tr1::shared_ptr<dotk::Vector<Real> > & dgrad_,
                                                   std::tr1::shared_ptr<dotk::matrix<Real> > & dprimal_storage_,
                                                   std::tr1::shared_ptr<dotk::matrix<Real> > & dgrad_storage_)
{
    /// Update limited memory sorage. The work set is later used to approximate the limited memory \n
    /// second-order operator \n
    /// In: \n
    ///     delta_primal_ = difference between current and previous primal solution, unchanged on exist. \n
    ///     std::tr1::shared_ptr<dotk::Vector<Real> > \n
    ///     delta_gradient_ = difference between current and previous gradient_, unchanged on exist . \n
    ///     std::tr1::shared_ptr<dotk::Vector<Real> > \n
    /// In/Out: \n
    ///     delta_primal_storage_ = limited limited memory storage of deltaPrimal \n
    ///     std::tr1::shared_ptr<dotk::matrix<Real> > \n
    ///     delta_gradient_storage_ = limited limited memory storage of deltaGradient \n
    ///     std::tr1::shared_ptr<dotk::matrix<Real> > \n
    /// \n
    bool is_this_the_first_optimization_iteration = this->getNumOptimizationItrDone() == 1 ? true : false;
    if(is_this_the_first_optimization_iteration || !(this->updateSecondOrderOperator()))
    {
        return;
    }
    int updates = this->getNumUpdatesStored();
    // Update limited memory work set work-set
    this->setSecantStorageFullFlag((updates == this->getMaxNumSecantStorage() ? true : false));
    if(this->IsSecantStorageFull() == true)
    {
        // DO IF: Number of updates is equal to the maximum number of previous solutions stored.
        updates = static_cast<size_t>(std::floor(updates / 2.));
        for(int index = 0; index < updates; ++index)
        {
            dprimal_storage_->basis(index)->update(1., *dprimal_storage_->basis(updates + index), 0.);
            dgrad_storage_->basis(index)->update(1., *dgrad_storage_->basis(updates + index), 0.);
        }
        dprimal_storage_->basis(updates)->update(1., *dprimal_, 0.);
        dgrad_storage_->basis(updates)->update(1., *dgrad_, 0.);
        ++updates;
        this->setNumUpdatesStored(updates);
        this->setUpdateSecondOrderOperator(false);
    }
    else
    {
        // DO IF: Number of updates is less to the maximum number of previous solutions stored.
        dprimal_storage_->basis(updates)->update(1., *dprimal_, 0.);
        dgrad_storage_->basis(updates)->update(1., *dgrad_, 0.);
        ++updates;
        this->setNumUpdatesStored(updates);
        this->setUpdateSecondOrderOperator(false);
    }
}

void DOTk_SecondOrderOperator::updateSecantStorage(const std::tr1::shared_ptr<dotk::Vector<Real> > & dprimal_,
                                                   const std::tr1::shared_ptr<dotk::Vector<Real> > & dgrad_,
                                                   std::vector<Real> & rho_storage_,
                                                   std::tr1::shared_ptr<dotk::matrix<Real> > & dprimal_storage_,
                                                   std::tr1::shared_ptr<dotk::matrix<Real> > & dgrad_storage_)
{
    /// Update limited memory sorage. The work set is later used to approximate the limited memory \n
    /// second-order operator \n
    /// In: \n
    ///     delta_primal_ = difference between current and previous primal solution, unchanged on exist. \n
    ///     std::tr1::shared_ptr<dotk::Vector<Real> > \n
    ///     delta_gradient_ = difference between current and previous gradient_, unchanged on exist . \n
    ///     std::tr1::shared_ptr<dotk::Vector<Real> > \n
    /// In/Out: \n
    ///     rho_storage_ = limited memory storage of inner product between deltaGradient and deltaPrimal. \n
    ///     std::tr1::shared_ptr<dotk::Vector<Real> > \n
    ///     delta_primal_storage_ = limited limited memory storage of deltaPrimal \n
    ///     std::tr1::shared_ptr<dotk::matrix<Real> > \n
    ///     delta_gradient_storage_ = limited limited memory storage of deltaGradient \n
    ///     std::tr1::shared_ptr<dotk::matrix<Real> > \n
    /// \n
    bool is_this_the_first_optimization_iteration = this->getNumOptimizationItrDone() == 1 ? true : false;
    if(is_this_the_first_optimization_iteration || !(this->updateSecondOrderOperator()))
    {
        return;
    }
    int updates = this->getNumUpdatesStored();
    // Update limited memory work set work-set
    Real kappa = 0.;
    this->setSecantStorageFullFlag((updates == this->getMaxNumSecantStorage() ? true : false));
    if(this->IsSecantStorageFull() == true)
    {
        // DO IF: Number of updates is equal to the maximum number of previous solutions stored.
        updates = static_cast<size_t>(std::floor(updates / 2.));
        for(int index = 0; index < updates; ++index)
        {
            dprimal_storage_->basis(index)->update(1., *dprimal_storage_->basis(updates + index), 0.);
            dgrad_storage_->basis(index)->update(1., *dgrad_storage_->basis(updates + index), 0.);
            rho_storage_[index] = rho_storage_[updates + index];
        }
        dprimal_storage_->basis(updates)->update(1., *dprimal_, 0.);
        dgrad_storage_->basis(updates)->update(1., *dgrad_, 0.);
        kappa = static_cast<Real>(1.0)
                / dgrad_storage_->basis(updates)->dot(*dprimal_storage_->basis(updates));
        rho_storage_[updates] = kappa;
        ++updates;
        this->setNumUpdatesStored(updates);
        this->setUpdateSecondOrderOperator(false);
    }
    else
    {
        // DO IF: Number of updates is less to the maximum number of previous solutions stored.
        dprimal_storage_->basis(updates)->update(1., *dprimal_, 0.);
        dgrad_storage_->basis(updates)->update(1., *dgrad_, 0.);
        kappa = static_cast<Real>(1.0) / dgrad_storage_->basis(updates)->dot(*dprimal_storage_->basis(updates));
        rho_storage_[updates] = kappa;
        ++updates;
        this->setNumUpdatesStored(updates);
        this->setUpdateSecondOrderOperator(false);
    }
}

void DOTk_SecondOrderOperator::apply(const std::tr1::shared_ptr<dotk::DOTk_OptimizationDataMng> & mng_,
                                     const std::tr1::shared_ptr<dotk::Vector<Real> > & vec_,
                                     const std::tr1::shared_ptr<dotk::Vector<Real> > & matrix_times_vec_)
{
}
