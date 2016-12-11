/*
 * DOTk_SecondOrderOperator.hpp
 *
 *  Created on: Sep 6, 2014
 *      Author: Miguel A. Aguilo Valentin (maguilo@sandia.gov)
 */

#ifndef DOTK_SECONDORDEROPERATOR_HPP_
#define DOTK_SECONDORDEROPERATOR_HPP_

#include <vector>
#include <tr1/memory>

#include "DOTk_Types.hpp"

namespace dotk
{

class DOTk_OptimizationDataMng;

template<typename ScalarType>
class Vector;
template<typename ScalarType>
class matrix;

class DOTk_SecondOrderOperator
{
public:
    explicit DOTk_SecondOrderOperator(size_t max_secant_storage_ = 0);
    virtual ~DOTk_SecondOrderOperator();

    Int getNumUpdatesStored() const;
    void setNumUpdatesStored(size_t value_);
    Int getMaxNumSecantStorage() const;
    void setMaxNumSecantStorages(size_t value_);
    Int getNumOptimizationItrDone() const;
    void setNumOptimizationItrDone(size_t itr_);

    Real getDiagonalScaleFactor() const;
    void setDiagonalScaleFactor(Real value_);
    Real getLowerBoundOnDiagonalScaleFactor() const;
    void setLowerBoundOnDiagonalScaleFactor(Real value_);

    void setUpdateSecondOrderOperator(bool update_secant_storage_);
    bool updateSecondOrderOperator() const;
    void setSecantStorageFullFlag(bool is_secant_storage_full_);
    bool IsSecantStorageFull() const;

    dotk::types::hessian_t getHessianType() const;
    void setHessianType(dotk::types::hessian_t type_);
    dotk::types::invhessian_t getInvHessianType() const;
    void setInvHessianType(dotk::types::invhessian_t type_);

    Real getBarzilaiBorweinStep(const std::tr1::shared_ptr<dotk::Vector<Real> > & dprimal_,
                                const std::tr1::shared_ptr<dotk::Vector<Real> > & dgrad_);

    void computeDeltaPrimal(const std::tr1::shared_ptr<dotk::Vector<Real> > & new_primal_,
                            const std::tr1::shared_ptr<dotk::Vector<Real> > & old_primal_,
                            std::tr1::shared_ptr<dotk::Vector<Real> > & delta_primal_);
    void computeDeltaGradient(const std::tr1::shared_ptr<dotk::Vector<Real> > & new_gradient_,
                              const std::tr1::shared_ptr<dotk::Vector<Real> > & old_gradient_,
                              std::tr1::shared_ptr<dotk::Vector<Real> > & delta_gradient_);

    void updateSecantStorage(const std::tr1::shared_ptr<dotk::Vector<Real> > & dprimal_,
                             const std::tr1::shared_ptr<dotk::Vector<Real> > & dgrad_,
                             std::tr1::shared_ptr<dotk::matrix<Real> > & dprimal_storage_,
                             std::tr1::shared_ptr<dotk::matrix<Real> > & dgrad_storage_);
    void updateSecantStorage(const std::tr1::shared_ptr<dotk::Vector<Real> > & dprimal_,
                             const std::tr1::shared_ptr<dotk::Vector<Real> > & dgrad_,
                             std::vector<Real> & rho_storage_,
                             std::tr1::shared_ptr<dotk::matrix<Real> > & dprimal_storage_,
                             std::tr1::shared_ptr<dotk::matrix<Real> > & dgrad_storage_);

    virtual void apply(const std::tr1::shared_ptr<dotk::DOTk_OptimizationDataMng> & mng_,
                       const std::tr1::shared_ptr<dotk::Vector<Real> > & vec_,
                       const std::tr1::shared_ptr<dotk::Vector<Real> > & matrix_times_vec_);

private:
    size_t m_NumUpdatesStored;
    size_t m_MaxNumSecantStorages;
    size_t m_NumOptimizationItrDone;
    Real m_DiagonalScaleFactor;
    Real m_LowerBoundOnDiagonalScaleFactor;

    bool m_IsSecantStorageFull;
    bool m_UpdateSecondOrderOperator;
    dotk::types::hessian_t m_HessianType;
    dotk::types::invhessian_t m_InvHessianType;

private:
    DOTk_SecondOrderOperator(const dotk::DOTk_SecondOrderOperator &);
    DOTk_SecondOrderOperator operator=(const dotk::DOTk_SecondOrderOperator &);
};

}

#endif /* DOTK_SECONDORDEROPERATOR_HPP_ */
