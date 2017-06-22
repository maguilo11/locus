/*
 * DOTk_LeftPrecCG.hpp
 *
 *  Created on: Nov 3, 2014
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_LEFTPRECCG_HPP_
#define DOTK_LEFTPRECCG_HPP_

#include "DOTk_KrylovSolver.hpp"

namespace dotk
{

class DOTk_Primal;
class DOTk_KrylovSolver;
class DOTk_KrylovSolverDataMng;
class DOTk_OptimizationDataMng;
class DOTk_KrylovSolverStoppingCriterion;

template<typename ScalarType>
class Vector;

class DOTk_LeftPrecCG : public dotk::DOTk_KrylovSolver
{
public:
    explicit DOTk_LeftPrecCG(const std::shared_ptr<dotk::DOTk_KrylovSolverDataMng> & solver_mng_);
    DOTk_LeftPrecCG(const std::shared_ptr<dotk::DOTk_Primal> & primal_,
                    const std::shared_ptr<dotk::DOTk_LinearOperator> & linear_operator_);
    virtual ~DOTk_LeftPrecCG();

    void initialize(const std::shared_ptr<dotk::Vector<Real> > & rhs_vec_,
                    const std::shared_ptr<dotk::DOTk_KrylovSolverStoppingCriterion> & criterion_,
                    const std::shared_ptr<dotk::DOTk_OptimizationDataMng> & opt_mng_);
    void pcg(const std::shared_ptr<dotk::Vector<Real> > & rhs_vec_,
             const std::shared_ptr<dotk::DOTk_KrylovSolverStoppingCriterion> & criterion_,
             const std::shared_ptr<dotk::DOTk_OptimizationDataMng> & opt_mng_);

    virtual void setMaxNumKrylovSolverItr(size_t itr_);
    virtual const std::shared_ptr<dotk::DOTk_KrylovSolverDataMng> & getDataMng() const;
    virtual const std::shared_ptr<dotk::DOTk_LinearOperator> & getLinearOperator() const;
    virtual const std::shared_ptr<dotk::Vector<Real> > & getDescentDirection();
    virtual void solve(const std::shared_ptr<dotk::Vector<Real> > & rhs_vec_,
                       const std::shared_ptr<dotk::DOTk_KrylovSolverStoppingCriterion> & criterion_,
                       const std::shared_ptr<dotk::DOTk_OptimizationDataMng> & opt_mng_);

private:
    std::shared_ptr<dotk::DOTk_KrylovSolverDataMng> m_DataMng;
    std::shared_ptr<dotk::Vector<Real> > m_ConjugateDirection;

private:
    DOTk_LeftPrecCG(const dotk::DOTk_LeftPrecCG &);
    dotk::DOTk_LeftPrecCG & operator=(const dotk::DOTk_LeftPrecCG &);
};

}

#endif /* DOTK_LEFTPRECCG_HPP_ */
