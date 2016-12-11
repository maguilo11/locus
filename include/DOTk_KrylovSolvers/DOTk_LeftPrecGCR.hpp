/*
 * DOTk_LeftPrecGCR.hpp
 *
 *  Created on: Nov 3, 2014
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_LEFTPRECGCR_HPP_
#define DOTK_LEFTPRECGCR_HPP_

#include <vector>
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

class DOTk_LeftPrecGCR : public dotk::DOTk_KrylovSolver
{
public:
    explicit DOTk_LeftPrecGCR(const std::tr1::shared_ptr<dotk::DOTk_KrylovSolverDataMng> & solver_mng_);
    DOTk_LeftPrecGCR(const std::tr1::shared_ptr<dotk::DOTk_Primal> & variable_,
                     const std::tr1::shared_ptr<dotk::DOTk_LinearOperator> & linear_operator_,
                     size_t max_num_itr_ = 200);
    virtual ~DOTk_LeftPrecGCR();

    void initialize(const std::tr1::shared_ptr<dotk::Vector<Real> > & rhs_vec_,
                    const std::tr1::shared_ptr<dotk::DOTk_KrylovSolverStoppingCriterion> & criterion_,
                    const std::tr1::shared_ptr<dotk::DOTk_OptimizationDataMng> & opt_mng_);
    void pgcr(const std::tr1::shared_ptr<dotk::Vector<Real> > & rhs_vec_,
              const std::tr1::shared_ptr<dotk::DOTk_KrylovSolverStoppingCriterion> & criterion_,
              const std::tr1::shared_ptr<dotk::DOTk_OptimizationDataMng> & opt_mng_);

    virtual void setMaxNumKrylovSolverItr(size_t itr_);
    virtual const std::tr1::shared_ptr<dotk::DOTk_KrylovSolverDataMng> & getDataMng() const;
    virtual const std::tr1::shared_ptr<dotk::DOTk_LinearOperator> & getLinearOperator() const;
    virtual const std::tr1::shared_ptr<dotk::Vector<Real> > & getDescentDirection();
    virtual void solve(const std::tr1::shared_ptr<dotk::Vector<Real> > & rhs_vec_,
                       const std::tr1::shared_ptr<dotk::DOTk_KrylovSolverStoppingCriterion> & criterion_,
                       const std::tr1::shared_ptr<dotk::DOTk_OptimizationDataMng> & opt_mng_);

private:
    std::tr1::shared_ptr<dotk::DOTk_KrylovSolverDataMng> m_DataMng;

    std::vector<Real> mBetaCoefficients;
    std::vector< std::tr1::shared_ptr<dotk::Vector<Real> > > mConjugateDirectionStorage;
    std::vector< std::tr1::shared_ptr<dotk::Vector<Real> > > mLinearOperatorTimesConjugateDirStorage;

private:
    void initialize(const std::tr1::shared_ptr<dotk::Vector<Real> > vec_template_);
    void updateBetaCoefficientStorage(size_t current_itr_);
    void updateConjugateDirectionStorage(size_t current_itr_);
    void updateLinearOperatorTimesConjugateDirStorage(size_t current_itr_);

private:
    DOTk_LeftPrecGCR(const dotk::DOTk_LeftPrecGCR &);
    dotk::DOTk_LeftPrecGCR & operator=(const dotk::DOTk_LeftPrecGCR &);
};

}

#endif /* DOTK_LEFTPRECGCR_HPP_ */
