/*
 * DOTk_LeftPrecCR.hpp
 *
 *  Created on: Nov 3, 2014
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_LEFTPRECCR_HPP_
#define DOTK_LEFTPRECCR_HPP_

#include "DOTk_KrylovSolver.hpp"

namespace dotk
{

class DOTk_Primal;
class DOTk_KrylovSolver;
class DOTk_KrylovSolverDataMng;
class DOTk_OptimizationDataMng;
class DOTk_KrylovSolverStoppingCriterion;

template<class Type>
class vector;

class DOTk_LeftPrecCR : public dotk::DOTk_KrylovSolver
{
public:
    explicit DOTk_LeftPrecCR(const std::tr1::shared_ptr<dotk::DOTk_KrylovSolverDataMng> & mng_);
    DOTk_LeftPrecCR(const std::tr1::shared_ptr<dotk::DOTk_Primal> & primal_,
                    const std::tr1::shared_ptr<dotk::DOTk_LinearOperator> & linear_operator_);
    virtual ~DOTk_LeftPrecCR();

    void initialize(const std::tr1::shared_ptr<dotk::vector<Real> > & rhs_vec_,
                    const std::tr1::shared_ptr<dotk::DOTk_KrylovSolverStoppingCriterion> & criterion_,
                    const std::tr1::shared_ptr<dotk::DOTk_OptimizationDataMng> & opt_mng_);
    void pcr(const std::tr1::shared_ptr<dotk::vector<Real> > & rhs_vec_,
             const std::tr1::shared_ptr<dotk::DOTk_KrylovSolverStoppingCriterion> & criterion_,
             const std::tr1::shared_ptr<dotk::DOTk_OptimizationDataMng> & opt_mng_);

    virtual void setMaxNumKrylovSolverItr(size_t itr_);
    virtual const std::tr1::shared_ptr<dotk::DOTk_KrylovSolverDataMng> & getDataMng() const;
    virtual const std::tr1::shared_ptr<dotk::DOTk_LinearOperator> & getLinearOperator() const;
    virtual const std::tr1::shared_ptr<dotk::vector<Real> > & getDescentDirection();
    virtual void solve(const std::tr1::shared_ptr<dotk::vector<Real> > & rhs_vec_,
                       const std::tr1::shared_ptr<dotk::DOTk_KrylovSolverStoppingCriterion> & criterion_,
                       const std::tr1::shared_ptr<dotk::DOTk_OptimizationDataMng> & opt_mng_);

private:
    std::tr1::shared_ptr<dotk::DOTk_KrylovSolverDataMng> m_DataMng;

    std::tr1::shared_ptr<dotk::vector<Real> > mConjugateDirection;
    std::tr1::shared_ptr<dotk::vector<Real> > mLinearOperatorTimesRes;

private:
    void initialize(const std::tr1::shared_ptr<dotk::vector<Real> > vec_);

private:
    DOTk_LeftPrecCR(const dotk::DOTk_LeftPrecCR &);
    dotk::DOTk_LeftPrecCR & operator=(const dotk::DOTk_LeftPrecCR &);
};

}

#endif /* DOTK_LEFTPRECCR_HPP_ */
