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

template<typename ScalarType>
class Vector;

class DOTk_LeftPrecCR : public dotk::DOTk_KrylovSolver
{
public:
    explicit DOTk_LeftPrecCR(const std::shared_ptr<dotk::DOTk_KrylovSolverDataMng> & aSolverDataMng);
    DOTk_LeftPrecCR(const std::shared_ptr<dotk::DOTk_Primal> & aPrimal,
                    const std::shared_ptr<dotk::DOTk_LinearOperator> & aLinearOperator);
    virtual ~DOTk_LeftPrecCR();

    void initialize(const std::shared_ptr<dotk::Vector<Real> > & aRhsVector,
                    const std::shared_ptr<dotk::DOTk_KrylovSolverStoppingCriterion> & aCriterion,
                    const std::shared_ptr<dotk::DOTk_OptimizationDataMng> & aMng);
    void pcr(const std::shared_ptr<dotk::Vector<Real> > & aRhsVector,
             const std::shared_ptr<dotk::DOTk_KrylovSolverStoppingCriterion> & aCriterion,
             const std::shared_ptr<dotk::DOTk_OptimizationDataMng> & aMng);

    virtual void setMaxNumKrylovSolverItr(size_t itr_);
    virtual const std::shared_ptr<dotk::DOTk_KrylovSolverDataMng> & getDataMng() const;
    virtual const std::shared_ptr<dotk::DOTk_LinearOperator> & getLinearOperator() const;
    virtual const std::shared_ptr<dotk::Vector<Real> > & getDescentDirection();
    virtual void solve(const std::shared_ptr<dotk::Vector<Real> > & aRhsVector,
                       const std::shared_ptr<dotk::DOTk_KrylovSolverStoppingCriterion> & aCriterion,
                       const std::shared_ptr<dotk::DOTk_OptimizationDataMng> & aMng);

private:
    std::shared_ptr<dotk::DOTk_KrylovSolverDataMng> m_DataMng;

    std::shared_ptr<dotk::Vector<Real> > mConjugateDirection;
    std::shared_ptr<dotk::Vector<Real> > mLinearOperatorTimesRes;

private:
    void initialize(const std::shared_ptr<dotk::Vector<Real> > vec_);

private:
    DOTk_LeftPrecCR(const dotk::DOTk_LeftPrecCR &);
    dotk::DOTk_LeftPrecCR & operator=(const dotk::DOTk_LeftPrecCR &);
};

}

#endif /* DOTK_LEFTPRECCR_HPP_ */
