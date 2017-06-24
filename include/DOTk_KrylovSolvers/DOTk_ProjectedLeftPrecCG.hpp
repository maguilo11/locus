/*
 * DOTk_ProjectedLeftPrecCG.hpp
 *
 *  Created on: Feb 23, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_PROJECTEDLEFTPRECCG_HPP_
#define DOTK_PROJECTEDLEFTPRECCG_HPP_

#include "DOTk_KrylovSolver.hpp"

namespace dotk
{

class DOTk_Primal;
class DOTk_KrylovSolver;
class DOTk_LinearOperator;
class DOTk_KrylovSolverDataMng;
class DOTk_OptimizationDataMng;
class DOTk_KrylovSolverStoppingCriterion;

template<typename ScalarType>
class Vector;

class DOTk_ProjectedLeftPrecCG : public dotk::DOTk_KrylovSolver
{
public:
    explicit DOTk_ProjectedLeftPrecCG(const std::shared_ptr<dotk::DOTk_KrylovSolverDataMng> & aSolverDataMng);
    DOTk_ProjectedLeftPrecCG(const std::shared_ptr<dotk::DOTk_Primal> & primal_,
                             const std::shared_ptr<dotk::DOTk_LinearOperator> & aLinearOperator,
                             size_t aMaxNumIterations = 200);
    virtual ~DOTk_ProjectedLeftPrecCG();

    Real getOrthogonalityTolerance() const;
    void setOrthogonalityTolerance(Real aInput);
    Real getInexactnessTolerance() const;
    void setInexactnessTolerance(Real aInput);
    Real getOrthogonalityMeasure(size_t aRow, size_t aColumn) const;

    virtual void setMaxNumKrylovSolverItr(size_t aMaxNumIterations);
    virtual const std::shared_ptr<dotk::DOTk_KrylovSolverDataMng> & getDataMng() const;
    virtual const std::shared_ptr<dotk::DOTk_LinearOperator> & getLinearOperator() const;
    virtual const std::shared_ptr<dotk::Vector<Real> > &
    getDescentDirection();

    bool checkOrthogonalityMeasure();
    void initialize(const std::shared_ptr<dotk::Vector<Real> > & aRhsVector,
                    const std::shared_ptr<dotk::DOTk_OptimizationDataMng> & aMng);
    void ppcg(const std::shared_ptr<dotk::Vector<Real> > & aRhsVector,
              const std::shared_ptr<dotk::DOTk_KrylovSolverStoppingCriterion> & aCriterion,
              const std::shared_ptr<dotk::DOTk_OptimizationDataMng> & aMng);
    virtual void solve(const std::shared_ptr<dotk::Vector<Real> > & aRhsVector,
                       const std::shared_ptr<dotk::DOTk_KrylovSolverStoppingCriterion> & aCriterion,
                       const std::shared_ptr<dotk::DOTk_OptimizationDataMng> & aMng);

private:
    void setProjectedConjugateDirection();
    bool checkInexactnessMeasure(Real aCurvature, Real aNormOldResidual, Real aOldResidualDotProjectedPrecResidual);
    void initialize();
    void setFirstSolution();
    void clear();

private:
    Real m_InexactnessTolerance;
    Real m_OrthogonalityTolerance;

    std::vector<std::vector<Real> > m_OrthogonalityMeasure;
    std::vector<Real> m_OneOverNormPreconditionerTimesResidual;
    std::shared_ptr<dotk::DOTk_KrylovSolverDataMng> m_DataMng;
    std::shared_ptr<dotk::Vector<Real> > m_ProjectedConjugateDirection;

private:
    DOTk_ProjectedLeftPrecCG(const dotk::DOTk_ProjectedLeftPrecCG &);
    dotk::DOTk_ProjectedLeftPrecCG & operator=(const dotk::DOTk_ProjectedLeftPrecCG &);
};

}

#endif /* DOTK_PROJECTEDLEFTPRECCG_HPP_ */
