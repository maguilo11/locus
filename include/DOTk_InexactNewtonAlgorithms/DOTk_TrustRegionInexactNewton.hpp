/*
 * DOTk_TrustRegionInexactNewton.hpp
 *
 *  Created on: Nov 4, 2014
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_TRUSTREGIONINEXACTNEWTON_HPP_
#define DOTK_TRUSTREGIONINEXACTNEWTON_HPP_

#include "DOTk_InexactNewtonAlgorithms.hpp"

namespace dotk
{

class DOTk_Primal;
class DOTk_KrylovSolver;
class DOTk_LinearOperator;
class DOTk_KrylovSolverDataMng;
class DOTk_TrustRegionInexactNewtonIO;
class DOTk_TrustRegionAlgorithmsDataMng;

template<typename ScalarType>
class Vector;

class DOTk_TrustRegionInexactNewton : public dotk::DOTk_InexactNewtonAlgorithms
{
public:
    DOTk_TrustRegionInexactNewton(const std::shared_ptr<dotk::DOTk_LinearOperator> & aLinearOperator,
                                  const std::shared_ptr<dotk::DOTk_TrustRegionAlgorithmsDataMng> & aMng);
    DOTk_TrustRegionInexactNewton(const std::shared_ptr<dotk::DOTk_LinearOperator> & aLinearOperator,
                                  const std::shared_ptr<dotk::DOTk_TrustRegionAlgorithmsDataMng> & aMng,
                                  const std::shared_ptr<dotk::DOTk_KrylovSolverDataMng> & aSolverMng);
    virtual ~DOTk_TrustRegionInexactNewton();

    void setNewObjectiveFunctionValue(Real aInput);
    Real getNewObjectiveFunctionValue() const;

    virtual void setNumItrDone(size_t aInput);
    void setMaxNumKrylovSolverItr(size_t aInput);

    void setLeftPrecCgKrylovSolver(const std::shared_ptr<dotk::DOTk_Primal> & aPrimal,
                                   size_t aMaxNumIterations = 200);
    void setPrecGmresKrylovSolver(const std::shared_ptr<dotk::DOTk_Primal> & aPrimal,
                                  size_t aMaxNumIterations = 200);
    void setLeftPrecCgnrKrylovSolver(const std::shared_ptr<dotk::DOTk_Primal> & aPrimal,
                                     size_t aMaxNumIterations = 200);
    void setLeftPrecCgneKrylovSolver(const std::shared_ptr<dotk::DOTk_Primal> & aPrimal,
                                     size_t aMaxNumIterations = 200);
    void setLeftPrecCrKrylovSolver(const std::shared_ptr<dotk::DOTk_Primal> & aPrimal,
                                   size_t aMaxNumIterations = 200);
    void setLeftPrecGcrKrylovSolver(const std::shared_ptr<dotk::DOTk_Primal> & aPrimal,
                                    size_t aMaxNumIterations = 200);
    void setProjLeftPrecCgKrylovSolver(const std::shared_ptr<dotk::DOTk_Primal> & aPrimal,
                                       size_t aMaxNumIterations = 200);

    void printDiagnosticsAndSolutionEveryItr();
    void printDiagnosticsEveryItrAndSolutionAtTheEnd();

    void getMin();

private:
    void initialize();
    void checkAlgorithmInputs();
    void computeActualReduction();
    void solveTrustRegionSubProblem();
    void computeScaledInexactNewtonStep();
    bool checkTrustRegionSubProblemConvergence();

private:
    Real m_NewObjectiveFuncValue;

    std::shared_ptr<dotk::DOTk_KrylovSolver> m_KrylovSolver;
    std::shared_ptr<dotk::DOTk_TrustRegionInexactNewtonIO> m_IO;
    std::shared_ptr<dotk::DOTk_LinearOperator> m_LinearOperator;
    std::shared_ptr<dotk::DOTk_TrustRegionAlgorithmsDataMng> m_DataMng;

    std::shared_ptr<dotk::Vector<Real> > m_WorkVector;
    std::shared_ptr<dotk::Vector<Real> > m_HessTimesTrialStep;

private:
    DOTk_TrustRegionInexactNewton(const dotk::DOTk_TrustRegionInexactNewton&);
    dotk::DOTk_TrustRegionInexactNewton operator=(const dotk::DOTk_TrustRegionInexactNewton&);
};

}

#endif /* DOTK_TRUSTREGIONINEXACTNEWTON_HPP_ */
