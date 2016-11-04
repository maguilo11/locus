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

template<class Type>
class vector;

class DOTk_TrustRegionInexactNewton : public dotk::DOTk_InexactNewtonAlgorithms
{
public:
    DOTk_TrustRegionInexactNewton(const std::tr1::shared_ptr<dotk::DOTk_LinearOperator> & hessian_,
                                  const std::tr1::shared_ptr<dotk::DOTk_TrustRegionAlgorithmsDataMng> & mng_);
    DOTk_TrustRegionInexactNewton(const std::tr1::shared_ptr<dotk::DOTk_LinearOperator> & hessian_,
                                  const std::tr1::shared_ptr<dotk::DOTk_TrustRegionAlgorithmsDataMng> & mng_,
                                  const std::tr1::shared_ptr<dotk::DOTk_KrylovSolverDataMng> & solver_mng_);
    virtual ~DOTk_TrustRegionInexactNewton();

    void setNewObjectiveFunctionValue(Real value_);
    Real getNewObjectiveFunctionValue() const;

    virtual void setNumItrDone(size_t itr_);
    void setMaxNumKrylovSolverItr(size_t itr_);

    void setLeftPrecCgKrylovSolver(const std::tr1::shared_ptr<dotk::DOTk_Primal> & primal_,
                                   size_t max_num_itr_ = 200);
    void setPrecGmresKrylovSolver(const std::tr1::shared_ptr<dotk::DOTk_Primal> & primal_,
                                  size_t max_num_itr_ = 200);
    void setLeftPrecCgnrKrylovSolver(const std::tr1::shared_ptr<dotk::DOTk_Primal> & primal_,
                                     size_t max_num_itr_ = 200);
    void setLeftPrecCgneKrylovSolver(const std::tr1::shared_ptr<dotk::DOTk_Primal> & primal_,
                                     size_t max_num_itr_ = 200);
    void setLeftPrecCrKrylovSolver(const std::tr1::shared_ptr<dotk::DOTk_Primal> & primal_,
                                   size_t max_num_itr_ = 200);
    void setLeftPrecGcrKrylovSolver(const std::tr1::shared_ptr<dotk::DOTk_Primal> & primal_,
                                    size_t max_num_itr_ = 200);
    void setProjLeftPrecCgKrylovSolver(const std::tr1::shared_ptr<dotk::DOTk_Primal> & primal_,
                                       size_t max_num_itr_ = 200);

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

    std::tr1::shared_ptr<dotk::DOTk_KrylovSolver> m_KrylovSolver;
    std::tr1::shared_ptr<dotk::DOTk_TrustRegionInexactNewtonIO> m_IO;
    std::tr1::shared_ptr<dotk::DOTk_LinearOperator> m_LinearOperator;
    std::tr1::shared_ptr<dotk::DOTk_TrustRegionAlgorithmsDataMng> m_DataMng;

    std::tr1::shared_ptr<dotk::vector<Real> > m_WorkVector;
    std::tr1::shared_ptr<dotk::vector<Real> > m_HessTimesTrialStep;

private:
    DOTk_TrustRegionInexactNewton(const dotk::DOTk_TrustRegionInexactNewton&);
    dotk::DOTk_TrustRegionInexactNewton operator=(const dotk::DOTk_TrustRegionInexactNewton&);
};

}

#endif /* DOTK_TRUSTREGIONINEXACTNEWTON_HPP_ */
