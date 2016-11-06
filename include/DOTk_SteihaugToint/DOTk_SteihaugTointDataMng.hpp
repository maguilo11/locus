/*
 * DOTk_SteihaugTointDataMng.hpp
 *
 *  Created on: Sep 5, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_STEIHAUGTOINTDATAMNG_HPP_
#define DOTK_STEIHAUGTOINTDATAMNG_HPP_

#include "DOTk_OptimizationDataMng.hpp"

namespace dotk
{

class DOTk_Primal;
class DOTk_FirstOrderOperator;
class DOTk_AssemblyManager;

template<typename Type>
class vector;
template<typename Type>
class DOTk_ObjectiveFunction;
template<typename Type>
class DOTk_EqualityConstraint;

class DOTk_SteihaugTointDataMng : public dotk::DOTk_OptimizationDataMng
{
public:
    DOTk_SteihaugTointDataMng(const std::tr1::shared_ptr<dotk::DOTk_Primal> & primal_,
                              const std::tr1::shared_ptr<dotk::DOTk_ObjectiveFunction<Real> > & objective_);
    DOTk_SteihaugTointDataMng(const std::tr1::shared_ptr<dotk::DOTk_Primal> & primal_,
                              const std::tr1::shared_ptr<dotk::DOTk_ObjectiveFunction<Real> > & objective_,
                              const std::tr1::shared_ptr<dotk::DOTk_EqualityConstraint<Real> > & equality_);
    virtual ~DOTk_SteihaugTointDataMng();

    void setUserDefinedGradient();
    void setForwardFiniteDiffGradient(const std::tr1::shared_ptr<dotk::DOTk_Primal> & epsilon_);
    void setCentralFiniteDiffGradient(const std::tr1::shared_ptr<dotk::DOTk_Primal> & epsilon_);
    void setBackwardFiniteDiffGradient(const std::tr1::shared_ptr<dotk::DOTk_Primal> & epsilon_);
    void setParallelForwardFiniteDiffGradient(const std::tr1::shared_ptr<dotk::DOTk_Primal> & epsilon_);
    void setParallelCentralFiniteDiffGradient(const std::tr1::shared_ptr<dotk::DOTk_Primal> & epsilon_);
    void setParallelBackwardFiniteDiffGradient(const std::tr1::shared_ptr<dotk::DOTk_Primal> & epsilon_);

    virtual void computeGradient();
    virtual Real evaluateObjective();
    virtual Real evaluateObjective(const std::tr1::shared_ptr<dotk::vector<Real> > & input_);
    virtual size_t getObjectiveFunctionEvaluationCounter() const;
    virtual const std::tr1::shared_ptr<dotk::DOTk_Primal> & getPrimalStruc() const;
    virtual const std::tr1::shared_ptr<dotk::DOTk_AssemblyManager> & getRoutinesMng() const;

private:
    void initialize();
    void setFiniteDiffPerturbationVector(const std::tr1::shared_ptr<dotk::DOTk_Primal> & epsilon_);

private:
    std::tr1::shared_ptr<dotk::DOTk_Primal> m_PrimalStruc;
    std::tr1::shared_ptr<dotk::DOTk_FirstOrderOperator> m_Gradient;
    std::tr1::shared_ptr<dotk::DOTk_AssemblyManager> m_AssemblyMng;

private:
    DOTk_SteihaugTointDataMng(const dotk::DOTk_SteihaugTointDataMng &);
    dotk::DOTk_SteihaugTointDataMng & operator=(const dotk::DOTk_SteihaugTointDataMng & rhs_);
};

}

#endif /* DOTK_STEIHAUGTOINTDATAMNG_HPP_ */
