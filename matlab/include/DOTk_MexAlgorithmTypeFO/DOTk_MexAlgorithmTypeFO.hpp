/*
 * DOTk_MexAlgorithmTypeFO.hpp
 *
 *  Created on: Apr 12, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_MEXALGORITHMTYPEFO_HPP_
#define DOTK_MEXALGORITHMTYPEFO_HPP_

#include <mex.h>
#include <tr1/memory>

#include "DOTk_Types.hpp"

namespace dotk
{

class DOTk_Primal;
class DOTk_LineSearchStepMng;
class DOTk_FirstOrderAlgorithm;
class DOTk_ProjectedLineSearchStep;
class DOTk_LineSearchAlgorithmsDataMng;

class DOTk_MexAlgorithmTypeFO
{
public:
    explicit DOTk_MexAlgorithmTypeFO(const mxArray* options_);
    virtual ~DOTk_MexAlgorithmTypeFO();

    void setNumDuals(const mxArray* options_);
    size_t getNumDuals() const;
    void setNumControls(const mxArray* options_);

    size_t getNumControls() const;
    size_t getMaxNumAlgorithmItr() const;
    double getGradientTolerance() const;
    double getTrialStepTolerance() const;
    double getObjectiveTolerance() const;

    size_t getMaxNumLineSearchItr() const;
    double getLineSearchContractionFactor() const;
    double getLineSearchStagnationTolerance() const;

    dotk::types::problem_t getProblemType() const;
    dotk::types::line_search_t getLineSearchMethod() const;

    void setLineSearchStepMng(dotk::DOTk_LineSearchStepMng & mng_);
    void gatherOutputData(const dotk::DOTk_FirstOrderAlgorithm & algorithm_,
                          const dotk::DOTk_LineSearchAlgorithmsDataMng & mng_,
                          mxArray* output_[]);
    void setBoundConstraintMethod(const mxArray* options_,
                                  const std::tr1::shared_ptr<dotk::DOTk_Primal> & primal_,
                                  const std::tr1::shared_ptr<dotk::DOTk_ProjectedLineSearchStep> & step_);

    virtual void solve(const mxArray* input_[], mxArray* output_[]) = 0;

private:
    void initialize(const mxArray* options_);

private:
    size_t m_NumDuals;
    size_t m_NumControls;
    size_t m_MaxNumAlgorithmItr;
    size_t m_MaxNumLineSearchItr;

    double m_GradientTolerance;
    double m_TrialStepTolerance;
    double m_ObjectiveTolerance;
    double m_LineSearchContractionFactor;
    double m_LineSearchStagnationTolerance;

    dotk::types::problem_t m_ProblemType;
    dotk::types::line_search_t m_LineSearchMethod;

private:
    DOTk_MexAlgorithmTypeFO(const dotk::DOTk_MexAlgorithmTypeFO&);
    dotk::DOTk_MexAlgorithmTypeFO& operator=(const dotk::DOTk_MexAlgorithmTypeFO&);
};

}

#endif /* DOTK_MEXALGORITHMTYPEFO_HPP_ */
