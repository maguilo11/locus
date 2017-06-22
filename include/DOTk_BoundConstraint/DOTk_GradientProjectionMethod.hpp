/*
 * DOTk_GradientProjectionMethod.hpp
 *
 *  Created on: Oct 16, 2016
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_GRADIENTPROJECTIONMETHOD_HPP_
#define DOTK_GRADIENTPROJECTIONMETHOD_HPP_

#include <fstream>
#include "DOTk_Types.hpp"

namespace dotk
{

class DOTk_Primal;
class DOTk_BoundConstraints;
class DOTk_LineSearchStepMng;
class DOTk_LineSearchAlgorithmsDataMng;

class GradientProjectionMethod
{
public:
    GradientProjectionMethod(const std::shared_ptr<dotk::DOTk_Primal> & primal_,
                             const std::shared_ptr<dotk::DOTk_LineSearchStepMng> & step_,
                             const std::shared_ptr<dotk::DOTk_LineSearchAlgorithmsDataMng> & mng_);
    ~GradientProjectionMethod();

    void getMin();

    void printDiagnostics();
    void setMaxNumIterations(size_t input_);
    void setObjectiveTolerance(double input_);
    void setProjectedGradientTolerance(double input_);

    size_t getIterationCount() const;
    dotk::types::stop_criterion_t getStoppingCriterion() const;

private:
    void reset();
    void checkBounds();
    void diagnostics();
    void openOutputFile(const char* const name_);
    void closeOutputFile();
    void setStoppingCriterion(dotk::types::stop_criterion_t input_);

    bool checkStoppinCritera();

private:
    bool m_PrintOutputFile;
    size_t m_IterationCount;
    size_t m_MaxNumIterations;
    double m_ObjectiveTolerance;
    double m_ProjectedGradientTolerance;
    dotk::types::stop_criterion_t m_StoppingCriterion;

    std::ofstream m_OutputFile;

    std::shared_ptr<dotk::DOTk_Primal> m_Primal;
    std::shared_ptr<dotk::DOTk_BoundConstraints> m_Bounds;
    std::shared_ptr<dotk::DOTk_LineSearchStepMng> m_LineSearchStep;
    std::shared_ptr<dotk::DOTk_LineSearchAlgorithmsDataMng> m_Data;

private:
    GradientProjectionMethod(const dotk::GradientProjectionMethod &);
    dotk::GradientProjectionMethod & operator=(const dotk::GradientProjectionMethod &);
};

}

#endif /* DOTK_GRADIENTPROJECTIONMETHOD_HPP_ */
