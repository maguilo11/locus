/*
 * DOTk_LineSearchStep.hpp
 *
 *  Created on: Sep 26, 2016
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_LINESEARCHSTEP_HPP_
#define DOTK_LINESEARCHSTEP_HPP_

#include "DOTk_LineSearchStepMng.hpp"

namespace dotk
{

class DOTk_Primal;
class DOTk_LineSearch;
class DOTk_OptimizationDataMng;

class DOTk_LineSearchStep : public dotk::DOTk_LineSearchStepMng
{
public:
    explicit DOTk_LineSearchStep(const std::shared_ptr<dotk::DOTk_Primal> & primal_);
    virtual ~DOTk_LineSearchStep();


    void setArmijoLineSearch(const std::shared_ptr<dotk::DOTk_Primal> & primal_, Real contraction_factor_ = 0.5);
    void setGoldsteinLineSearch(const std::shared_ptr<dotk::DOTk_Primal> & primal_,
                                Real constant_ = 0.9,
                                Real contraction_factor_ = 0.5);
    void setCubicLineSearch(const std::shared_ptr<dotk::DOTk_Primal> & primal_, Real contraction_factor_ = 0.5);
    void setGoldenSectionLineSearch(const std::shared_ptr<dotk::DOTk_Primal> & primal_,
                                    Real contraction_factor_ = 0.5);

    void setContractionFactor(Real input_);
    void setMaxNumIterations(size_t input_);
    void setStagnationTolerance(Real input_);

    Real step() const;
    size_t iterations() const;
    void build(const std::shared_ptr<dotk::DOTk_Primal> & primal_, dotk::types::line_search_t type_);
    void solveSubProblem(const std::shared_ptr<dotk::DOTk_OptimizationDataMng> & mng_);

private:
    std::shared_ptr<dotk::DOTk_LineSearch> m_LineSearch;

private:
    DOTk_LineSearchStep(const dotk::DOTk_LineSearchStep &);
    dotk::DOTk_LineSearchStep & operator=(const dotk::DOTk_LineSearchStep &);
};

}

#endif /* DOTK_LINESEARCHSTEP_HPP_ */
