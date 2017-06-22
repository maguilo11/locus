/*
 * DOTk_ProjectedLineSearchStep.hpp
 *
 *  Created on: Sep 26, 2016
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_PROJECTEDLINESEARCHSTEP_HPP_
#define DOTK_PROJECTEDLINESEARCHSTEP_HPP_

#include "DOTk_LineSearchStepMng.hpp"

namespace dotk
{

class DOTk_Primal;
class DOTk_LineSearch;
class DOTk_BoundConstraint;
class DOTk_OptimizationDataMng;

template<typename ScalarType>
class Vector;

class DOTk_ProjectedLineSearchStep : public dotk::DOTk_LineSearchStepMng
{
public:
    explicit DOTk_ProjectedLineSearchStep(const std::shared_ptr<dotk::DOTk_Primal> & primal_);
    DOTk_ProjectedLineSearchStep(const std::shared_ptr<dotk::DOTk_Primal> & primal_,
                                 const std::shared_ptr<dotk::DOTk_LineSearch> & step_);
    virtual ~DOTk_ProjectedLineSearchStep();

    void setMaxNumFeasibleItr(size_t itr_);
    void setArmijoBoundConstraintMethodStep();
    void setBoundConstraintMethodContractionStep(Real input_);
    void setFeasibleDirectionConstraint(const std::shared_ptr<dotk::DOTk_Primal> & primal_);
    void setProjectionAlongFeasibleDirConstraint(const std::shared_ptr<dotk::DOTk_Primal> & primal_);

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
    std::shared_ptr<dotk::DOTk_BoundConstraint> m_BoundConstraint;

private:
    DOTk_ProjectedLineSearchStep(const dotk::DOTk_ProjectedLineSearchStep &);
    dotk::DOTk_ProjectedLineSearchStep & operator=(const dotk::DOTk_ProjectedLineSearchStep &);
};

}

#endif /* DOTK_PROJECTEDLINESEARCHSTEP_HPP_ */
