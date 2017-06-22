/*
 * DOTk_ProjectedStep.hpp
 *
 *  Created on: Oct 23, 2016
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_PROJECTEDSTEP_HPP_
#define DOTK_PROJECTEDSTEP_HPP_

#include "DOTk_LineSearchStepMng.hpp"

namespace dotk
{

class DOTk_Primal;
class DOTk_LineSearch;
class DOTk_BoundConstraints;
class DOTk_OptimizationDataMng;

template<typename ScalarType>
class Vector;

class DOTk_ProjectedStep : public dotk::DOTk_LineSearchStepMng
{
public:
    explicit DOTk_ProjectedStep(const std::shared_ptr<dotk::DOTk_Primal> & primal_);
    virtual ~DOTk_ProjectedStep();


    void setArmijoLineSearch(const std::shared_ptr<dotk::DOTk_Primal> & primal_, Real contraction_factor_ = 0.5);
    void setGoldsteinLineSearch(const std::shared_ptr<dotk::DOTk_Primal> & primal_,
                                Real constant_ = 0.9,
                                Real contraction_factor_ = 0.5);
    void setCubicLineSearch(const std::shared_ptr<dotk::DOTk_Primal> & primal_, Real contraction_factor_ = 0.5);
    void setGoldenSectionLineSearch(const std::shared_ptr<dotk::DOTk_Primal> & primal_, Real contraction_factor_ = 0.5);

    void setContractionFactor(Real input_);
    void setMaxNumIterations(size_t input_);
    void setStagnationTolerance(Real input_);

    Real step() const;
    size_t iterations() const;
    void build(const std::shared_ptr<dotk::DOTk_Primal> & primal_, dotk::types::line_search_t type_);
    void solveSubProblem(const std::shared_ptr<dotk::DOTk_OptimizationDataMng> & mng_);

private:
    std::shared_ptr<dotk::Vector<Real> > m_WorkVector;
    std::shared_ptr<dotk::Vector<Real> > m_LowerBound;
    std::shared_ptr<dotk::Vector<Real> > m_UpperBound;
    std::shared_ptr<dotk::DOTk_LineSearch> m_LineSearch;
    std::shared_ptr<dotk::DOTk_BoundConstraints> m_BoundConstraint;

private:
    DOTk_ProjectedStep(const dotk::DOTk_ProjectedStep &);
    dotk::DOTk_ProjectedStep & operator=(const dotk::DOTk_ProjectedStep &);
};

}

#endif /* DOTK_PROJECTEDSTEP_HPP_ */
