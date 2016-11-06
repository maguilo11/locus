/*
 * DOTk_TangentialProblemCriterion.hpp
 *
 *  Created on: Dec 10, 2014
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_TANGENTIALPROBLEMCRITERION_HPP_
#define DOTK_TANGENTIALPROBLEMCRITERION_HPP_

#include "DOTk_KrylovSolverStoppingCriterion.hpp"

namespace dotk
{

class DOTk_KrylovSolver;

template<typename Type>
class vector;

class DOTk_TangentialProblemCriterion : public dotk::DOTk_KrylovSolverStoppingCriterion
{
public:
    explicit DOTk_TangentialProblemCriterion(const std::tr1::shared_ptr<dotk::vector<Real> > & map_);
    virtual ~DOTk_TangentialProblemCriterion();

    void setTangentialTolerance(Real tolerance_);
    Real getTangentialTolerance() const;
    void setTangentialToleranceContractionFactor(Real factor_);
    Real getTangentialToleranceContractionFactor() const;
    void setCurrentTrialStep(const std::tr1::shared_ptr<dotk::vector<Real> > & trial_step_);

    virtual Real evaluate(const dotk::DOTk_KrylovSolver * const solver_,
                          const std::tr1::shared_ptr<dotk::vector<Real> > & kernel_vector_);

private:
    void initialize();
    Real computeStoppingTolerance(const std::tr1::shared_ptr<dotk::vector<Real> > & kernel_vector_);

private:
    std::vector<Real> m_StoppingCriterionOptions;
    std::tr1::shared_ptr<dotk::vector<Real> > m_WorkVector;

private:
    DOTk_TangentialProblemCriterion(const dotk::DOTk_TangentialProblemCriterion &);
    dotk::DOTk_TangentialProblemCriterion & operator=(const dotk::DOTk_TangentialProblemCriterion &);
};

}

#endif /* DOTK_TANGENTIALPROBLEMCRITERION_HPP_ */
