/*
 * DOTk_QuasiNormalProbCriterion.hpp
 *
 *  Created on: Feb 23, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_QUASINORMALPROBCRITERION_HPP_
#define DOTK_QUASINORMALPROBCRITERION_HPP_

#include "DOTk_KrylovSolverStoppingCriterion.hpp"

namespace dotk
{

class DOTk_KrylovSolver;

template<typename ScalarType>
class Vector;

class DOTk_QuasiNormalProbCriterion: public dotk::DOTk_KrylovSolverStoppingCriterion
{
public:
    DOTk_QuasiNormalProbCriterion();
    virtual ~DOTk_QuasiNormalProbCriterion();

    Real getStoppingTolerance() const;
    void setStoppingTolerance(Real tolerance_);
    Real getRelativeTolerance() const;
    void setRelativeTolerance(Real tolerance_);
    Real getTrustRegionRadiusPenaltyParameter() const;
    void setTrustRegionRadiusPenaltyParameter(Real penalty_);

    virtual Real evaluate(const dotk::DOTk_KrylovSolver* const solver_,
                          const std::shared_ptr<dotk::Vector<Real> > & kernel_vector_);

private:
    void initialize();

private:
    DOTk_QuasiNormalProbCriterion(const dotk::DOTk_QuasiNormalProbCriterion &);
    dotk::DOTk_QuasiNormalProbCriterion & operator=(const dotk::DOTk_QuasiNormalProbCriterion &);
};

}

#endif /* DOTK_QUASINORMALPROBCRITERION_HPP_ */
