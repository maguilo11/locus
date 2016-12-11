/*
 * DOTk_UtilsCCSA.hpp
 *
 *  Created on: Dec 13, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_UTILSCCSA_HPP_
#define DOTK_UTILSCCSA_HPP_

#include <tr1/memory>

#include "DOTk_Types.hpp"

namespace dotk
{

class DOTk_DataMngCCSA;

template<typename ScalarType>
class Vector;

namespace ccsa
{

enum dual_solver_t
{
    NONLINEAR_CG = 1, QUASI_NEWTON = 2,
};

enum subproblem_t
{
    GCMMA = 1, MMA = 2,
};

enum stopping_criterion_t
{
    GRADIENT_TOLERANCE = 1,
    STEP_TOLERANCE = 2,
    OBJECTIVE_TOLERANCE = 3,
    RESIDUAL_TOLERANCE = 4,
    CONTROL_STAGNATION = 5,
    OBJECTIVE_STAGNATION = 6,
    MAX_NUMBER_ITERATIONS = 7,
    FEASIBILITY_MEASURE = 8,
    OPTIMALITY_AND_FEASIBILITY_MET = 9,
    NOT_CONVERGED = 10,
};

Real computeResidualNorm(const std::tr1::shared_ptr<dotk::Vector<Real> > & primal_,
                         const std::tr1::shared_ptr<dotk::Vector<Real> > & dual_,
                         const std::tr1::shared_ptr<dotk::DOTk_DataMngCCSA> & data_mng_);

}

}

#endif /* DOTK_UTILSCCSA_HPP_ */
