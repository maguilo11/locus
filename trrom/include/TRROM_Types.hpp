/*
 * TRROM_Types.hpp
 *
 *  Created on: Aug 20, 2014
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef TRROM_TYPES_HPP_
#define TRROM_TYPES_HPP_

#include <map>
#include <cmath>
#include <math.h>
#include <cstdio>
#include <vector>
#include <limits>
#include <string>
#include <memory>
#include <cstdlib>
#include <utility>
#include <assert.h>
#include <iostream>

namespace trrom
{

namespace types
{

enum fidelity_t
{
    LOW_FIDELITY = 1, HIGH_FIDELITY = 2,
};

enum linear_operator_t
{
    REDUCED_HESSIAN = 1, SECANT_HESSIAN = 2, USER_DEFINED_MATRIX = 3
};

enum gradient_t
{
    USER_DEFINED_GRAD = 1,
};

enum stop_criterion_t
{
    NaN_TRIAL_STEP_NORM = 1,
    NaN_GRADIENT_NORM = 2,
    GRADIENT_TOL_SATISFIED = 3,
    TRIAL_STEP_TOL_SATISFIED = 4,
    OBJECTIVE_FUNC_TOL_SATISFIED = 5,
    MAX_NUM_ITR_REACHED = 6,
    OPTIMALITY_AND_FEASIBILITY_SATISFIED = 7,
    TRUST_REGION_RADIUS_SMALLER_THAN_TRIAL_STEP_NORM = 8,
    ACTUAL_REDUCTION_TOL_SATISFIED = 9,
    STAGNATION_MEASURE = 10,
    NaN_OPTIMALITY_NORM = 11,
    NaN_FEASIBILITY_NORM = 12,
    OPT_ALG_HAS_NOT_CONVERGED = 13
};

enum solver_stop_criterion_t
{
    NaN_CURVATURE_DETECTED = 1,
    ZERO_CURVATURE_DETECTED = 2,
    NEGATIVE_CURVATURE_DETECTED = 3,
    INF_CURVATURE_DETECTED = 4,
    SOLVER_TOLERANCE_SATISFIED = 5,
    TRUST_REGION_VIOLATED = 6,
    MAX_SOLVER_ITR_REACHED = 7,
    SOLVER_DID_NOT_CONVERGED = 8,
    NaN_RESIDUAL_NORM = 9,
    INF_RESIDUAL_NORM = 10,
    INVALID_INEXACTNESS_MEASURE = 11,
    INVALID_ORTHOGONALITY_MEASURE = 12,
};

enum display_t
{
    ITERATION = 1, FINAL = 2, OFF = 3, DETAILED = 4
};

enum variable_t
{
    STATE = 1, CONTROL = 2, DUAL = 3, PRIMAL = 4, SLACKS = 5, UNDEFINED_VARIABLE = 6
};

enum left_prec_t
{
    LEFT_PRECONDITIONER_DISABLED = 1,
};

}

}

typedef int Int;
typedef double Real;

#endif /* TRROM_TYPES_HPP_ */
