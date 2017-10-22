/*
 * Locus_Types.hpp
 *
 *  Created on: Oct 21, 2017
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef LOCUS_TYPES_HPP_
#define LOCUS_TYPES_HPP_

namespace locus
{

struct algorithm
{
    enum stop_t
    {
        NaN_NORM_TRIAL_STEP = 1,
        NaN_NORM_GRADIENT = 2,
        NORM_GRADIENT = 3,
        NORM_STEP = 4,
        OBJECTIVE_STAGNATION = 5,
        MAX_NUMBER_ITERATIONS = 6,
        OPTIMALITY_AND_FEASIBILITY = 7,
        ACTUAL_REDUCTION_TOLERANCE = 8,
        CONTROL_STAGNATION = 9,
        NaN_OBJECTIVE_GRADIENT = 10,
        NaN_FEASIBILITY_VALUE = 11,
        NOT_CONVERGED = 12
    };
};

} // namespace locus

#endif /* LOCUS_TYPES_HPP_ */
