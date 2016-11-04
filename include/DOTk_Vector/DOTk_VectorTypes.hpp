/*
 * DOTk_VectorTypes.hpp
 *
 *  Created on: Apr 3, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_VECTORTYPES_HPP_
#define DOTK_VECTORTYPES_HPP_

namespace dotk
{

namespace types
{

enum container_t
{
    SERIAL_VECTOR = 1,
    SERIAL_ARRAY = 2,
    OMP_VECTOR = 3,
    OMP_ARRAY = 4,
    MPI_VECTOR = 5,
    MPI_ARRAY = 6,
    MPIx_VECTOR = 7,
    MPIx_ARRAY = 8,
    PRIMAL_VECTOR = 9,
    MULTI_VECTOR = 10,
    USER_DEFINED_CONTAINER = 11,
    UNDEFINED_DOTK_CONTAINER = 12,
};

}

}

#endif /* DOTK_VECTORTYPES_HPP_ */
