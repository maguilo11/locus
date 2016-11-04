/*
 * DOTk_MatrixTypes.hpp
 *
 *  Created on: May 26, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_MATRIXTYPES_HPP_
#define DOTK_MATRIXTYPES_HPP_

namespace dotk
{

namespace types
{

enum matrix_t
{
    SERIAL_DENSE_MATRIX = 1,
    OMP__DENSE_MATRIX = 2,
    MPI__DENSE_MATRIX = 3,
    MPI__DENSE_OMP_MATRIX = 4,
    SERIAL_ROW_MATRIX = 5,
    SERIAL_COLUMN_MATRIX = 6,
    SERIAL_UPPER_TRI_MATRIX = 7,
};

}

}

typedef double Real;

#endif /* DOTK_MATRIXTYPES_HPP_ */
