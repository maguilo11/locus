/*
 * DOTk_ParallelUtils.hpp
 *
 *  Created on: Apr 3, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_PARALLELUTILS_HPP_
#define DOTK_PARALLELUTILS_HPP_

namespace dotk
{

namespace parallel
{

MPI_Datatype mpiDataType(const std::type_info & type_id_);
void loadBalance(const int & global_dim_, const int & comm_size_, int* input_);
void loadBalance(const int & nrows_, const int & ncols_, const int & comm_size_, int* row_counts_, int* data_counts_);

}

}

#endif /* DOTK_PARALLELUTILS_HPP_ */
