/*
 * DOTk_MexFactoriesAlgorithmTypeGB.hpp
 *
 *  Created on: Apr 23, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_MEXFACTORIESALGORITHMTYPEGB_HPP_
#define DOTK_MEXFACTORIESALGORITHMTYPEGB_HPP_

#include <mex.h>
#include <tr1/memory>

namespace dotk
{

class DOTk_Primal;
class DOTk_OptimizationDataMng;

namespace mex
{

template<typename Algorithm>
void buildKrylovSolver(const mxArray* options_,
                       const std::tr1::shared_ptr<dotk::DOTk_Primal> & primal_,
                       Algorithm & algorithm_);

template<typename Manager>
void buildTrustRegionMethod(const mxArray* options_, Manager & mng_);

template<typename Manager>
void buildGradient(const mxArray* options_, const std::tr1::shared_ptr<dotk::DOTk_Primal> & epsilon_, Manager & mng_);

}

}

#endif /* DOTK_MEXFACTORIESALGORITHMTYPEGB_HPP_ */
