/*
 * DOTk_MexContainerFactory.hpp
 *
 *  Created on: Apr 23, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_MEXCONTAINERFACTORY_HPP_
#define DOTK_MEXCONTAINERFACTORY_HPP_

#include <mex.h>

namespace dotk
{

class DOTk_Dual;
class DOTk_State;
class DOTk_Primal;
class DOTk_Control;

namespace mex
{

void buildDualContainer(const mxArray* options_, dotk::DOTk_Dual & dual_);
void buildStateContainer(const mxArray* options_, dotk::DOTk_State & state_);
void buildPrimalContainer(const mxArray* options_, dotk::DOTk_Primal & primal_);
void buildControlContainer(const mxArray* options_, dotk::DOTk_Control & control_);

void buildDualContainer(const mxArray* options_, dotk::DOTk_Primal & primal_);
void buildStateContainer(const mxArray* options_, dotk::DOTk_Primal & primal_);
void buildControlContainer(const mxArray* options_, dotk::DOTk_Primal & primal_);

}

}

#endif /* DOTK_MEXCONTAINERFACTORY_HPP_ */
