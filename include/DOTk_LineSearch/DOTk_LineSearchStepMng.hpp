/*
 * DOTk_LineSearchStepMng.hpp
 *
 *  Created on: Sep 26, 2016
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_LINESEARCHSTEPMNG_HPP_
#define DOTK_LINESEARCHSTEPMNG_HPP_

#include "DOTk_Types.hpp"

namespace dotk
{

class DOTk_Primal;
class DOTk_OptimizationDataMng;

class DOTk_LineSearchStepMng
{
public:
    virtual ~DOTk_LineSearchStepMng()
    {
    }

    virtual void setContractionFactor(Real input_) = 0;
    virtual void setMaxNumIterations(size_t input_) = 0;
    virtual void setStagnationTolerance(Real input_) = 0;

    virtual Real step() const = 0;
    virtual size_t iterations() const = 0;
    virtual void build(const std::tr1::shared_ptr<dotk::DOTk_Primal> & primal_, dotk::types::line_search_t type_) = 0;
    virtual void solveSubProblem(const std::tr1::shared_ptr<dotk::DOTk_OptimizationDataMng> & mng_) = 0;
};

}

#endif /* DOTK_LINESEARCHSTEPMNG_HPP_ */
