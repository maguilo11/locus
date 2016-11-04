/*
 * DOTk_LineSearchFactory.cpp
 *
 *  Created on: Sep 15, 2014
 *      Author: Miguel A. Aguilo Valentin

 */

#include "vector.hpp"
#include "DOTk_FirstOrderOperator.hpp"
#include "DOTk_OptimizationDataMng.hpp"
#include "DOTk_ArmijoLineSearch.hpp"
#include "DOTk_GoldsteinLineSearch.hpp"
#include "DOTk_BacktrackingCubicInterpolation.hpp"
#include "DOTk_GoldenSectionLineSearch.hpp"
#include "DOTk_LineSearchFactory.hpp"

namespace dotk
{

DOTk_LineSearchFactory::DOTk_LineSearchFactory() :
        m_FactoryType(dotk::types::LINE_SEARCH_DISABLED)
{
}

DOTk_LineSearchFactory::DOTk_LineSearchFactory(dotk::types::line_search_t type_) :
        m_FactoryType(type_)
{
}

DOTk_LineSearchFactory::~DOTk_LineSearchFactory()
{
}

void DOTk_LineSearchFactory::setFactoryType(dotk::types::line_search_t type_)
{
    m_FactoryType = type_;
}

dotk::types::line_search_t DOTk_LineSearchFactory::getFactoryType() const
{
    return (m_FactoryType);
}

void DOTk_LineSearchFactory::buildArmijoLineSearch(const std::tr1::shared_ptr<dotk::vector<Real> > & vector_,
                                                   std::tr1::shared_ptr<dotk::DOTk_LineSearch> & line_search_)
{
    line_search_.reset(new dotk::DOTk_ArmijoLineSearch(vector_));
}

void DOTk_LineSearchFactory::buildGoldsteinLineSearch(const std::tr1::shared_ptr<dotk::vector<Real> > & vector_,
                                                      std::tr1::shared_ptr<dotk::DOTk_LineSearch> & line_search_)
{
    line_search_.reset(new dotk::DOTk_GoldsteinLineSearch(vector_));
}

void DOTk_LineSearchFactory::buildCubicLineSearch(const std::tr1::shared_ptr<dotk::vector<Real> > & vector_,
                                                  std::tr1::shared_ptr<dotk::DOTk_LineSearch> & line_search_)
{
    line_search_.reset(new dotk::DOTk_BacktrackingCubicInterpolation(vector_));
}

void DOTk_LineSearchFactory::buildGoldenSectionLineSearch(const std::tr1::shared_ptr<dotk::vector<Real> > & vector_,
                                                          std::tr1::shared_ptr<dotk::DOTk_LineSearch> & line_search_)
{
    line_search_.reset(new dotk::DOTk_GoldenSectionLineSearch(vector_));
}

void DOTk_LineSearchFactory::build(const std::tr1::shared_ptr<dotk::vector<Real> > & vector_,
                                   std::tr1::shared_ptr<dotk::DOTk_LineSearch> & line_search_) const
{
    switch(this->getFactoryType())
    {
        case dotk::types::LINE_SEARCH_DISABLED:
        {
            break;
        }
        case dotk::types::BACKTRACKING_ARMIJO:
        {
            line_search_.reset(new dotk::DOTk_ArmijoLineSearch(vector_));
            break;
        }
        case dotk::types::BACKTRACKING_GOLDSTEIN:
        {
            line_search_.reset(new dotk::DOTk_GoldsteinLineSearch(vector_));
            break;
        }
        case dotk::types::BACKTRACKING_CUBIC_INTRP:
        {
            line_search_.reset(new dotk::DOTk_BacktrackingCubicInterpolation(vector_));
            break;
        }
        case dotk::types::GOLDENSECTION:
        {
            line_search_.reset(new dotk::DOTk_GoldenSectionLineSearch(vector_));
            break;
        }
        default:
        {
            std::cout
                    << "\nDOTk WARNING: Invalid line search type, Default step set to Backtracking Cubic Interpolation.\n"
                    << std::flush;
            line_search_.reset(new dotk::DOTk_BacktrackingCubicInterpolation(vector_));
            break;
        }
    }
}

}
