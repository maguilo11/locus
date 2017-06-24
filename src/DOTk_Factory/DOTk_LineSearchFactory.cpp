/*
 * DOTk_LineSearchFactory.cpp
 *
 *  Created on: Sep 15, 2014
 *      Author: Miguel A. Aguilo Valentin

 */

#include <iostream>

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

DOTk_LineSearchFactory::DOTk_LineSearchFactory(dotk::types::line_search_t aType) :
        m_FactoryType(aType)
{
}

DOTk_LineSearchFactory::~DOTk_LineSearchFactory()
{
}

void DOTk_LineSearchFactory::setFactoryType(dotk::types::line_search_t aType)
{
    m_FactoryType = aType;
}

dotk::types::line_search_t DOTk_LineSearchFactory::getFactoryType() const
{
    return (m_FactoryType);
}

void DOTk_LineSearchFactory::buildArmijoLineSearch(const std::shared_ptr<dotk::Vector<Real> > & aVector,
                                                   std::shared_ptr<dotk::DOTk_LineSearch> & aOutput)
{
    aOutput = std::make_shared<dotk::DOTk_ArmijoLineSearch>(aVector);
}

void DOTk_LineSearchFactory::buildGoldsteinLineSearch(const std::shared_ptr<dotk::Vector<Real> > & aVector,
                                                      std::shared_ptr<dotk::DOTk_LineSearch> & aOutput)
{
    aOutput = std::make_shared<dotk::DOTk_GoldsteinLineSearch>(aVector);
}

void DOTk_LineSearchFactory::buildCubicLineSearch(const std::shared_ptr<dotk::Vector<Real> > & aVector,
                                                  std::shared_ptr<dotk::DOTk_LineSearch> & aOutput)
{
    aOutput = std::make_shared<dotk::DOTk_BacktrackingCubicInterpolation>(aVector);
}

void DOTk_LineSearchFactory::buildGoldenSectionLineSearch(const std::shared_ptr<dotk::Vector<Real> > & aVector,
                                                          std::shared_ptr<dotk::DOTk_LineSearch> & aOutput)
{
    aOutput = std::make_shared<dotk::DOTk_GoldenSectionLineSearch>(aVector);
}

void DOTk_LineSearchFactory::build(const std::shared_ptr<dotk::Vector<Real> > & aVector,
                                   std::shared_ptr<dotk::DOTk_LineSearch> & aOutput) const
{
    switch(this->getFactoryType())
    {
        case dotk::types::LINE_SEARCH_DISABLED:
        {
            break;
        }
        case dotk::types::BACKTRACKING_ARMIJO:
        {
            aOutput = std::make_shared<dotk::DOTk_ArmijoLineSearch>(aVector);
            break;
        }
        case dotk::types::BACKTRACKING_GOLDSTEIN:
        {
            aOutput = std::make_shared<dotk::DOTk_GoldsteinLineSearch>(aVector);
            break;
        }
        case dotk::types::BACKTRACKING_CUBIC_INTRP:
        {
            aOutput = std::make_shared<dotk::DOTk_BacktrackingCubicInterpolation>(aVector);
            break;
        }
        case dotk::types::GOLDENSECTION:
        {
            aOutput = std::make_shared<dotk::DOTk_GoldenSectionLineSearch>(aVector);
            break;
        }
        default:
        {
            std::cout
                    << "\nDOTk WARNING: Invalid line search type, Default step set to Backtracking Cubic Interpolation.\n"
                    << std::flush;
            aOutput = std::make_shared<dotk::DOTk_BacktrackingCubicInterpolation>(aVector);
            break;
        }
    }
}

}
