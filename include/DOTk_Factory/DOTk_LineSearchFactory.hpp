/*
 * DOTk_LineSearchFactory.hpp
 *
 *  Created on: Sep 10, 2014
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_LINESEARCHFACTORY_HPP_
#define DOTK_LINESEARCHFACTORY_HPP_

#include <memory>
#include "DOTk_Types.hpp"

namespace dotk
{

class DOTk_LineSearch;

template<typename Type>
class Vector;

class DOTk_LineSearchFactory
{
public:
    DOTk_LineSearchFactory();
    explicit DOTk_LineSearchFactory(dotk::types::line_search_t aType);
    ~DOTk_LineSearchFactory();

    void setFactoryType(dotk::types::line_search_t aType);
    dotk::types::line_search_t getFactoryType() const;

    void buildArmijoLineSearch(const std::shared_ptr<dotk::Vector<Real> > & aVector,
                               std::shared_ptr<dotk::DOTk_LineSearch> & aOutput);
    void buildGoldsteinLineSearch(const std::shared_ptr<dotk::Vector<Real> > & aVector,
                                  std::shared_ptr<dotk::DOTk_LineSearch> & aOutput);
    void buildCubicLineSearch(const std::shared_ptr<dotk::Vector<Real> > & aVector,
                              std::shared_ptr<dotk::DOTk_LineSearch> & aOutput);
    void buildGoldenSectionLineSearch(const std::shared_ptr<dotk::Vector<Real> > & aVector,
                                      std::shared_ptr<dotk::DOTk_LineSearch> & aOutput);

    void build(const std::shared_ptr<dotk::Vector<Real> > & aVector,
               std::shared_ptr<dotk::DOTk_LineSearch> & aOutput) const;

private:
    dotk::types::line_search_t m_FactoryType;

private:
    DOTk_LineSearchFactory(const dotk::DOTk_LineSearchFactory &);
    DOTk_LineSearchFactory operator=(const dotk::DOTk_LineSearchFactory &);
};

}

#endif /* DOTK_LINESEARCHFACTORY_HPP_ */
