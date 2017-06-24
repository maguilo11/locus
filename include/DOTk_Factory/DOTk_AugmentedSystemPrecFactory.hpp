/*
 * DOTk_AugmentedSystemPrecFactory.hpp
 *
 *  Created on: Feb 23, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_AUGMENTEDSYSTEMPRECFACTORY_HPP_
#define DOTK_AUGMENTEDSYSTEMPRECFACTORY_HPP_

#include <memory>

namespace dotk
{

class DOTk_Primal;
class DOTk_LeftPreconditioner;

class DOTk_AugmentedSystemPrecFactory
{
public:
    explicit DOTk_AugmentedSystemPrecFactory(size_t aMaxNumSolverIterations);
    ~DOTk_AugmentedSystemPrecFactory();

    void buildAugmentedSystemPrecWithPcgSolver
    (const std::shared_ptr<dotk::DOTk_Primal> & aPrimal,
     std::shared_ptr<dotk::DOTk_LeftPreconditioner> & aPreconditioner);
    void buildAugmentedSystemPrecWithGcrSolver
    (const std::shared_ptr<dotk::DOTk_Primal> & aPrimal,
     std::shared_ptr<dotk::DOTk_LeftPreconditioner> & aPreconditioner);
    void buildAugmentedSystemPrecWithCrSolver
    (const std::shared_ptr<dotk::DOTk_Primal> & aPrimal,
     std::shared_ptr<dotk::DOTk_LeftPreconditioner> & aPreconditioner);
    void buildAugmentedSystemPrecWithCgneSolver
    (const std::shared_ptr<dotk::DOTk_Primal> & aPrimal,
     std::shared_ptr<dotk::DOTk_LeftPreconditioner> & aPreconditioner);
    void buildAugmentedSystemPrecWithCgnrSolver
    (const std::shared_ptr<dotk::DOTk_Primal> & aPrimal,
     std::shared_ptr<dotk::DOTk_LeftPreconditioner> & aPreconditioner);
    void buildAugmentedSystemPrecWithGmresSolver
    (const std::shared_ptr<dotk::DOTk_Primal> & aPrimal,
     std::shared_ptr<dotk::DOTk_LeftPreconditioner> & aPreconditioner);

private:
    size_t m_MaxNumSolverItr;

private:
    DOTk_AugmentedSystemPrecFactory(const dotk::DOTk_AugmentedSystemPrecFactory &);
    dotk::DOTk_AugmentedSystemPrecFactory & operator=(const dotk::DOTk_AugmentedSystemPrecFactory &);
};

}

#endif /* DOTK_AUGMENTEDSYSTEMPRECFACTORY_HPP_ */
