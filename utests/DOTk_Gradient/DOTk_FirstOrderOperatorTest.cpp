/*
 * DOTk_FirstOrderOperatorTest.cpp
 *
 *  Created on: Sep 9, 2014
 *      Author: Miguel A. Aguilo Valentin
 */

#include "gtest/gtest.h"

#include "DOTk_AssemblyManager.hpp"
#include "DOTk_Rosenbrock.hpp"
#include "vector.hpp"
#include "DOTk_Primal.hpp"
#include "DOTk_UserDefinedGrad.hpp"
#include "DOTk_GtestDOTkVecTools.hpp"
#include "DOTk_LineSearchMngTypeULP.hpp"

namespace DOTkFirstOrderOperatorTest
{

TEST(DOTk_FirstOrderOperator, checkGrad)
{
    std::tr1::shared_ptr<dotk::vector<Real> > new_grad = dotk::gtest::allocateControl();
    new_grad->fill(std::numeric_limits<Real>::quiet_NaN());

    std::tr1::shared_ptr<dotk::vector<Real> > old_grad = new_grad->clone();
    old_grad->fill(2);

    dotk::DOTk_FirstOrderOperator grad;
    grad.checkGrad(old_grad, new_grad);

    dotk::gtest::checkResults(*new_grad, *old_grad);
}

TEST(DOTk_UserDefinedGrad, gradient)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);
    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    dotk::DOTk_LineSearchMngTypeULP mng(primal, objective);

    dotk::DOTk_UserDefinedGrad grad;
    EXPECT_EQ(dotk::types::USER_DEFINED_GRAD, grad.type());
    grad.gradient(&mng);

    std::tr1::shared_ptr<dotk::vector<Real> > gold = dotk::gtest::allocateControl();
    (*gold)[0] = 1602.;
    (*gold)[1] = -400.;
    dotk::gtest::checkResults(*mng.getNewGradient(), *gold);
}

}
