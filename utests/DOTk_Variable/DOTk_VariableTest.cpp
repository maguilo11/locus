/*
 * DOTk_VariableTest.cpp
 *
 *  Created on: Feb 27, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#include "gtest/gtest.h"

#include "DOTk_Dual.hpp"
#include "DOTk_State.hpp"
#include "DOTk_Control.hpp"
#include "DOTk_GtestDOTkVecTools.hpp"

namespace DOTkVariable
{

TEST(DOTk_Variable, control)
{
    std::tr1::shared_ptr<dotk::vector<Real> > data = dotk::gtest::allocateData(5, 1.);
    dotk::DOTk_Control control(*data);

    EXPECT_EQ(dotk::types::CONTROL, control.type());
    EXPECT_EQ(5, control.size());
    dotk::gtest::checkResults(*control.data(), *data);

    std::tr1::shared_ptr<dotk::vector<Real> > lower = dotk::gtest::allocateData(5, -1.);
    std::tr1::shared_ptr<dotk::vector<Real> > upper = dotk::gtest::allocateData(5, 5.);
    dotk::DOTk_Control control_bounds_active(*data, *lower, *upper);

    EXPECT_EQ(dotk::types::CONTROL, control_bounds_active.type());
    EXPECT_EQ(5, control_bounds_active.size());
    dotk::gtest::checkResults(*control_bounds_active.data(), *data);
    dotk::gtest::checkResults(*control_bounds_active.lowerBound(), *lower);
    dotk::gtest::checkResults(*control_bounds_active.upperBound(), *upper);
}

TEST(DOTk_Variable, state)
{
    std::tr1::shared_ptr<dotk::vector<Real> > data = dotk::gtest::allocateData(6, 1.);
    dotk::DOTk_State state(*data);

    EXPECT_EQ(dotk::types::STATE, state.type());
    EXPECT_EQ(6, state.size());
    dotk::gtest::checkResults(*state.data(), *data);

    std::tr1::shared_ptr<dotk::vector<Real> > lower = dotk::gtest::allocateData(6, -1.);
    std::tr1::shared_ptr<dotk::vector<Real> > upper = dotk::gtest::allocateData(6, 5.);
    dotk::DOTk_State state_bounds_active(*data, *lower, *upper);

    EXPECT_EQ(dotk::types::STATE, state_bounds_active.type());
    EXPECT_EQ(6, state_bounds_active.size());
    dotk::gtest::checkResults(*state_bounds_active.data(), *data);
    dotk::gtest::checkResults(*state_bounds_active.lowerBound(), *lower);
    dotk::gtest::checkResults(*state_bounds_active.upperBound(), *upper);
}

TEST(DOTk_Variable, dual)
{
    std::tr1::shared_ptr<dotk::vector<Real> > data = dotk::gtest::allocateData(4, 1.);
    dotk::DOTk_Dual dual(*data);

    EXPECT_EQ(dotk::types::DUAL, dual.type());
    EXPECT_EQ(4, dual.size());
    dotk::gtest::checkResults(*dual.data(), *data);

    std::tr1::shared_ptr<dotk::vector<Real> > lower = dotk::gtest::allocateData(4, -1.);
    std::tr1::shared_ptr<dotk::vector<Real> > upper = dotk::gtest::allocateData(4, 5.);
    dotk::DOTk_Dual dual_bounds_active(*data, *lower, *upper);

    EXPECT_EQ(dotk::types::DUAL, dual_bounds_active.type());
    EXPECT_EQ(4, dual_bounds_active.size());
    dotk::gtest::checkResults(*dual_bounds_active.data(), *data);
    dotk::gtest::checkResults(*dual_bounds_active.lowerBound(), *lower);
    dotk::gtest::checkResults(*dual_bounds_active.upperBound(), *upper);
}

}
