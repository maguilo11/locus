/*
 * DOTk_MultiVectorTest.cpp
 *
 *  Created on: Jul 4, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#include "gtest/gtest.h"

#include "DOTk_SerialArray.hpp"
#include "DOTk_MultiVector.hpp"
#include "DOTk_MultiVector.cpp"
#include "DOTk_GtestDOTkVecTools.hpp"

namespace DOTkMultiVectorTest
{

TEST(DOTkMultiVariableVector, scale)
{
    size_t num_duals = 10;
    size_t num_controls = 8;
    dotk::StdArray<Real> duals(num_duals, 1.);
    dotk::StdArray<Real> control(num_controls, 2.);
    dotk::DOTk_MultiVector<Real> multi_vector(control, duals);

    size_t dim = num_duals + num_controls;
    EXPECT_TRUE(multi_vector.size() == dim);
    EXPECT_TRUE(multi_vector.dual()->size() == num_duals);
    EXPECT_TRUE(multi_vector.state().use_count() == 0);
    EXPECT_TRUE(multi_vector.control()->size() == num_controls);

    multi_vector.scale(2);

    std::vector<Real> dual_gold(num_duals, 2.);
    std::vector<Real> control_gold(num_controls, 4.);

    dotk::gtest::checkResults(dual_gold.size(), dual_gold.data(), *multi_vector.dual());
    dotk::gtest::checkResults(control_gold.size(), control_gold.data(), *multi_vector.control());
}

TEST(DOTkMultiVariableVector, elementWiseMultiplication)
{
    size_t num_duals = 10;
    size_t num_controls = 8;
    dotk::StdArray<Real> dual(num_duals, 2.);
    dotk::StdArray<Real> control(num_controls, 3.);
    dotk::DOTk_MultiVector<Real> x(control, dual);
    dotk::DOTk_MultiVector<Real> y(control, dual);
    x.elementWiseMultiplication(y);

    std::vector<Real> dual_gold(num_duals, 4.);
    std::vector<Real> control_gold(num_controls, 9.);

    dotk::gtest::checkResults(dual_gold.size(), dual_gold.data(), *x.dual());
    dotk::gtest::checkResults(control_gold.size(), control_gold.data(), *x.control());
}

TEST(DOTkMultiVariableVector, axpy)
{
    size_t num_duals = 10;
    size_t num_controls = 8;
    dotk::StdArray<Real> dual(num_duals, 1.);
    dotk::StdArray<Real> control(num_controls, 2.);
    dotk::DOTk_MultiVector<Real> multi_vector(control, dual);

    dotk::DOTk_MultiVector<Real> input(control, dual);
    multi_vector.update(2, input, 1.);

    std::vector<Real> dual_gold(num_duals, 3.);
    std::vector<Real> control_gold(num_controls, 6.);

    dotk::gtest::checkResults(dual_gold.size(), dual_gold.data(), *multi_vector.dual());
    dotk::gtest::checkResults(control_gold.size(), control_gold.data(), *multi_vector.control());
}

TEST(DOTkMultiVariableVector, max)
{
    size_t num_duals = 10;
    size_t num_controls = 8;
    dotk::StdArray<Real> dual(num_duals, 1.);
    dotk::StdArray<Real> control(num_controls, 2.);
    dotk::DOTk_MultiVector<Real> multi_vector(control, dual);

    Real value = multi_vector.max();

    Real tolerance = 1e-8;
    EXPECT_NEAR(2, value, tolerance);
}

TEST(DOTkMultiVariableVector, min)
{
    size_t num_duals = 10;
    size_t num_controls = 8;
    dotk::StdArray<Real> dual(num_duals, 1.);
    dotk::StdArray<Real> control(num_controls, 2.);
    dotk::DOTk_MultiVector<Real> multi_vector(control, dual);

    Real value = multi_vector.min();

    Real tolerance = 1e-8;
    EXPECT_NEAR(1., value, tolerance);
}

TEST(DOTkMultiVariableVector, abs)
{
    size_t num_duals = 10;
    size_t num_controls = 8;
    dotk::StdArray<Real> dual(num_duals, -1.);
    dotk::StdArray<Real> control(num_controls, -2.);
    dotk::DOTk_MultiVector<Real> multi_vector(control, dual);

    multi_vector.abs();

    std::vector<Real> dual_gold(num_duals, 1.);
    std::vector<Real> control_gold(num_controls, 2.);

    dotk::gtest::checkResults(dual_gold.size(), dual_gold.data(), *multi_vector.dual());
    dotk::gtest::checkResults(control_gold.size(), control_gold.data(), *multi_vector.control());
}

TEST(DOTkMultiVariableVector, sum)
{
    size_t num_duals = 10;
    size_t num_controls = 8;
    dotk::StdArray<Real> dual(num_duals, 1.);
    dotk::StdArray<Real> control(num_controls, -2.);
    dotk::DOTk_MultiVector<Real> multi_vector(control, dual);

    Real value = multi_vector.sum();

    Real tolerance = 1e-8;
    EXPECT_NEAR(-6, value, tolerance);
}

TEST(DOTkMultiVariableVector, dot)
{
    size_t num_duals = 10;
    size_t num_controls = 8;
    dotk::StdArray<Real> dual(num_duals, 1.);
    dotk::StdArray<Real> control(num_controls, 2.);
    dotk::DOTk_MultiVector<Real> multi_vector(control, dual);

    dual.fill(1.);
    control.fill(1.);
    dotk::DOTk_MultiVector<Real> input(control, dual);

    Real value = multi_vector.dot(input);

    Real tolerance = 1e-8;
    EXPECT_NEAR(26, value, tolerance);
}

TEST(DOTkMultiVariableVector, norm)
{
    size_t num_duals = 10;
    size_t num_controls = 8;
    dotk::StdArray<Real> dual(num_duals, 1.);
    dotk::StdArray<Real> control(num_controls, 2.);
    dotk::DOTk_MultiVector<Real> multi_vector(control, dual);

    Real value = multi_vector.norm();

    Real tolerance = 1e-8;
    EXPECT_NEAR(6.48074069840, value, tolerance);
}

TEST(DOTkMultiVariableVector, fill)
{
    size_t num_duals = 10;
    size_t num_controls = 8;
    dotk::StdArray<Real> dual(num_duals, 1.);
    dotk::StdArray<Real> control(num_controls, 2.);
    dotk::DOTk_MultiVector<Real> multi_vector(control, dual);

    multi_vector.fill(4);

    std::vector<Real> dual_gold(num_duals, 4.);
    std::vector<Real> control_gold(num_controls, 4.);

    dotk::gtest::checkResults(dual_gold.size(), dual_gold.data(), *multi_vector.dual());
    dotk::gtest::checkResults(control_gold.size(), control_gold.data(), *multi_vector.control());
}

TEST(DOTkMultiVariableVector, copy)
{
    size_t num_duals = 10;
    size_t num_controls = 8;
    dotk::StdArray<Real> dual(num_duals, 1.);
    dotk::StdArray<Real> control(num_controls, 2.);
    dotk::DOTk_MultiVector<Real> multi_vector(control, dual);

    std::tr1::shared_ptr<dotk::Vector<Real> > input = multi_vector.clone();
    input->dual()->fill(-8);
    input->control()->fill(-16);

    multi_vector.update(1., *input, 0.);

    std::vector<Real> dual_gold(num_duals, -8.);
    std::vector<Real> control_gold(num_controls, -16.);

    dotk::gtest::checkResults(dual_gold.size(), dual_gold.data(), *multi_vector.dual());
    dotk::gtest::checkResults(control_gold.size(), control_gold.data(), *multi_vector.control());
}

TEST(DOTkMultiVariableVector, gather)
{
    size_t num_duals = 10;
    size_t num_controls = 8;
    dotk::StdArray<Real> dual(num_duals, 1.);
    dotk::StdArray<Real> control(num_controls, 2.);
    dotk::DOTk_MultiVector<Real> multi_vector(control, dual);
    EXPECT_TRUE(multi_vector.size() == 18);

    std::vector<Real> output(multi_vector.size());
    multi_vector.gather(output.data());

    dotk::gtest::checkResults(num_controls, output.data(), *multi_vector.control());
    dotk::gtest::checkResults(num_duals, output.data() + num_controls, *multi_vector.dual());
}

TEST(DOTkMultiVariableVector, operator_braket)
{
    size_t num_duals = 10;
    size_t num_controls = 8;
    dotk::StdArray<Real> dual(num_duals, 1.);
    dotk::StdArray<Real> control(num_controls, 2.);
    dotk::DOTk_MultiVector<Real> multi_vector(control, dual);

    for(size_t index = 0; index < multi_vector.size(); ++index)
    {
        multi_vector[index] = index;
    }

    Real control_gold[] = {0, 1, 2, 3, 4, 5, 6, 7};
    Real dual_gold[] = {8, 9, 10, 11, 12, 13, 14, 15, 16, 17};

    dotk::gtest::checkResults(num_controls, control_gold, *multi_vector.control());
    dotk::gtest::checkResults(num_duals, dual_gold, *multi_vector.dual());
}

TEST(DOTkMultiVariableVector, scale2)
{
    size_t num_duals = 10;
    size_t num_states = 6;
    size_t num_controls = 8;
    dotk::StdArray<Real> duals(num_duals, 1.);
    dotk::StdArray<Real> state(num_states, 3.);
    dotk::StdArray<Real> control(num_controls, 2.);
    dotk::DOTk_MultiVector<Real> multi_vector(control, state, duals);

    size_t dim = num_duals + num_controls + num_states;
    EXPECT_TRUE(multi_vector.size() == dim);
    EXPECT_TRUE(multi_vector.dual()->size() == num_duals);
    EXPECT_TRUE(multi_vector.state()->size() == num_states);
    EXPECT_TRUE(multi_vector.control()->size() == num_controls);

    multi_vector.scale(2);

    std::vector<Real> dual_gold(num_duals, 2.);
    std::vector<Real> state_gold(num_states, 6.);
    std::vector<Real> control_gold(num_controls, 4.);

    dotk::gtest::checkResults(dual_gold.size(), dual_gold.data(), *multi_vector.dual());
    dotk::gtest::checkResults(state_gold.size(), state_gold.data(), *multi_vector.state());
    dotk::gtest::checkResults(control_gold.size(), control_gold.data(), *multi_vector.control());
}

TEST(DOTkMultiVariableVector, cwiseProd2)
{
    size_t num_duals = 10;
    size_t num_states = 6;
    size_t num_controls = 8;
    dotk::StdArray<Real> dual(num_duals, 2.);
    dotk::StdArray<Real> state(num_states, 4.);
    dotk::StdArray<Real> control(num_controls, 3.);
    dotk::DOTk_MultiVector<Real> x(control, state, dual);
    dotk::DOTk_MultiVector<Real> y(control, state, dual);
    x.elementWiseMultiplication(y);

    std::vector<Real> dual_gold(num_duals, 4.);
    std::vector<Real> state_gold(num_states, 16);
    std::vector<Real> control_gold(num_controls, 9.);

    dotk::gtest::checkResults(dual_gold.size(), dual_gold.data(), *x.dual());
    dotk::gtest::checkResults(state_gold.size(), state_gold.data(), *x.state());
    dotk::gtest::checkResults(control_gold.size(), control_gold.data(), *x.control());
}

TEST(DOTkMultiVariableVector, axpy2)
{
    size_t num_duals = 10;
    size_t num_states = 6;
    size_t num_controls = 8;
    dotk::StdArray<Real> dual(num_duals, 1.);
    dotk::StdArray<Real> state(num_states, 3.);
    dotk::StdArray<Real> control(num_controls, 2.);
    dotk::DOTk_MultiVector<Real> multi_vector(control, state, dual);

    dotk::DOTk_MultiVector<Real> input(control, state, dual);
    multi_vector.update(2, input, 1.);

    std::vector<Real> dual_gold(num_duals, 3.);
    std::vector<Real> state_gold(num_states, 9);
    std::vector<Real> control_gold(num_controls, 6.);

    dotk::gtest::checkResults(dual_gold.size(), dual_gold.data(), *multi_vector.dual());
    dotk::gtest::checkResults(state_gold.size(), state_gold.data(), *multi_vector.state());
    dotk::gtest::checkResults(control_gold.size(), control_gold.data(), *multi_vector.control());
}

TEST(DOTkMultiVariableVector, max2)
{
    size_t num_duals = 10;
    size_t num_states = 6;
    size_t num_controls = 8;
    dotk::StdArray<Real> dual(num_duals, 1.);
    dotk::StdArray<Real> state(num_states, 3.);
    dotk::StdArray<Real> control(num_controls, 2.);
    dotk::DOTk_MultiVector<Real> multi_vector(control, state, dual);

    Real value = multi_vector.max();

    Real tolerance = 1e-8;
    EXPECT_NEAR(3, value, tolerance);
}

TEST(DOTkMultiVariableVector, min2)
{
    size_t num_duals = 10;
    size_t num_states = 6;
    size_t num_controls = 8;
    dotk::StdArray<Real> dual(num_duals, 1.);
    dotk::StdArray<Real> state(num_states, 3.);
    dotk::StdArray<Real> control(num_controls, 2.);
    dotk::DOTk_MultiVector<Real> multi_vector(control, state, dual);

    Real value = multi_vector.min();

    Real tolerance = 1e-8;
    EXPECT_NEAR(1., value, tolerance);
}

TEST(DOTkMultiVariableVector, abs2)
{
    size_t num_duals = 10;
    size_t num_states = 6;
    size_t num_controls = 8;
    dotk::StdArray<Real> dual(num_duals, -1.);
    dotk::StdArray<Real> state(num_states, -3.);
    dotk::StdArray<Real> control(num_controls, -2.);
    dotk::DOTk_MultiVector<Real> multi_vector(control, state, dual);

    multi_vector.abs();

    std::vector<Real> dual_gold(num_duals, 1.);
    std::vector<Real> state_gold(num_states, 3);
    std::vector<Real> control_gold(num_controls, 2.);

    dotk::gtest::checkResults(dual_gold.size(), dual_gold.data(), *multi_vector.dual());
    dotk::gtest::checkResults(state_gold.size(), state_gold.data(), *multi_vector.state());
    dotk::gtest::checkResults(control_gold.size(), control_gold.data(), *multi_vector.control());
}

TEST(DOTkMultiVariableVector, sum2)
{
    size_t num_duals = 10;
    size_t num_states = 6;
    size_t num_controls = 8;
    dotk::StdArray<Real> dual(num_duals, 1.);
    dotk::StdArray<Real> state(num_states, 3.);
    dotk::StdArray<Real> control(num_controls, -2.);
    dotk::DOTk_MultiVector<Real> multi_vector(control, state, dual);

    Real value = multi_vector.sum();

    Real tolerance = 1e-8;
    EXPECT_NEAR(12, value, tolerance);
}

TEST(DOTkMultiVariableVector, dot2)
{
    size_t num_duals = 10;
    size_t num_states = 6;
    size_t num_controls = 8;
    dotk::StdArray<Real> dual(num_duals, 1.);
    dotk::StdArray<Real> state(num_states, 3.);
    dotk::StdArray<Real> control(num_controls, 2.);
    dotk::DOTk_MultiVector<Real> multi_vector(control, state, dual);

    dual.fill(1.);
    state.fill(1.);
    control.fill(1.);
    dotk::DOTk_MultiVector<Real> input(control, state, dual);

    Real value = multi_vector.dot(input);

    Real tolerance = 1e-8;
    EXPECT_NEAR(44, value, tolerance);
}

TEST(DOTkMultiVariableVector, norm2)
{
    size_t num_duals = 10;
    size_t num_states = 6;
    size_t num_controls = 8;
    dotk::StdArray<Real> dual(num_duals, 1.);
    dotk::StdArray<Real> state(num_states, 3.);
    dotk::StdArray<Real> control(num_controls, 2.);
    dotk::DOTk_MultiVector<Real> multi_vector(control, state, dual);

    Real value = multi_vector.norm();

    Real tolerance = 1e-6;
    EXPECT_NEAR(9.7979589711, value, tolerance);
}

TEST(DOTkMultiVariableVector, fill2)
{
    size_t num_duals = 10;
    size_t num_states = 6;
    size_t num_controls = 8;
    dotk::StdArray<Real> dual(num_duals, 1.);
    dotk::StdArray<Real> state(num_states, 3.);
    dotk::StdArray<Real> control(num_controls, 2.);
    dotk::DOTk_MultiVector<Real> multi_vector(control, state, dual);

    multi_vector.fill(4);

    std::vector<Real> dual_gold(num_duals, 4);
    std::vector<Real> state_gold(num_states, 4);
    std::vector<Real> control_gold(num_controls, 4);

    dotk::gtest::checkResults(dual_gold.size(), dual_gold.data(), *multi_vector.dual());
    dotk::gtest::checkResults(state_gold.size(), state_gold.data(), *multi_vector.state());
    dotk::gtest::checkResults(control_gold.size(), control_gold.data(), *multi_vector.control());
}

TEST(DOTkMultiVariableVector, copy2)
{
    size_t num_duals = 10;
    size_t num_states = 6;
    size_t num_controls = 8;
    dotk::StdArray<Real> dual(num_duals, 1.);
    dotk::StdArray<Real> state(num_states, 3.);
    dotk::StdArray<Real> control(num_controls, 2.);
    dotk::DOTk_MultiVector<Real> multi_vector(control, state, dual);

    std::tr1::shared_ptr<dotk::Vector<Real> > input = multi_vector.clone();
    input->dual()->fill(-8);
    input->state()->fill(-2);
    input->control()->fill(-16);

    multi_vector.update(1., *input, 0.);

    std::vector<Real> dual_gold(num_duals, -8.);
    std::vector<Real> state_gold(num_states, -2.);
    std::vector<Real> control_gold(num_controls, -16.);

    dotk::gtest::checkResults(dual_gold.size(), dual_gold.data(), *multi_vector.dual());
    dotk::gtest::checkResults(state_gold.size(), state_gold.data(), *multi_vector.state());
    dotk::gtest::checkResults(control_gold.size(), control_gold.data(), *multi_vector.control());
}

TEST(DOTkMultiVariableVector, gather2)
{
    size_t num_duals = 10;
    size_t num_states = 6;
    size_t num_controls = 8;
    dotk::StdArray<Real> dual(num_duals, 1.);
    dotk::StdArray<Real> state(num_states, 3.);
    dotk::StdArray<Real> control(num_controls, 2.);
    dotk::DOTk_MultiVector<Real> multi_vector(control, state, dual);
    EXPECT_TRUE(multi_vector.size() == 24);

    std::vector<Real> output(multi_vector.size());
    multi_vector.gather(output.data());

    dotk::gtest::checkResults(num_controls, output.data(), *multi_vector.control());
    dotk::gtest::checkResults(num_states, output.data() + num_controls, *multi_vector.state());
    dotk::gtest::checkResults(num_duals, output.data() + num_controls + num_states, *multi_vector.dual());
}

TEST(DOTkMultiVariableVector, operator_braket2)
{
    size_t num_duals = 10;
    size_t num_states = 6;
    size_t num_controls = 8;
    dotk::StdArray<Real> dual(num_duals, 1.);
    dotk::StdArray<Real> state(num_states, 3.);
    dotk::StdArray<Real> control(num_controls, 2.);
    dotk::DOTk_MultiVector<Real> multi_vector(control, state, dual);

    for(size_t index = 0; index < multi_vector.size(); ++index)
    {
        multi_vector[index] = index;
    }

    Real control_gold[] = {0, 1, 2, 3, 4, 5, 6, 7};
    Real state_gold[] = {8, 9, 10, 11, 12, 13};
    Real dual_gold[] = {14, 15, 16, 17, 18, 19, 20, 21, 22, 23};

    dotk::gtest::checkResults(num_controls, control_gold, *multi_vector.control());
    dotk::gtest::checkResults(num_states, state_gold, *multi_vector.state());
    dotk::gtest::checkResults(num_duals, dual_gold, *multi_vector.dual());
}

}
