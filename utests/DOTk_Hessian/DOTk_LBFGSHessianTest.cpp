/*
 * DOTk_LBFGSHessianTest.cpp
 *
 *  Created on: Sep 6, 2014
 *      Author: Miguel A. Aguilo Valentin
 */

#include "gtest/gtest.h"

#include "vector.hpp"
#include "DOTk_Primal.hpp"
#include "DOTk_LBFGSHessian.hpp"

#include "DOTk_AssemblyManager.hpp"
#include "DOTk_Rosenbrock.hpp"
#include "DOTk_LineSearchMngTypeULP.hpp"

#include "DOTk_GtestDOTkVecTools.hpp"

namespace DOTkLBFGSHessianTest
{

TEST(DOTk_LBFGSHessian, getDeltaPrimalStorage)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);
    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    dotk::DOTk_LineSearchMngTypeULP mng(primal, objective);

    size_t secant_storage = 3;
    dotk::DOTk_LBFGSHessian hess(*mng.getMatrixTimesVector(), secant_storage);
    (*hess.getDeltaPrimalStorage(0))[0] = 1.164953510500657;
    (*hess.getDeltaPrimalStorage(0))[1] = 0.626839082632431;

    std::tr1::shared_ptr<dotk::Vector<Real> > gold = primal->control()->clone();
    (*gold)[0] = 1.164953510500657;
    (*gold)[1] = 0.626839082632431;
    dotk::gtest::checkResults(*hess.getDeltaPrimalStorage(0), *gold);
}

TEST(DOTk_LBFGSHessian, getDeltaGradStorage)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);
    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    dotk::DOTk_LineSearchMngTypeULP mng(primal, objective);

    size_t secant_storage = 3;
    dotk::DOTk_LBFGSHessian hess(*mng.getMatrixTimesVector(), secant_storage);
    (*hess.getDeltaGradStorage(0))[0] = 0.059059777981351;
    (*hess.getDeltaGradStorage(0))[1] = 1.79707178369482;

    std::tr1::shared_ptr<dotk::Vector<Real> > gold = primal->control()->clone();
    (*gold)[0] = 0.059059777981351;
    (*gold)[1] = 1.79707178369482;
    dotk::gtest::checkResults(*hess.getDeltaGradStorage(0), *gold);
}

TEST(DOTk_LBFGSHessian, getDeltaGradPrimalInnerProductStorage)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);
    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    dotk::DOTk_LineSearchMngTypeULP mng(primal, objective);

    size_t secant_storage = 3;
    dotk::DOTk_LBFGSHessian hess(*mng.getMatrixTimesVector(), secant_storage);
    EXPECT_EQ(secant_storage, (*hess.getDeltaGradPrimalInnerProductStorage()).size());
    (*hess.getDeltaGradPrimalInnerProductStorage())[0] = 0.059059777981351;
    (*hess.getDeltaGradPrimalInnerProductStorage())[1] = 1.79707178369482;
    (*hess.getDeltaGradPrimalInnerProductStorage())[2] = 0.264068528817227;

    std::vector<Real> gold(secant_storage, 0.);
    gold[0] = 0.059059777981351;
    gold[1] = 1.79707178369482;
    gold[2] = 0.264068528817227;
    dotk::gtest::checkResults((*hess.getDeltaGradPrimalInnerProductStorage()), gold);
}

TEST(DOTk_LBFGSHessian, getHessian)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);
    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    dotk::DOTk_LineSearchMngTypeULP mng(primal, objective);

    size_t secant_storage = 3;
    dotk::DOTk_LBFGSHessian hess(*mng.getMatrixTimesVector(), secant_storage);
    EXPECT_EQ(dotk::types::LBFGS_HESS, hess.getHessianType());

    hess.setNumUpdatesStored(3);
    EXPECT_EQ(3, hess.getNumUpdatesStored());
    // Set Delta Primal Storage
    (*hess.getDeltaPrimalStorage(0))[0] = 1.164953510500657;
    (*hess.getDeltaPrimalStorage(0))[1] = 0.626839082632431;
    (*hess.getDeltaPrimalStorage(1))[0] = 0.075080154677683;
    (*hess.getDeltaPrimalStorage(1))[1] = 0.351606902768522;
    (*hess.getDeltaPrimalStorage(2))[0] = 0.696512535163682;
    (*hess.getDeltaPrimalStorage(2))[1] = -1.696142480747077;
    // Set Delta Gradient Storage
    (*hess.getDeltaGradStorage(0))[0] = 0.059059777981351;
    (*hess.getDeltaGradStorage(0))[1] = 1.79707178369482;
    (*hess.getDeltaGradStorage(1))[0] = 0.264068528817227;
    (*hess.getDeltaGradStorage(1))[1] = 0.871673288690637;
    (*hess.getDeltaGradStorage(2))[0] = -1.446171539339335;
    (*hess.getDeltaGradStorage(2))[1] = -0.701165345682908;
    for(size_t j = 0; j < hess.getMaxNumSecantStorage(); ++j)
    {
        (*hess.getDeltaGradPrimalInnerProductStorage())[j] =
                hess.getDeltaGradStorage(j)->dot(*hess.getDeltaPrimalStorage(j));
    }
    std::tr1::shared_ptr<dotk::Vector<Real> > trial_step = primal->control()->clone();
    trial_step->fill(1.);
    std::tr1::shared_ptr<dotk::Vector<Real> > Hess_times_vec = primal->control()->clone();
    hess.getHessian(trial_step, Hess_times_vec);

    std::tr1::shared_ptr<dotk::Vector<Real> > gold = primal->control()->clone();
    (*gold)[0] = 18.562869970651185;
    (*gold)[1] = 8.888762989742279;
    dotk::gtest::checkResults(*Hess_times_vec, *gold);
}

}
