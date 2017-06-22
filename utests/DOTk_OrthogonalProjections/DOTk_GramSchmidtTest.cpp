/*
 * DOTk_GramSchmidtTest.cpp
 *
 *  Created on: Feb 27, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#include "gtest/gtest.h"

#include "vector.hpp"
#include "DOTk_Primal.hpp"
#include "DOTk_MultiVector.hpp"
#include "DOTk_MultiVector.cpp"
#include "DOTk_GramSchmidt.hpp"
#include "DOTk_GtestDOTkVecTools.hpp"
#include "DOTk_TrustRegionAlgorithmsDataMng.hpp"

namespace DOTkGramSchmidtTest
{

TEST(DOTk_GramSchmidt, gramSchmidt)
{
    size_t nduals = 3;
    size_t ncontrols = 5;
    std::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialDualArray(nduals, 0);
    primal->allocateSerialControlArray(ncontrols, 0);

    dotk::DOTk_GramSchmidt projection(primal, nduals);

    std::shared_ptr<dotk::Vector<Real> >
        vector(new dotk::DOTk_MultiVector<Real>(*primal->control(), *primal->dual()));
    size_t index = 0;
    (*vector->control())[0] = -1;
    (*vector->control())[1] = -2;
    (*vector->control())[2] = -3;
    (*vector->control())[3] = -4;
    (*vector->control())[4] = -5;
    projection.setOrthogonalVector(index, vector);
    vector->control()->scale(2);
    projection.setLinearOperatorTimesOrthoVector(index, vector);

    (*vector->control())[0] = -6;
    (*vector->control())[1] = -7;
    (*vector->control())[2] = -8;
    (*vector->control())[3] = -9;
    (*vector->control())[4] = -10;
    index = 1;
    projection.setOrthogonalVector(1, vector);
    vector->control()->scale(2);
    projection.setLinearOperatorTimesOrthoVector(index, vector);

    (*vector->control())[0] = -11;
    (*vector->control())[1] = -12;
    (*vector->control())[2] = -13;
    (*vector->control())[3] = -14;
    (*vector->control())[4] = -15;
    index = 2;
    projection.setOrthogonalVector(2, vector);
    vector->control()->scale(2);
    projection.setLinearOperatorTimesOrthoVector(index, vector);

    index = 0;
    vector->fill(1);
    projection.gramSchmidt(index, vector);

    index = 1;
    projection.gramSchmidt(index, vector);

    index = 2;
    projection.gramSchmidt(index, vector);

    (*vector->dual())[0] = -1;
    (*vector->dual())[1] = -1;
    (*vector->dual())[2] = -1;
    (*vector->control())[0] = -2;
    (*vector->control())[1] = -3;
    (*vector->control())[2] = -4;
    (*vector->control())[3] = -5;
    (*vector->control())[4] = -6;
    dotk::gtest::checkResults(*projection.getOrthogonalVector(0), *vector);

    (*vector->dual())[0] = -1.21428571428;
    (*vector->dual())[1] = -1.21428571428;
    (*vector->dual())[2] = -1.21428571428;
    (*vector->control())[0] = -7.4285714285;
    (*vector->control())[1] = -8.6428571428;
    (*vector->control())[2] = -9.8571428571;
    (*vector->control())[3] = -11.071428571;
    (*vector->control())[4] = -12.285714285;
    dotk::gtest::checkResults(*projection.getOrthogonalVector(1), *vector);

    (*vector->dual())[0] = -1.33379362289;
    (*vector->dual())[1] = -1.33379362289;
    (*vector->dual())[2] = -1.33379362289;
    (*vector->control())[0] = -13.159678634;
    (*vector->control())[1] = -14.493472257;
    (*vector->control())[2] = -15.827265880;
    (*vector->control())[3] = -17.161059503;
    (*vector->control())[4] = -18.494853125;
    dotk::gtest::checkResults(*projection.getOrthogonalVector(2), *vector);
}

TEST(DOTk_GramSchmidt, clear)
{
    size_t ncontrols = 5;
    std::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 0);
    std::shared_ptr<dotk::DOTk_TrustRegionAlgorithmsDataMng> mng(new dotk::DOTk_TrustRegionAlgorithmsDataMng(primal));

    size_t num_ortho_vectors = 3;
    dotk::DOTk_GramSchmidt projection(primal, num_ortho_vectors);
    (*primal->control())[0] = -1;
    (*primal->control())[1] = -2;
    (*primal->control())[2] = -3;
    (*primal->control())[3] = -4;
    (*primal->control())[4] = -5;
    mng->getNewPrimal()->update(1., *primal->control(), 0.);
    projection.setOrthogonalVector(0, primal->control());
    projection.setLinearOperatorTimesOrthoVector(0, mng->getNewPrimal());
    dotk::gtest::checkResults(*mng->getNewPrimal(), *projection.getOrthogonalVector(0));
    dotk::gtest::checkResults(*mng->getNewPrimal(), *projection.getLinearOperatorTimesOrthoVector(0));

    (*primal->control())[0] = -6;
    (*primal->control())[1] = -7;
    (*primal->control())[2] = -8;
    (*primal->control())[3] = -9;
    (*primal->control())[4] = -10;
    mng->getNewPrimal()->update(1., *primal->control(), 0.);
    projection.setOrthogonalVector(1, primal->control());
    projection.setLinearOperatorTimesOrthoVector(1, mng->getNewPrimal());
    dotk::gtest::checkResults(*mng->getNewPrimal(), *projection.getOrthogonalVector(1));
    dotk::gtest::checkResults(*mng->getNewPrimal(), *projection.getLinearOperatorTimesOrthoVector(1));

    (*primal->control())[0] = -11;
    (*primal->control())[1] = -12;
    (*primal->control())[2] = -13;
    (*primal->control())[3] = -14;
    (*primal->control())[4] = -15;
    mng->getNewPrimal()->update(1., *primal->control(), 0.);
    projection.setOrthogonalVector(2, primal->control());
    projection.setLinearOperatorTimesOrthoVector(2, mng->getNewPrimal());
    dotk::gtest::checkResults(*mng->getNewPrimal(), *projection.getOrthogonalVector(2));
    dotk::gtest::checkResults(*mng->getNewPrimal(), *projection.getLinearOperatorTimesOrthoVector(2));

    projection.clear();

    std::shared_ptr<dotk::Vector<Real> > gold = primal->control()->clone();
    dotk::gtest::checkResults(*gold, *projection.getOrthogonalVector(0));
    dotk::gtest::checkResults(*gold, *projection.getLinearOperatorTimesOrthoVector(0));
    dotk::gtest::checkResults(*gold, *projection.getOrthogonalVector(1));
    dotk::gtest::checkResults(*gold, *projection.getLinearOperatorTimesOrthoVector(1));
    dotk::gtest::checkResults(*gold, *projection.getOrthogonalVector(2));
    dotk::gtest::checkResults(*gold, *projection.getLinearOperatorTimesOrthoVector(2));
}


}
