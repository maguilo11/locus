/*
 * DOTk_ArnoldiProjectionTest.cpp
 *
 *  Created on: Nov 4, 2014
 *      Author: Miguel A. Aguilo Valentin
 */

#include "gtest/gtest.h"

#include "vector.hpp"
#include "DOTk_Primal.hpp"
#include "DOTk_MultiVector.hpp"
#include "DOTk_MultiVector.cpp"
#include "DOTk_ArnoldiProjection.hpp"
#include "DOTk_GtestDOTkVecTools.hpp"

namespace DOTkArnoldiProjectionTest
{


TEST(DOTk_ArnoldiProjection, setAndGetKrylovSubspaceDim)
{
    size_t ncontrols = 2;
    size_t krylov_subspace_dim = 200;
    std::shared_ptr<dotk::DOTk_Primal> primal = std::make_shared<dotk::DOTk_Primal>();
    primal->allocateSerialControlArray(ncontrols);
    dotk::DOTk_ArnoldiProjection projection(primal, krylov_subspace_dim);

    EXPECT_EQ(200, projection.getKrylovSubspaceDim());
    projection.setKrylovSubspaceDim(3);
    EXPECT_EQ(3, projection.getKrylovSubspaceDim());
}

TEST(DOTk_ArnoldiProjection, setAndGetProjectionType)
{
    size_t ncontrols = 2;
    size_t krylov_subspace_dim = 200;
    std::shared_ptr<dotk::DOTk_Primal> primal = std::make_shared<dotk::DOTk_Primal>();
    primal->allocateSerialControlArray(ncontrols);
    dotk::DOTk_ArnoldiProjection projection(primal, krylov_subspace_dim);

    EXPECT_EQ(dotk::types::ARNOLDI, projection.getProjectionType());
    projection.setProjectionType(dotk::types::PROJECTION_DISABLED);
    EXPECT_EQ(dotk::types::PROJECTION_DISABLED, projection.getProjectionType());
}

TEST(DOTk_ArnoldiProjection, setAndGetOrthogonalVector)
{
    size_t ncontrols = 2;
    size_t krylov_subspace_dim = 2;
    std::shared_ptr<dotk::DOTk_Primal> primal = std::make_shared<dotk::DOTk_Primal>();
    primal->allocateSerialControlArray(ncontrols);
    dotk::DOTk_ArnoldiProjection projection(primal, krylov_subspace_dim);

    std::shared_ptr<dotk::Vector<Real> > orthogonal_vector = primal->control()->clone();
    projection.setOrthogonalVector(0, orthogonal_vector);
    dotk::gtest::checkResults(*orthogonal_vector, *projection.getOrthogonalVector(0));
    projection.setOrthogonalVector(1, orthogonal_vector);
    dotk::gtest::checkResults(*orthogonal_vector, *projection.getOrthogonalVector(1));
}

TEST(DOTk_ArnoldiProjection, arnoldi)
{
    size_t nduals = 2;
    size_t ncontrols = 2;
    size_t krylov_subspace_dim = 2;
    std::shared_ptr<dotk::DOTk_Primal> primal = std::make_shared<dotk::DOTk_Primal>();
    primal->allocateSerialDualArray(nduals, 0);
    primal->allocateSerialControlArray(ncontrols, 0);
    dotk::DOTk_ArnoldiProjection projection(primal, krylov_subspace_dim);

    std::shared_ptr<dotk::Vector<Real> > ortho_vector =
            std::make_shared<dotk::DOTk_MultiVector<Real>>(*primal->control(), *primal->dual());
    (*ortho_vector->dual())[0] = 0.443285802300365;
    (*ortho_vector->dual())[1] = 0.380162139476383;
    (*ortho_vector->control())[0] = 0.600561554982781;
    (*ortho_vector->control())[1] = 0.546168713736264;

    size_t orthogonal_vector_index = 0;
    projection.setOrthogonalVector(orthogonal_vector_index, ortho_vector);
    ortho_vector->fill(1);
    Real norm_residual = std::sqrt(ortho_vector->dot(*ortho_vector));
    projection.setInitialResidual(norm_residual);

    projection.arnoldi(orthogonal_vector_index, ortho_vector);

    // RESULTS
    std::shared_ptr<dotk::Vector<Real> > gold = ortho_vector->clone();
    (*gold->dual())[0] = 0.43667601435716247;
    (*gold->dual())[1] = 0.37449358182591613;
    (*gold->control())[0] = 0.59160664484427306;
    (*gold->control())[1] = 0.53802484952885077;
    dotk::gtest::checkResults(*gold, *ortho_vector);
}

}
