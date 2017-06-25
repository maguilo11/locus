/*
 * TRROM_MOCK_SVD.cpp
 *
 *  Created on: Aug 18, 2016
 */

#include <cassert>

#include "TRROM_Vector.hpp"
#include "TRROM_Matrix.hpp"
#include "TRROM_MOCK_SVD.hpp"

namespace trrom
{

namespace mock
{

SVD::SVD() :
        m_Vector()
{
}

SVD::~SVD()
{
}

void SVD::solve(const std::shared_ptr<trrom::Matrix<double> > & data_,
                std::shared_ptr<trrom::Vector<double> > & singular_values_,
                std::shared_ptr<trrom::Matrix<double> > & left_singular_vectors_,
                std::shared_ptr<trrom::Matrix<double> > & right_singular_vectors_)
{
    assert(data_->getNumRows() == 4);
    assert(data_->getNumCols() == 4);

    m_Vector = singular_values_->create(4);
    left_singular_vectors_ = data_->create();
    right_singular_vectors_ = data_->create();

    // Set singular values
    (*m_Vector)[0] = 13.337774133491964;
    (*m_Vector)[1] = 10.205479188810592;
    (*m_Vector)[2] = 8.616542289422034;
    (*m_Vector)[3] = 7.259970706918076;
    singular_values_ = m_Vector->create();
    singular_values_->update(1., *m_Vector, 0.);

    left_singular_vectors_ = data_->create(4, 4);
    // Column 1
    (*left_singular_vectors_).replaceGlobalValue(0, 0, 0.874014425350118);
    (*left_singular_vectors_).replaceGlobalValue(1, 0, 0.147480414171853);
    (*left_singular_vectors_).replaceGlobalValue(2, 0, 0.429742945713252);
    (*left_singular_vectors_).replaceGlobalValue(3, 0, 0.172247822410906);
    // Column 2
    (*left_singular_vectors_).replaceGlobalValue(0, 1, 0.280715377808905);
    (*left_singular_vectors_).replaceGlobalValue(1, 1, -0.872818994255086);
    (*left_singular_vectors_).replaceGlobalValue(2, 1, -0.349052263051439);
    (*left_singular_vectors_).replaceGlobalValue(3, 1, 0.193774089051695);
    // Column 3
    (*left_singular_vectors_).replaceGlobalValue(0, 2, 0.395444832978763);
    (*left_singular_vectors_).replaceGlobalValue(1, 2, 0.264242088660374);
    (*left_singular_vectors_).replaceGlobalValue(2, 2, -0.663406691099328);
    (*left_singular_vectors_).replaceGlobalValue(3, 2, -0.577659990699930);
    // Column 4
    (*left_singular_vectors_).replaceGlobalValue(0, 3, 0.030348723398292);
    (*left_singular_vectors_).replaceGlobalValue(1, 3, 0.382900312462190);
    (*left_singular_vectors_).replaceGlobalValue(2, 3, -0.503363765554302);
    (*left_singular_vectors_).replaceGlobalValue(3, 3, 0.774009835358340);

    right_singular_vectors_ = data_->create(4, 4);
    // Column 1
    (*right_singular_vectors_).replaceGlobalValue(0, 0, 0.773647003360372);
    (*right_singular_vectors_).replaceGlobalValue(1, 0, 0.104674763508203);
    (*right_singular_vectors_).replaceGlobalValue(2, 0, 0.556759977375391);
    (*right_singular_vectors_).replaceGlobalValue(3, 0, -0.283781316631262);
    // Column 2
    (*right_singular_vectors_).replaceGlobalValue(0, 1, 0.324743456263993);
    (*right_singular_vectors_).replaceGlobalValue(1, 1, -0.809621039894205);
    (*right_singular_vectors_).replaceGlobalValue(2, 1, -0.423539924337457);
    (*right_singular_vectors_).replaceGlobalValue(3, 1, -0.244273191051664);
    // Column 3
    (*right_singular_vectors_).replaceGlobalValue(0, 2, 0.541826805904227);
    (*right_singular_vectors_).replaceGlobalValue(1, 2, 0.290308644585233);
    (*right_singular_vectors_).replaceGlobalValue(2, 2, -0.494099141178771);
    (*right_singular_vectors_).replaceGlobalValue(3, 2, 0.614825700478688);
    // Column 4
    (*right_singular_vectors_).replaceGlobalValue(0, 3, 0.049352955421706);
    (*right_singular_vectors_).replaceGlobalValue(1, 3, 0.499277334278467);
    (*right_singular_vectors_).replaceGlobalValue(2, 3, -0.516234732240650);
    (*right_singular_vectors_).replaceGlobalValue(3, 3, -0.694109595449707);
}

}

}
