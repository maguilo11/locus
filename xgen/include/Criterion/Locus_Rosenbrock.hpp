/*
 * Locus_Rosenbrock.hpp
 *
 *  Created on: Oct 21, 2017
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef LOCUS_ROSENBROCK_HPP_
#define LOCUS_ROSENBROCK_HPP_

#include <cmath>
#include <memory>
#include <cassert>

#include "Locus_Vector.hpp"
#include "Locus_Criterion.hpp"
#include "Locus_MultiVector.hpp"

namespace locus
{

template<typename ScalarType, typename OrdinalType = size_t>
class Rosenbrock : public locus::Criterion<ScalarType, OrdinalType>
{
public:
    Rosenbrock()
    {
    }
    virtual ~Rosenbrock()
    {
    }

    /*!
     * Evaluate Rosenbrock function:
     *      f(\mathbf{x}) = 100 * \left(x_2 - x_1^2\right)^2 + \left(1 - x_1\right)^2
     * */
    ScalarType value(const locus::MultiVector<ScalarType, OrdinalType> & aState,
                     const locus::MultiVector<ScalarType, OrdinalType> & aControl)
    {
        assert(aControl.getNumVectors() == static_cast<OrdinalType>(1));

        const OrdinalType tVectorIndex = 0;
        const locus::Vector<ScalarType, OrdinalType> & tMyControl = aControl[tVectorIndex];

        ScalarType tOutput = static_cast<ScalarType>(100.)
                * std::pow((tMyControl[1] - tMyControl[0] * tMyControl[0]), static_cast<ScalarType>(2))
                + std::pow(static_cast<ScalarType>(1) - tMyControl[0], static_cast<ScalarType>(2));

        return (tOutput);
    }
    /*!
     * Compute Rosenbrock gradient:
     *      \frac{\partial{f}}{\partial x_1} = -400 * \left(x_2 - x_1^2\right) * x_1 +
     *                                          \left(2 * \left(1 - x_1\right) \right)
     *      \frac{\partial{f}}{\partial x_2} = 200 * \left(x_2 - x_1^2\right)
     * */
    void gradient(const locus::MultiVector<ScalarType, OrdinalType> & aState,
                  const locus::MultiVector<ScalarType, OrdinalType> & aControl,
                  locus::MultiVector<ScalarType, OrdinalType> & aOutput)
    {
        assert(aOutput.getNumVectors() == static_cast<OrdinalType>(1));
        assert(aControl.getNumVectors() == static_cast<OrdinalType>(1));

        const OrdinalType tVectorIndex = 0;
        locus::Vector<ScalarType, OrdinalType> & tMyOutput = aOutput[tVectorIndex];
        const locus::Vector<ScalarType, OrdinalType> & tMyControl = aControl[tVectorIndex];

        tMyOutput[0] = static_cast<ScalarType>(-400) * (tMyControl[1] - (tMyControl[0] * tMyControl[0])) * tMyControl[0]
                + static_cast<ScalarType>(2) * tMyControl[0] - static_cast<ScalarType>(2);
        tMyOutput[1] = static_cast<ScalarType>(200) * (tMyControl[1] - (tMyControl[0] * tMyControl[0]));
    }
    /*!
     * Compute Rosenbrock Hessian times vector:
     * */
    void hessian(const locus::MultiVector<ScalarType, OrdinalType> & aState,
                 const locus::MultiVector<ScalarType, OrdinalType> & aControl,
                 const locus::MultiVector<ScalarType, OrdinalType> & aVector,
                 locus::MultiVector<ScalarType, OrdinalType> & aOutput)
    {
        assert(aOutput.getNumVectors() == static_cast<OrdinalType>(1));
        assert(aVector.getNumVectors() == static_cast<OrdinalType>(1));
        assert(aControl.getNumVectors() == static_cast<OrdinalType>(1));

        const OrdinalType tVectorIndex = 0;
        locus::Vector<ScalarType, OrdinalType> & tMyOutput = aOutput[tVectorIndex];
        const locus::Vector<ScalarType, OrdinalType> & tMyVector = aVector[tVectorIndex];
        const locus::Vector<ScalarType, OrdinalType> & tMyControl = aControl[tVectorIndex];

        tMyOutput[0] = ((static_cast<ScalarType>(2)
                - static_cast<ScalarType>(400) * (tMyControl[1] - (tMyControl[0] * tMyControl[0]))
                + static_cast<ScalarType>(800) * (tMyControl[0] * tMyControl[0])) * tMyVector[0])
                - (static_cast<ScalarType>(400) * tMyControl[0] * tMyVector[1]);
        tMyOutput[1] = (static_cast<ScalarType>(-400) * tMyControl[0] * tMyVector[0])
                + (static_cast<ScalarType>(200) * tMyVector[1]);
    }
    std::shared_ptr<locus::Criterion<ScalarType, OrdinalType>> create() const
    {
        std::shared_ptr<locus::Criterion<ScalarType, OrdinalType>> tOutput =
                std::make_shared<locus::Rosenbrock<ScalarType, OrdinalType>>();
        return (tOutput);
    }

private:
    Rosenbrock(const locus::Rosenbrock<ScalarType, OrdinalType> & aRhs);
    locus::Rosenbrock<ScalarType, OrdinalType> & operator=(const locus::Rosenbrock<ScalarType, OrdinalType> & aRhs);
};

} // namespace locus

#endif /* LOCUS_ROSENBROCK_HPP_ */
