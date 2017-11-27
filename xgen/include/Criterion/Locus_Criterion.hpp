/*
 * Locus_Criterion.hpp
 *
 *  Created on: Oct 6, 2017
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef LOCUS_CRITERION_HPP_
#define LOCUS_CRITERION_HPP_

#include <memory>

namespace locus
{

template<typename ScalarType, typename OrdinalType>
class MultiVector;

template<typename ScalarType, typename OrdinalType = size_t>
class Criterion
{
public:
    virtual ~Criterion()
    {
    }

    /*!
     * Evaluates criterion of type f(\mathbf{u}(\mathbf{z}),\mathbf{z})\colon\mathbb{R}^{n_u}\times\mathbb{R}^{n_z}
     * \rightarrow\mathbb{R}, where u denotes the state and z denotes the control variables. This criterion
     * is typically associated with nonlinear programming optimization problems. For instance, PDE constrasize_t
     * optimization problems.
     *  Parameters:
     *    \param In
     *          aControl: control variables
     *
     *  \return Objective function value
     **/
    virtual ScalarType value(const locus::MultiVector<ScalarType, OrdinalType> & aControl) = 0;
    /*!
     * Computes the gradient of a criterion of type f(\mathbf{u}(\mathbf{z}),\mathbf{z})\colon\mathbb{R}^{n_u}
     * \times\mathbb{R}^{n_z}\rightarrow\mathbb{R}, where u denotes the state and z denotes the control variables.
     * This criterion is typically associated with nonlinear programming optimization problems. For instance, PDE
     * constraint optimization problems.
     *  Parameters:
     *    \param In
     *          aControl: control variables
     *    \param Out
     *          aOutput: gradient
     **/
    virtual void gradient(const locus::MultiVector<ScalarType, OrdinalType> & aControl,
                          locus::MultiVector<ScalarType, OrdinalType> & aOutput) = 0;
    /*!
     * Computes the application of a vector to the Hessian of a criterion of type f(\mathbf{u}(\mathbf{z}),\mathbf{z})
     * \colon\mathbb{R}^{n_u}\times\mathbb{R}^{n_z}\rightarrow\mathbb{R}, where u denotes the state and z denotes the
     * control variables. This criterion is typically associated with nonlinear programming optimization problems.
     * For instance, PDE constraint optimization problems.
     *  Parameters:
     *    \param In
     *          aControl: control variables
     *    \param In
     *          aVector:  direction vector
     *    \param Out
     *          aOutput:  Hessian times direction vector
     **/
    virtual void hessian(const locus::MultiVector<ScalarType, OrdinalType> & aControl,
                         const locus::MultiVector<ScalarType, OrdinalType> & aVector,
                         locus::MultiVector<ScalarType, OrdinalType> & aOutput)
    {
    }
    //! Creates an object of type locus::Criterion
    virtual std::shared_ptr<locus::Criterion<ScalarType, OrdinalType>> create() const = 0;
};

}

#endif /* LOCUS_CRITERION_HPP_ */
