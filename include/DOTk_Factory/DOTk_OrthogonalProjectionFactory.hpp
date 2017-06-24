/*
 * DOTk_OrthogonalProjectionFactory.hpp
 *
 *  Created on: Nov 3, 2014
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_ORTHOGONALPROJECTIONFACTORY_HPP_
#define DOTK_ORTHOGONALPROJECTIONFACTORY_HPP_

#include <string>
#include <memory>

namespace dotk
{

class DOTk_Primal;
class DOTk_OrthogonalProjection;

class DOTk_OrthogonalProjectionFactory
{
public:
    DOTk_OrthogonalProjectionFactory(size_t aKrylovSubspaceDim, dotk::types::projection_t aType);
    ~DOTk_OrthogonalProjectionFactory();

    std::string getWarningMsg() const;
    size_t getKrylovSubspaceDim() const;
    dotk::types::projection_t getFactoryType() const;

    void buildGramSchmidt(const std::shared_ptr<dotk::DOTk_Primal> & aPrimal,
                          std::shared_ptr<dotk::DOTk_OrthogonalProjection> & aOutput);
    void buildArnoldiProjection(const std::shared_ptr<dotk::DOTk_Primal> & aPrimal,
                                std::shared_ptr<dotk::DOTk_OrthogonalProjection> & aOutput);
    void build(const std::shared_ptr<dotk::DOTk_Primal> & aPrimal,
               std::shared_ptr<dotk::DOTk_OrthogonalProjection> & aOutput);

private:
    void setWarningMsg(const std::string & aMsg);
    void setFactoryType(dotk::types::projection_t aType);

private:
    std::string m_WarningMsg;
    size_t m_KrylovSubspaceDimension;
    dotk::types::projection_t m_FactoryType;

private:
    DOTk_OrthogonalProjectionFactory(const dotk::DOTk_OrthogonalProjectionFactory &);
    dotk::DOTk_OrthogonalProjectionFactory & operator=(const dotk::DOTk_OrthogonalProjectionFactory &);
};

}

#endif /* DOTK_ORTHOGONALPROJECTIONFACTORY_HPP_ */
