/*
 * DOTk_OrthogonalProjectionFactory.hpp
 *
 *  Created on: Nov 3, 2014
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_ORTHOGONALPROJECTIONFACTORY_HPP_
#define DOTK_ORTHOGONALPROJECTIONFACTORY_HPP_

namespace dotk
{

class DOTk_Primal;
class DOTk_OrthogonalProjection;

class DOTk_OrthogonalProjectionFactory
{
public:
    DOTk_OrthogonalProjectionFactory(size_t krylov_subspace_dim_, dotk::types::projection_t type_);
    ~DOTk_OrthogonalProjectionFactory();

    std::string getWarningMsg() const;
    size_t getKrylovSubspaceDim() const;
    dotk::types::projection_t getFactoryType() const;

    void buildGramSchmidt(const std::tr1::shared_ptr<dotk::DOTk_Primal> & primal_,
                          std::tr1::shared_ptr<dotk::DOTk_OrthogonalProjection> & projection_);
    void buildArnoldiProjection(const std::tr1::shared_ptr<dotk::DOTk_Primal> & primal_,
                                std::tr1::shared_ptr<dotk::DOTk_OrthogonalProjection> & projection_);
    void build(const std::tr1::shared_ptr<dotk::DOTk_Primal> & primal_,
               std::tr1::shared_ptr<dotk::DOTk_OrthogonalProjection> & projection_);

private:
    void setWarningMsg(const std::string & msg_);
    void setFactoryType(dotk::types::projection_t type_);

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
