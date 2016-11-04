/*
 * DOTk_NonlinearCGFactory.hpp
 *
 *  Created on: Sep 15, 2014
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_NONLINEARCGFACTORY_HPP_
#define DOTK_NONLINEARCGFACTORY_HPP_

#include "DOTk_Types.hpp"

namespace dotk
{

class DOTk_LinearOperator;
class DOTk_DescentDirection;

class DOTk_NonlinearCGFactory
{
public:
    DOTk_NonlinearCGFactory();
    ~DOTk_NonlinearCGFactory();

    void buildDanielsNlcg(const std::tr1::shared_ptr<dotk::DOTk_LinearOperator> & hessian_,
                          std::tr1::shared_ptr<dotk::DOTk_DescentDirection> & dir_);
    void buildFletcherReevesNlcg(std::tr1::shared_ptr<dotk::DOTk_DescentDirection> & dir_);
    void buildPolakRibiereNlcg(std::tr1::shared_ptr<dotk::DOTk_DescentDirection> & dir_);
    void buildHestenesStiefelNlcg(std::tr1::shared_ptr<dotk::DOTk_DescentDirection> & dir_);
    void buildConjugateDescentNlcg(std::tr1::shared_ptr<dotk::DOTk_DescentDirection> & dir_);
    void buildHagerZhangNlcg(std::tr1::shared_ptr<dotk::DOTk_DescentDirection> & dir_);
    void buildDaiLiaoNlcg(std::tr1::shared_ptr<dotk::DOTk_DescentDirection> & dir_);
    void buildDaiYuanNlcg(std::tr1::shared_ptr<dotk::DOTk_DescentDirection> & dir_);
    void buildDaiYuanHybridNlcg(std::tr1::shared_ptr<dotk::DOTk_DescentDirection> & dir_);
    void buildPerryShannoNlcg(std::tr1::shared_ptr<dotk::DOTk_DescentDirection> & dir_);
    void buildLiuStoreyNlcg(std::tr1::shared_ptr<dotk::DOTk_DescentDirection> & dir_);

private:
    DOTk_NonlinearCGFactory(const dotk::DOTk_NonlinearCGFactory &);
    DOTk_NonlinearCGFactory operator=(const dotk::DOTk_NonlinearCGFactory &);
};

}

#endif /* DOTK_NONLINEARCGFACTORY_HPP_ */
