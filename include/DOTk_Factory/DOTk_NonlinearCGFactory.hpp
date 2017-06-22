/*
 * DOTk_NonlinearCGFactory.hpp
 *
 *  Created on: Sep 15, 2014
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_NONLINEARCGFACTORY_HPP_
#define DOTK_NONLINEARCGFACTORY_HPP_

#include <memory>
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

    void buildDanielsNlcg(const std::shared_ptr<dotk::DOTk_LinearOperator> & hessian_,
                          std::shared_ptr<dotk::DOTk_DescentDirection> & dir_);
    void buildFletcherReevesNlcg(std::shared_ptr<dotk::DOTk_DescentDirection> & dir_);
    void buildPolakRibiereNlcg(std::shared_ptr<dotk::DOTk_DescentDirection> & dir_);
    void buildHestenesStiefelNlcg(std::shared_ptr<dotk::DOTk_DescentDirection> & dir_);
    void buildConjugateDescentNlcg(std::shared_ptr<dotk::DOTk_DescentDirection> & dir_);
    void buildHagerZhangNlcg(std::shared_ptr<dotk::DOTk_DescentDirection> & dir_);
    void buildDaiLiaoNlcg(std::shared_ptr<dotk::DOTk_DescentDirection> & dir_);
    void buildDaiYuanNlcg(std::shared_ptr<dotk::DOTk_DescentDirection> & dir_);
    void buildDaiYuanHybridNlcg(std::shared_ptr<dotk::DOTk_DescentDirection> & dir_);
    void buildPerryShannoNlcg(std::shared_ptr<dotk::DOTk_DescentDirection> & dir_);
    void buildLiuStoreyNlcg(std::shared_ptr<dotk::DOTk_DescentDirection> & dir_);

private:
    DOTk_NonlinearCGFactory(const dotk::DOTk_NonlinearCGFactory &);
    DOTk_NonlinearCGFactory operator=(const dotk::DOTk_NonlinearCGFactory &);
};

}

#endif /* DOTK_NONLINEARCGFACTORY_HPP_ */
