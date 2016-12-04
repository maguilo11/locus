function [ObjectiveFunctionOperators] = getObjectiveFunctionOperators()
ObjectiveFunctionOperators.Fval=@(u,z) Fval(u,z);
ObjectiveFunctionOperators.F_u=@(u,z) F_u(u,z);
ObjectiveFunctionOperators.F_z=@(u,z) F_z(u,z);
ObjectiveFunctionOperators.F_uu=@(u,z,du) F_uu(u,z,du);
ObjectiveFunctionOperators.F_uz=@(u,z,dz) F_uz(u,z,dz);
ObjectiveFunctionOperators.F_zz=@(u,z,dz) F_zz(u,z,dz);
ObjectiveFunctionOperators.F_zu=@(u,z,du) F_zu(u,z,du);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [fval] = Fval(u,z)
global GLB_INVP;

z_all = ones(GLB_INVP.nVertGrid,1);
z_all(GLB_INVP.FreeVertex) = z;

%%%%%%%%%%% evaluate material model dependent density function
[density] = ...
    getMaterialModelRelatedQuantity(z_all, GLB_INVP.model_t, 'zero');

%%%%%%%%%%% compute updated shear and bulk modulus in terms of density
Gnew = density.*GLB_INVP.G;
Bnew = density.*GLB_INVP.B;

%%%%%%%%%%% compute compliance term
u_all = GLB_INVP.u_dirichlet;
u_all(GLB_INVP.FreeNodes) = u;
K = computeStiffnessMatrix(Gnew, Bnew);
compliance = (GLB_INVP.theta/2) * (u_all'*K*u_all);

%%%%%%%%%%% compute volume term
volume = ( (1/2) * ( GLB_INVP.alpha*(z_all'*(GLB_INVP.Ms*z_all)) ...
    - GLB_INVP.VolumeLimit ) );

switch GLB_INVP.reg
    case{'Tikhonov'}
        reg = 0.5 * GLB_INVP.beta * ...
            (z'*(GLB_INVP.Ms(GLB_INVP.FreeVertex,GLB_INVP.FreeVertex)*z));
    case{'TV'}
        reg = 0.5*GLB_INVP.beta * sqrt( z' * ...
            (GLB_INVP.Ss(GLB_INVP.FreeVertex,GLB_INVP.FreeVertex) * z) ...
            + GLB_INVP.gamma );
end
fval = compliance + volume + reg;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [f_z] = F_z(u,z)
global GLB_INVP;

z_all = ones(GLB_INVP.nVertGrid,1);
z_all(GLB_INVP.FreeVertex) = z;

%%%%%%%%%%% evaluate material model dependent density function
[density] = ...
    getMaterialModelRelatedQuantity(z_all, GLB_INVP.model_t, 'first');

%%%%%%%%%%% compute updated shear and bulk modulus in terms of density
Gnew = density.*GLB_INVP.G;
Bnew = density.*GLB_INVP.B;

%%%%%%%%%%% compute compliance term
[K_z] = computePerturbedStiffness(u,Gnew,Bnew);
u_all = GLB_INVP.u_dirichlet;
u_all(GLB_INVP.FreeNodes) = u;
compliance = (GLB_INVP.theta/2) * (u_all'*K_z(:,GLB_INVP.FreeVertex))';

%%%%%%%%%%% compute volume term
volume = GLB_INVP.alpha * (GLB_INVP.Ms(GLB_INVP.FreeVertex,:)*z_all);

switch GLB_INVP.reg
    case{'Tikhonov'}
        reg = GLB_INVP.beta * ...
            (GLB_INVP.Ms(GLB_INVP.FreeVertex,GLB_INVP.FreeVertex)*z);
    case{'TV'}
        reg = GLB_INVP.beta * 0.5 * ( 1.0 / sqrt( z' * ...
            (GLB_INVP.Ss(GLB_INVP.FreeVertex,GLB_INVP.FreeVertex) * z) ...
            + GLB_INVP.gamma ) ) * ...
            (GLB_INVP.Ss(GLB_INVP.FreeVertex,GLB_INVP.FreeVertex) * z);
end
f_z = compliance + volume + reg;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [f_u] = F_u(u,z)
global GLB_INVP;

z_all = ones(GLB_INVP.nVertGrid,1);
z_all(GLB_INVP.FreeVertex) = z;

%%%%%%%%%%% evaluate material model dependent density function
[density] = ...
    getMaterialModelRelatedQuantity(z_all, GLB_INVP.model_t, 'zero');

%%%%%%%%%%% compute updated shear and bulk modulus in terms of density
Gnew = density.*GLB_INVP.G;
Bnew = density.*GLB_INVP.B;

%%%%%%%%%%% compute compliance term
K = computeStiffnessMatrix(Gnew, Bnew);
u_all = GLB_INVP.u_dirichlet;
u_all(GLB_INVP.FreeNodes) = u;
ff_u = GLB_INVP.theta * K*u_all;
f_u = ff_u(GLB_INVP.FreeNodes);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function  [fzz_x_dz] = F_zz(u,z,dz)
global GLB_INVP;

z_all = ones(GLB_INVP.nVertGrid,1);
z_all(GLB_INVP.FreeVertex) = z;

%%%%%%%%%%% evaluate material model dependent density function
[density] = ...
    getMaterialModelRelatedQuantity(z_all, GLB_INVP.model_t, 'second');

%%%%%%%%%%% compute updated shear and bulk modulus in terms of density
Gnew = density.*GLB_INVP.G;
Bnew = density.*GLB_INVP.B;

%%%%%%%%%%% compute compliance term
[K_zz] = computePerturbedStiffness(u,Gnew,Bnew);
u_all = GLB_INVP.u_dirichlet;
u_all(GLB_INVP.FreeNodes) = u;
compliance = (GLB_INVP.theta/2) * ((u_all'*K_zz(:,GLB_INVP.FreeVertex))');


%%%%%%%%%%% compute volume term
volume = GLB_INVP.alpha * ...
    (GLB_INVP.Ms(GLB_INVP.FreeVertex,GLB_INVP.FreeVertex)*dz);

switch GLB_INVP.reg
    case{'Tikhonov'}
        reg = GLB_INVP.beta * ...
            (GLB_INVP.Ms(GLB_INVP.FreeVertex,GLB_INVP.FreeVertex)*dz);
    case{'TV'}
        S_k  = GLB_INVP.Ss(GLB_INVP.FreeVertex,GLB_INVP.FreeVertex) * z;
        St_k = GLB_INVP.Ss(GLB_INVP.FreeVertex,GLB_INVP.FreeVertex)' * z;
        reg = (-0.5 * GLB_INVP.beta * ( ...
            ( (z' * S_k + GLB_INVP.gamma)^(-3/2) ) * ...
            ((St_k'*dz)*S_k) ...
            - ((1.0 / sqrt(z' * S_k + GLB_INVP.gamma)) * ...
            (GLB_INVP.Ss(GLB_INVP.FreeVertex,GLB_INVP.FreeVertex)*dz)) ));
end
fzz_x_dz = compliance + volume + reg;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function  [fuu_x_du] = F_uu(u,z,du)
global GLB_INVP;

z_all = ones(GLB_INVP.nVertGrid,1);
z_all(GLB_INVP.FreeVertex) = z;

%%%%%%%%%%% evaluate material model dependent density function
[density] = ...
    getMaterialModelRelatedQuantity(z_all, GLB_INVP.model_t, 'zero');

%%%%%%%%%%% compute updated shear and bulk modulus in terms of density
Gnew = density.*GLB_INVP.G;
Bnew = density.*GLB_INVP.B;

%%%%%%%%%%% compute compliance term
K = computeStiffnessMatrix(Gnew, Bnew);
fuu_x_du = GLB_INVP.theta * ...
    K(GLB_INVP.FreeNodes,GLB_INVP.FreeNodes)*du;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function  [fzu_x_du] = F_zu(u,z,du)
global GLB_INVP;

z_all = ones(GLB_INVP.nVertGrid,1);
z_all(GLB_INVP.FreeVertex) = z;

%%%%%%%%%%% evaluate material model dependent density function
[density] = ...
    getMaterialModelRelatedQuantity(z_all, GLB_INVP.model_t, 'first');

%%%%%%%%%%% compute updated shear and bulk modulus in terms of density
Gnew = density.*GLB_INVP.G;
Bnew = density.*GLB_INVP.B;

%%%%%%%%%%% compute compliance term
K_z = computePerturbedStiffness(u,Gnew,Bnew);
fzu_x_du = GLB_INVP.theta * ...
    (K_z(GLB_INVP.FreeNodes,GLB_INVP.FreeVertex)')*du;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function  [fuz_x_dz] = F_uz(u,z,dz)
global GLB_INVP;

z_all = ones(GLB_INVP.nVertGrid,1);
z_all(GLB_INVP.FreeVertex) = z;

%%%%%%%%%%% evaluate material model dependent density function
[density] = ...
    getMaterialModelRelatedQuantity(z_all, GLB_INVP.model_t, 'first');

%%%%%%%%%%% compute updated shear and bulk modulus in terms of density
Gnew = density.*GLB_INVP.G;
Bnew = density.*GLB_INVP.B;

%%%%%%%%%%% compute compliance term
K_z = computePerturbedStiffness(u,Gnew,Bnew);
fuz_x_dz = GLB_INVP.theta * ...
    (K_z(GLB_INVP.FreeNodes,GLB_INVP.FreeVertex))*dz;
end
