function [EqualityConstraintOperators] = getEqualityConstraintOperators()
EqualityConstraintOperators.Cval=@(u,z) Cval(u,z);
EqualityConstraintOperators.C_u=@(u,z,du) C_u(u,z,du);
EqualityConstraintOperators.C_z=@(u,z,dz) C_z(u,z,dz);
EqualityConstraintOperators.adjC_u=@(u,z,lambda) adjC_u(u,z,lambda);
EqualityConstraintOperators.adjC_z=@(u,z,lambda) adjC_z(u,z,lambda);
EqualityConstraintOperators.adjC_uu=@(u,z,lambda,du) adjC_uu(u,z,lambda,du);
EqualityConstraintOperators.adjC_uz=@(u,z,lambda,dz) adjC_uz(u,z,lambda,dz);
EqualityConstraintOperators.adjC_zz=@(u,z,lambda,dz) adjC_zz(u,z,lambda,dz);
EqualityConstraintOperators.adjC_zu=@(u,z,lambda,du) adjC_zu(u,z,lambda,du);
EqualityConstraintOperators.setRestoreOperators = ...
    @(restore) setRestoreOperators(restore);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [cval] = Cval(u,z)
global GLB_INVP;

z_all = ones(GLB_INVP.nVertGrid,1);
z_all(GLB_INVP.FreeVertex) = z;

% Separate state and control variables.
u_new = GLB_INVP.u_dirichlet;
u_new(GLB_INVP.FreeNodes) = u;

%%%%%%%%%%% evaluate material model dependent density function
[density] = ...
    getMaterialModelRelatedQuantity(z_all, GLB_INVP.model_t, 'zero');

%%%%%%%%%%% compute updated shear and bulk modulus in terms of density
Gnew = density.*GLB_INVP.G;
Bnew = density.*GLB_INVP.B;

%%%%%%%%%%% compute compliance term
K = computeStiffnessMatrix(Gnew, Bnew);

%%%%%%%%%%% build global rhs vector
rhs_reshape = reshape(GLB_INVP.f, 1, numel(GLB_INVP.f));
rhs_mat = sparse(GLB_INVP.iVecIdxDof, GLB_INVP.iVecIdxDof, rhs_reshape);
rhs = spdiags(rhs_mat,0);

%%%%%%%%%%% compute k*grad(u)*grad(phi_j) - f*phi_j
Cval = K*u_new - rhs;

%%%%%%%%%%% build global rhs vector
cval = Cval(GLB_INVP.FreeNodes);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [Cu_x_du] = C_u(u,z,du)

global GLB_INVP;

z_all = ones(GLB_INVP.nVertGrid,1);
z_all(GLB_INVP.FreeVertex) = z;

%%%%%%%%%%% evaluate material model dependent density function
[density] = ...
    getMaterialModelRelatedQuantity(z_all, GLB_INVP.model_t, 'zero');

%%%%%%%%%%% compute updated shear and bulk modulus in terms of density
Gnew = density.*GLB_INVP.G;
Bnew = density.*GLB_INVP.B;

K = computeStiffnessMatrix(Gnew, Bnew);

GLB_INVP.Jac_u = K(GLB_INVP.FreeNodes,GLB_INVP.FreeNodes);
Cu_x_du = K(GLB_INVP.FreeNodes,GLB_INVP.FreeNodes)*du;

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [Cz_x_dz] = C_z(u,z,dz)
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

GLB_INVP.Jac_z = K_z(GLB_INVP.FreeNodes,GLB_INVP.FreeVertex);
Cz_x_dz = GLB_INVP.Jac_z * dz;

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [adjCu_x_lambda] = adjC_u(u,z,lambda)
global GLB_INVP;

z_all = ones(GLB_INVP.nVertGrid,1);
z_all(GLB_INVP.FreeVertex) = z;

%%%%%%%%%%% evaluate material model dependent density function
[density] = ...
    getMaterialModelRelatedQuantity(z_all, GLB_INVP.model_t, 'zero');

%%%%%%%%%%% compute updated shear and bulk modulus in terms of density
Gnew = density.*GLB_INVP.G;
Bnew = density.*GLB_INVP.B;

GLB_INVP.JacT_u = computeStiffnessMatrix(Gnew, Bnew);

GLB_INVP.JacT_u = GLB_INVP.JacT_u(GLB_INVP.FreeNodes,GLB_INVP.FreeNodes)';
adjCu_x_lambda = GLB_INVP.JacT_u*lambda;

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [adjCz_x_lambda] = adjC_z(u,z,lambda)
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
[GLB_INVP.JacT_z] = computePerturbedStiffness(u,Gnew,Bnew);

GLB_INVP.JacT_z = GLB_INVP.JacT_z(GLB_INVP.FreeNodes,GLB_INVP.FreeVertex)';
adjCz_x_lambda = GLB_INVP.JacT_z * lambda;

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [adjCuu_x_du] = adjC_uu(u,z,lambda,du)
adjCuu_x_du=zeros(size(u));
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [adjCuz_x_dz] = adjC_uz(u,z,lambda,dz)
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
K_z = computePerturbedStiffness(lambda,Gnew,Bnew);
adjCuz_x_dz = K_z(GLB_INVP.FreeNodes,GLB_INVP.FreeVertex)*dz;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [adjCzz_x_dz] = adjC_zz(u,z,lambda,dz)
adjCzz_x_dz=zeros(size(z));
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [adjCzu_x_du] = adjC_zu(u,z,lambda,du)
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
K_z = computePerturbedStiffness(lambda,Gnew,Bnew);
adjCzu_x_du = (K_z(GLB_INVP.FreeNodes,GLB_INVP.FreeVertex)')*du;

end

function [check] = setRestoreOperators(restore)
global GLB_INVP;
GLB_INVP.restore_operators=restore;
check = GLB_INVP.restore_operators;
end

