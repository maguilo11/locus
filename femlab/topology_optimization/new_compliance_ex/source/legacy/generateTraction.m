function [usr_par] = generateTraction(usr_par)
%
%  generateTraction(usr_par)
%
%  PURPOSE: compute Neumann boundary data (i.e. surface tractions)
%
%  Input:
%           usr_par    contains all input parameters, as well as additional
%                      computed quantities
%
%  Output:
%
%           tau        neumann boundary data (i.e. surface tractions) at
%                      the physical frame
%
%  AUTHOR:  Miguel Aguilo
%           Sandia National Laboratories
%           August 17, 2011

spaceDim = size(usr_par.cubPointsSide1PhysCoord,1);
numCubPointsSide1 = size(usr_par.cubPointsSide1PhysCoord,2);
numCells = size(usr_par.cubPointsSide1PhysCoord,3);
numDof = usr_par.numDof;

%%%%%%%%%%% build right hand side
f = zeros(spaceDim, numCubPointsSide1, numCells);

f(1,:,usr_par.NeumannCellsUS) = 0e0;
f(2,:,usr_par.NeumannCellsUS) = 1e0;

%%%%%%%%%%% integrate right hand side
tau = zeros(numDof, numCells);
intrepid_integrate(tau, f, usr_par.US_wNmat, 'COMP_BLAS');

usr_par.f = tau;

%%%%%%%%%%%% Generate dirichlet boundary data
usr_par.u_dirichlet = zeros(2*size(usr_par.mesh.p,1),1);

end
