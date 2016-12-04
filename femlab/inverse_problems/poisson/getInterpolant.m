function [u] = getInterpolant(usr_par,d)
%
%   getInterpolant(usr_par,d)
%
%   PURPOSE: Finds an interpolating polynomial for a piece of data d that
%            obeys the boundary conditions.
%       
%   The problem domain Omega is the square (xmin,xmax)x(ymin,ymax).
%
%   Input:
%           usr_par    contains all input parameters, as well as additional
%                      computed quantities
%           d          data (evaluated at cubature points)
%  
%   Output:
%   
%           u          Interpolated solution
%
%   AUTHOR:  Miguel Aguilo
%            Dookie's Coorp
%            April 1, 2011

nVertGrid     = usr_par.nVertGrid;

%%%%%%%%%%% Initialization of free nodes and lhs (i.e. solution) vectors.
u = zeros(nVertGrid,1);

%%%%%%%%%%% grab the integrated rhs 
cell_rhs=zeros(usr_par.numFields,usr_par.numCells);
intrepid_integrate(cell_rhs,d, ...
    usr_par.weighted_transformed_val_at_cub_points, 'COMP_BLAS');

%%%%%%%%%%% build global rhs vector
cell_rhs = reshape(cell_rhs, 1, numel(cell_rhs));
rhs_mat = sparse(usr_par.iVecIdx, usr_par.iVecIdx, cell_rhs);
b = spdiags(rhs_mat,0);

%%%%%%%%%%% Computation of the solution
u = usr_par.M \ b;
