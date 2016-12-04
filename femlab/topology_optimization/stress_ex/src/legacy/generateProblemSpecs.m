function [usr_par] = generateProblemSpecs(usr_par)

%%%%%%%%%%% General input data
usr_par.spaceDim      = 2;                  % physical spatial dimensions for cell (element)
usr_par.sideDim       = 1;                  % physical spatial dimensions for subcell (side)
usr_par.cellType      = 'Triangle';         % cell (element) topology 
usr_par.nVert         = 3;                  % number of cell vertices
usr_par.cubDegree     = 3;                  % max. degree of the polynomial that can be represented by the basis
usr_par.numFields     = 3;                  % number of fields (i.e. number of basis functions)
usr_par.sideType      = 'Line';             % subcell (side) topology
usr_par.nVertSide     = 2;                  % number of subcell vertices
usr_par.cubDegreeSide = 2;                  % subcells max. degree of the polynomial that can be represented by the basis
usr_par.numFieldsSide = 2;                  % subcells number of fields (i.e. number of basis functions)
usr_par.numSides      = 3;                  % number of sides for one cell
usr_par.numDof        = ...
    usr_par.spaceDim*usr_par.numFields;     % number of element degrees of freedom

% Steel
E = 10;
nu = 0.3;
factor = 1e-6;
usr_par.B = E / (3*(1 - (2*nu)));  % shear modulus
usr_par.G = E / (2*(1+nu));  % bulk modulus
usr_par.Bmin = factor * usr_par.B;
usr_par.Gmin = factor * usr_par.G;

end