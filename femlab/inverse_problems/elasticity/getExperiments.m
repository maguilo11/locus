function [u_exp] = getExperiments(u_fine, NX_fine, NY_fine, ...
    coarse_level, usr_par)
%
%   getExperiments(u_fine, NX_fine, NY_fine, coarse_level, usr_par)
%
%   PURPOSE: Generate experimental data for the inverse problem from the 
%            fem solution obtain with a finer mesh
%
%   Input:
%            u_fine        fem solution from a finer mesh
%
%            NX_fine       number of elements in the x-direction on the 
%                          fine mesh
%
%            NY_fine       number of elements in the y-direction on the 
%                          fine mesh
%
%            coarse_level  level of coarsening (by a factor of 2 in each 
%                          direction)
%
%            usr_par       contains all input parameters, as well as 
%                          additional computed quantities
%  
%   Output:
%   
%            u_coarse      fem solution obtained with a fine mesh at the 
%                          location of the vertex from the coarse mesh
%
%   AUTHOR:  Miguel Aguilo
%            Denis Ridzal
%            Sandia National Laboratories
%            February 14, 2011

skip_y = 2^coarse_level;

skip_x = (NY_fine+1)*skip_y;

NX_coarse = NX_fine / skip_y;
NY_coarse = NY_fine / skip_y;

array_y = [1 : skip_y : NY_fine+1];

% fem data at each vertex of the coarse mesh (obtain from the fine mesh)
u_coarse_at_phys_pt = zeros(1,(NX_coarse+1)*(NY_coarse+1));

% get u_coarse 
for i=1:NX_coarse+1
    u_coarse_at_phys_pt( (NY_coarse+1)*(i-1) + 1 : (NY_coarse+1)*i ) = ...
      u_fine( array_y + (i-1)*skip_x );    
end

% evaluate solution at the cubature points
%u_coarse_at_phys_pt = u_coarse_at_phys_pt(usr_par.mesh.t');
%u_exp = zeros(usr_par.numCubPoints, usr_par.numCells);
%intrepid_evaluate(u_exp, u_coarse_at_phys_pt, ...
%    usr_par.transformed_val_at_cub_points);
u_exp = u_coarse_at_phys_pt;
end
