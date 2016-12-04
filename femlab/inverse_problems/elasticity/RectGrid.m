
function [mesh] = RectGrid(xmin, xmax, ymin, ymax, nx, ny)
%
%  [mesh] = RectGrid(xmin, xmax, ymin, ymax, nx, ny)
%
%RECTGRID   sets up the grid for piecewise linear elements
%  in a rectangular domain.
%  
%  The grid is constructed by subdividing the  x-interval into
%  nx subintervals and the  y-interval into ny subintervals.
%  This generates a grid with  nx*ny  rectangles. Each rectangle 
%  is then subdivided into two triangles by cutting the rectangle 
%  from bottom left to top right.
%  
%
%
%  Input
%         xmin, xmax  size of the rectangle
%         ymin, ymax
%         nx          number of subintervals on x-interval
%         ny          number of subintervals on y-interval
%
%  Output
%         mesh       structure array with the following fields
%
%         mesh.p     Real nn x 2
%                    array containing the x- and y- coordinates
%                    of the nodes
%
%         mesh.t     Integer nt x 3   
%                    t(i,1:3) contains the indices of the vertices of
%                    triangle i. 
%
%         mesh.d     Integer nt x 6
%                    d(i,1:6) contains the indices of the degrees of 
%                    freedom of traingle i.
%
%         mesh.e     Integer nf x 4
%                    e(i,2:3) contains the indices of the element where
%                    edge i is located.    
%                    e(i,2:3) contains the indices of the vertices of
%                    edge i.
%                    Currently set to zero.
%                    e(i,3) = 1  Dirichlet bdry conds are imposed on edge i
%                    e(i,3) = 2  Neumann bdry conds are imposed on edge i
%                    e(i,3) = 3  Robin bdry conds are imposed on edge i
%                    Reference edge id
%                    edge(i,4) contains the reference subcell id where 
%                    a boundary condition is apply
%                    e(i,5) = 0  edges on low boundary
%                    e(i,5) = 1  edges on top or right boundary
%                    e(i,5) = 2  edges on left boundary
%                    Boundary marker
%                    edge(i,5) contains the boundary marker of edge i.
%                    
%     
%
%  Vertical ordering:
%  The triangles are ordered column wise, for instance:
% 
%    03 -------- 06 -------- 09 -------- 12
%     |  4     /  |  8     /  | 12     /  |
%     |     /     |     /     |     /     |
%     |  /    3   |  /    7   |  /    11  |
%    02 -------- 05 -------- 08 -------- 11
%     |  2     /  |  6     /  | 10     /  |      
%     |     /     |     /     |     /     |
%     |  /    1   |  /    5   |  /     9  |
%    01 -------- 04 -------- 07 -------- 10
%
%  The vertices and midpoints in a triangle are numbered
%  counterclockwise, for example
%          triangle 7: (05, 08, 09)
%          triangle 8: (05, 09, 06)
%
%  (Usually, the local grid.node numbering should not be important.)
%
%  number of triangles: 2*nx*ny,
%  number of vertices:  (nx+1)*(ny+1), 
%
%  AUTHOR:  Miguel Aguilo
%           Denis Ridzal
%           Sandia National Laboratories
%           August 24, 2011

nt = 2*nx*ny;
np = (nx+1)*(ny+1);

% Create arrays
mesh.t = zeros(nt,3);
mesh.p = zeros(np,2);

nxp1 = nx + 1;
nyp1 = ny + 1;

% Create triangles
nt  = 0;
for ix = 1:nx
   for iy = 1:ny
      
      iv  = (ix-1)*nyp1 + iy;
      iv1 = iv + nyp1;
      
      nt = nt + 1;
      mesh.t(nt,1) = iv;
      mesh.t(nt,2) = iv1;
      mesh.t(nt,3) = iv1+1;

      nt = nt+1;
      mesh.t(nt,1) = iv;
      mesh.t(nt,2) = iv1+1;
      mesh.t(nt,3) = iv+1;
  end
end

% Create triangle degrees of freedom array
mesh.d = zeros(nt,6);
mesh.d(:,1:2:end) = (2*mesh.t) - 1;     % dof in x-dir for triangle i
mesh.d(:,2:2:end) = 2*mesh.t;           % dof in y-dir for triangle i

% Create vertex coodinates

hx   = (xmax-xmin)/nx;
hy   = (ymax-ymin)/ny;
x    = xmin;

for ix = 1:nx

  % set coordinates for vertices with fixed 
  % x-coordinate at x
  i1 = (ix-1)*(ny+1)+1;
  i2 = ix*(ny+1);
  mesh.p(i1:i2,1) = x*ones(nyp1,1);
  mesh.p(i1:i2,2) = (ymin:hy:ymax)';
   
  x = x + hx;
end

% set coordinates for vertices with fixed 
% x-coordinate at xmax
i1 = nx*(ny+1)+1;
i2 = (nx+1)*(ny+1);
mesh.p(i1:i2,1) = xmax*ones(nyp1,1);
mesh.p(i1:i2,2) = (ymin:hy:ymax)';
   
% Set grid.edge (edges are numbered counter clock wise starting
% at lower left end).

mesh.e = zeros(2*(nx+ny),5);

% edges on left on left boundary
mesh.e(1:ny,1) = (2:2:2*ny)';
mesh.e(1:ny,2) = (1:ny)';
mesh.e(1:ny,3) = (2:ny+1)';
mesh.e(1:ny,4) = 2;

% edges on top boundary
mesh.e(ny+1:nx+ny,1) = (2*ny:2*ny:2*ny*nx)';
mesh.e(ny+1:nx+ny,2) = (ny+1:ny+1:np-1)';
mesh.e(ny+1:nx+ny,3) = (2*(ny+1):ny+1:np)';
mesh.e(ny+1:nx+ny,4) = 1;

% edges on right boundary
mesh.e(nx+ny+1:nx+2*ny,1) = (2*ny*(nx-1)+1:2:2*nx*ny)';
mesh.e(nx+ny+1:nx+2*ny,2) = (np-ny:np-1)';
mesh.e(nx+ny+1:nx+2*ny,3) = (np-ny+1:np)';
mesh.e(nx+ny+1:nx+2*ny,4) = 1;

% edges on lower boundary
mesh.e(nx+2*ny+1:2*(nx+ny),1) = (1:2*ny:2*ny*(nx-1)+1)';
mesh.e(nx+2*ny+1:2*(nx+ny),2) = (1:ny+1:np-2*ny-1)';
mesh.e(nx+2*ny+1:2*(nx+ny),3) = (ny+2:ny+1:np-ny)';
