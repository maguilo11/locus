function show(elements3,coordinates,u)
%SHOW   Presents two-dimensional piecewise affine function graphically.
%    SHOW(ELEMENTS3,COORDINATES,U) presents a two-dimensional
%    spline function graphically. ELEMENTS3 denotes a set of triangles
%    with dimension (no. of triangles) x 3. The array 
%    includes number of nodes. The nodes have to be counted clockwise.
%    or anti-clockwise. The coordinates of the nodes are stored in an
%    (no. of coordinates) x 2 - dimensional array called COORDINATES.  
%    Its i'th row defines the x- and y- coordinate of the i'th node. U is
%    a (no. of coordinates) x 1 - dimensional array containing in the
%    i'th row the value of the spline function at the i'th node.
%
%    Taken from the code supplementing J. Alberty, C. Carstensen and
%    S. A. Funken "Remarks around 50 lines of Matlab: short finite-
%    element implementation".

%scrsz = get(groot,'ScreenSize');
% **** MBB **** 
%figure('Position',[1 scrsz(4) scrsz(3)/3 scrsz(4)/6.5])
% **** Mitchell/Middle ****
%figure('Position',[1 scrsz(4) scrsz(3)/5 scrsz(4)/6])
% **** Mitchell/Square & Flower ****
%figure('Position',[1 scrsz(4) scrsz(3)/6 scrsz(4)/5])
% **** Mitchell/Cantilever ****
%figure('Position',[1 scrsz(4) scrsz(3)/5 scrsz(4)/6])
%
figure;
hold on;
trisurf(elements3,coordinates(:,1),coordinates(:,2),u','facecolor','interp')
view(0,90)
shading interp;
colormap(jet);
