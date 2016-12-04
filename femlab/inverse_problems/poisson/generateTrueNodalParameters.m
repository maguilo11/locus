function control = generateTrueNodalParameters(struc)
% function control = generateTrueNodalParameters(usr_par)
%
% Sets up material properties on the mesh.
%

%%%%%%%%%%% get the material properties

choice = struc.pdata;

mtol = 1e-8;

%%%%%%%%%%% get the material properties

switch choice
    
    case 'egg' % (extruded in 3D)
        
        % x and y coordinates of the circular region with different material properties
        center1    = [0.00  0.00];
        center2    = [0.00  0.10];
        center3    = [0.00  -0.35];
        
        % radius of the circular region with different material properties
        radius1    = 0.5+mtol;
        radius2    = 0.2+mtol;
        radius3    = 0.12+mtol;
        
        % difussion constant value for each material region
        mat_prop  = [0.5 0.8 0.6 0.3];
        
        %%%%%%%%%%% get diffusion spatial distribution
        layer1 = ( ( struc.mesh.p(:,1) - center1(1) ).^2/0.3 + ...
            ( struc.mesh.p(:,2) - center1(2) ).^2.*(0.7*struc.mesh.p(:,2)+1) ) <= radius1^2;
        layer2 = ( ( struc.mesh.p(:,1) - center2(1) ).^2 + ...
            ( struc.mesh.p(:,2) - center2(2) ).^2 ) <= radius2^2;
        layer3 = ( ( struc.mesh.p(:,1) - center3(1) ).^2 + ...
            ( struc.mesh.p(:,2) - center3(2) ).^2 ) <= radius3^2;
        layer4 = ~(layer1 | layer2 | layer3);
        control = mat_prop(1)*layer1 + ...
            (mat_prop(2)-mat_prop(1))*layer2 + ...
            (mat_prop(3)-mat_prop(1))*layer3 + ...
            mat_prop(4)*layer4;
        
        
    case 'sphere' % 1D, 2D, 3D sphere
        
        % coordinates of the spherical region with different material properties
        center    = [-0.25 -0.25 0.4];
        
        % radius of the circular region with different material properties
        radius    = 0.3+mtol;
        
        % value of difussion constant for each material region
        mat_prop  = [0.4 0.1];
        
        %%%%%%%%%%% compute nodal diffusion field
        val = 0;
        for i=1:size(struc.mesh.p,2)
            val = val + ( struc.mesh.p(:,i) - center(i) ).^2;
        end
        layer1 = val <= radius^2;
        layer2 = ~layer1;
        control = mat_prop(1)*layer1 + mat_prop(2)*layer2;
        
        
    case 'sphere_in_cube' % 1D, 2D, 3D sphere in cube
        
        % coordinates of the centers of the spherical and cube regions
        center    = [-0.2 -0.2 0.3];
        
        % radius of the spherical region
        radius1   = 0.2+mtol;
        
        % half-diagonal of the cube region
        radius2   = 0.5+mtol;
        
        % value of difussion constant for each material region: sphere, cube, rest
        mat_prop  = [0.3 0.15 0.1];
        
        %%%%%%%%%%% compute nodal diffusion field
        
        val = 0;
        for i=1:size(struc.mesh.p,2)
            val = val + ( struc.mesh.p(:,i) - center(i) ).^2;
        end
        layer1 = val <= radius1^2;
        
        val = 0;
        for i=1:size(struc.mesh.p,2)
            val = val + abs(struc.mesh.p(:,i) - center(i));
        end
        layer2 = val <= radius2;
        layer2 = layer2 & ~layer1;
        
        layer3 = ~(layer2 | layer1);
        
        control = mat_prop(1)*layer1 + mat_prop(2)*layer2 + mat_prop(3)*layer3;
        
end
