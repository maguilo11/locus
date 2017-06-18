function [struc] = setMaterialProperties(struc)

% Limit on material volume
struc.VolumeFraction = 0.4;
if(struc.multi_material==false)
    % Set material properties
    E = 1;
    nu = 0.3;
    %
    struc.density = 1;
    struc.num_materials = 1;
    struc.min_stiffness = 1e-9;
    struc.B = E / (3*(1 - (2*nu)));  % bulk modulus
    struc.G = E / (2*(1+nu));          % shear modulus
    struc.G = struc.G*ones(struc.nVertGrid,1);
    struc.B = struc.B*ones(struc.nVertGrid,1);
else
    % Set material properties
    E = [1; 0.8];
    nu = [0.3; 0.3];
    G = E ./ (2.*(1+nu));          % shear modulus
    B = E ./ (3.*(1 - (2*nu)));    % bulk modulus
    struc.density = [1; 0.95];
    struc.num_materials = length(E);
    struc.min_stiffness = [1e-9;1e-9];
    struc.G = ones(struc.nVertGrid,size(E,1));
    struc.B = ones(struc.nVertGrid,size(E,1));
    for index=1:struc.num_materials
        struc.G(:,index) = G(index)*struc.G(:,index);
        struc.B(:,index) = B(index)*struc.B(:,index);
    end
end

end