function [struc] = setMaterialProperties(struc)

if(struc.multi_material==false)
    % Set material properties
    E = 1;
    nu = 0.3;
    %
    struc.num_materials = 1;
    struc.min_stiffness = 1e-9;
    struc.B = E / (3*(1 - (2*nu)));  % bulk modulus
    struc.G = E / (2*(1+nu));          % shear modulus
    struc.G = struc.G*ones(struc.nVertGrid,1);
    struc.B = struc.B*ones(struc.nVertGrid,1);
    % Limit on material volume
    struc.VolumeFraction = 0.4;
else
    % Set material properties
    E = [0.9; 0.8];
    nu = [0.3; 0.3];
    G = E ./ (2.*(1+nu));          % shear modulus
    B = E ./ (3.*(1 - (2*nu)));    % bulk modulus
    struc.num_materials = length(E);
    struc.min_stiffness = [1e-9;1e-9];
    struc.G = ones(struc.nVertGrid,size(E,1));
    struc.B = ones(struc.nVertGrid,size(E,1));
    for index=1:struc.num_materials
        struc.G(:,index) = G(index)*struc.G(:,index);
        struc.B(:,index) = B(index)*struc.B(:,index);
    end
    % Limit on material volume
    struc.VolumeFraction = 0.4;
end

end