function [struc] = setMaterialProperties(struc)

% Set material properties
if(struc.multi_material==false)
    E = 1;
    nu = 0.3;
    stiffness_bound = 1e-9;
    %
    struc.B = E / (3*(1 - (2*nu)));  % bulk modulus
    struc.G = E / (2*(1+nu));          % shear modulus
    struc.G = struc.G*ones(struc.nVertGrid,1);
    struc.B = struc.B*ones(struc.nVertGrid,1);
    struc.Bmin = stiffness_bound * struc.B;
    struc.Gmin = stiffness_bound * struc.G;
else
    E = [0.8; 0.6];
    nu = [0.3; 0.25];
    G = E ./ (2.*(1+nu));          % shear modulus
    B = E ./ (3.*(1 - (2*nu)));    % bulk modulus
    stiffness_bound = [1e-9;1e-9];
    struc.G = ones(struc.nVertGrid,size(E,1));
    struc.B = ones(struc.nVertGrid,size(E,1));
    struc.Gmin = ones(struc.nVertGrid,size(E,1));
    struc.Bmin = ones(struc.nVertGrid,size(E,1));
    for index=1:length(E)
        struc.G(:,index) = G(index)*struc.G(:,index);
        struc.Gmin(:,index) = stiffness_bound(index)*struc.G(:,index);
        struc.B(:,index) = B(index)*struc.B(:,index);
        struc.Bmin(:,index) = stiffness_bound(index)*struc.B(:,index);
    end
end

end