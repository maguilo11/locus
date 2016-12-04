function [ParameterStruct] = generateFilterOperator(ParameterStruct)
% Input Radius
FilterRadius = ParameterStruct.FilterRadius;
% Centroid
elem_length = ParameterStruct.DomainSpecs.xmax / ...
    ParameterStruct.DomainSpecs.nx;
elem_heigth = ParameterStruct.DomainSpecs.ymax / ...
    ParameterStruct.DomainSpecs.ny;
x_centroid = elem_length / 3;
y_centroid = elem_heigth / 3;
% Compute element centroids
NumElems = ParameterStruct.DomainSpecs.nx * ParameterStruct.DomainSpecs.ny;
deltaRadius = zeros(NumElems, NumElems);
for i=1:ParameterStruct.DomainSpecs.nx
    for j=1:ParameterStruct.DomainSpecs.ny
        ith_elem_centroid_x_coord = elem_length*(i-1) + x_centroid;
        ith_elem_centroid_y_coord = elem_heigth*(j-1) + y_centroid;
        ith_elem = ParameterStruct.DomainSpecs.ny * (i-1) + j;

        for k=1:ParameterStruct.DomainSpecs.nx
            for l=1:ParameterStruct.DomainSpecs.ny
                jth_elem_centroid_x_coord = elem_length*(k-1) + x_centroid;
                jth_elem_centroid_y_coord = elem_heigth*(l-1) + y_centroid;
                jth_elem = ParameterStruct.DomainSpecs.ny * (k-1) + l;
                
                distance = ...
                    sqrt((ith_elem_centroid_x_coord - jth_elem_centroid_x_coord)^2 ...
                    + (ith_elem_centroid_y_coord - jth_elem_centroid_y_coord)^2);
                if(distance <= FilterRadius)
                    deltaRadius(ith_elem,jth_elem) = FilterRadius - distance;
                else
                    deltaRadius(ith_elem,jth_elem) = 0.;
                end
            end
        end
    end
end

Weights = zeros(NumElems, NumElems);
for i=1:ParameterStruct.DomainSpecs.nx
    for j=1:ParameterStruct.DomainSpecs.ny
        ith_elem_centroid_x_coord = elem_length*(i-1) + x_centroid;
        ith_elem_centroid_y_coord = elem_heigth*(j-1) + y_centroid;
        ith_elem = ParameterStruct.DomainSpecs.ny * (i-1) + j;

        for k=1:ParameterStruct.DomainSpecs.nx
            for l=1:ParameterStruct.DomainSpecs.ny
                jth_elem_centroid_x_coord = elem_length*(k-1) + x_centroid;
                jth_elem_centroid_y_coord = elem_heigth*(l-1) + y_centroid;
                jth_elem = ParameterStruct.DomainSpecs.ny * (k-1) + l;
                
                distance = ...
                    sqrt((ith_elem_centroid_x_coord - jth_elem_centroid_x_coord)^2 ...
                    + (ith_elem_centroid_y_coord - jth_elem_centroid_y_coord)^2);
                if(distance <= FilterRadius)
                    Weights(ith_elem,jth_elem) = (FilterRadius - distance) ...
                        / sum(deltaRadius(ith_elem,:));
                else
                    Weights(ith_elem,jth_elem) = 0;
                end
            end
        end
    end
end

ParameterStruct.Weights = Weights;

end