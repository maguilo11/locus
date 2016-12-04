function [a] = il_zeros(n1, n2, n3, n4, n5, n6)

  % sorted by typical frequency of use in Intrepid

  if nargin == 3
    a.data = zeros(n3,n2,n1);
    a.dims = [n1 n2 n3];
    return
  end

  if nargin == 4
    a.data = zeros(n4,n3,n2,n1);
    a.dims = [n1 n2 n3 n4];
    return
  end

  if nargin == 2
    a.data = zeros(n2,n1);
    a.dims = [n1 n2];
    return
  end

  if nargin == 1
    a.data = zeros(n1);
    a.dims = [n1];
    return
  end

  if nargin == 5
    a.data = zeros(n5,n4,n3,n2,n1);
    a.dims = [n1 n2 n3 n4 n5];
    return
  end

  if nargin == 6
    a.data = zeros(n6,n5,n4,n3,n2,n1);
    a.dims = [n1 n2 n3 n4 n5 n6];
    return
  end
  
end
