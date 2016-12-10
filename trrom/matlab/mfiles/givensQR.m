function [Q,R] = my_givens_QR(A) 
n = size(A,1); 
Q=eye(n); R=A; 
for j=1:n
    for i=n:(-1):j+1
        x=R(:,j);
        if norm([x(i-1),x(i)]) > 0
            c=x(i-1)/norm([x(i-1),x(i)]);
            s=-x(i)/norm([x(i-1),x(i)]);
            G=eye(n);
            G([i-1,i],[i-1,i])=[c,s;-s,c];
            R=G'*R;
            Q=Q*G;
        end
    end
end
8;