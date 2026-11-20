function [solution,feasible] = projectControl(request,A,b,lower,upper)
%PROJECTCONTROL Exact 3-variable Euclidean QP using active-set enumeration.
% min ||u-request||^2 / 2, A*u <= b, lower <= u <= upper.
% A projection in R^3 has a KKT representation with at most three independent
% active normals. Enumerating those sets needs only base MATLAB linear algebra.
request=reshape(request,3,1); lower=reshape(lower,3,1); upper=reshape(upper,3,1);
solution=min(max(request,lower),upper); feasible=false; tolerance=1e-7;
if all(A*solution<=b+tolerance) && all(lower<=upper)
    feasible=true; return; % projection onto the actuator box is already safe
end
if any(lower>upper), return; end
A=[A;eye(3);-eye(3)]; b=[b;upper;-lower];
norms=vecnorm(A,2,2);
if any(norms<1e-12 & b<-tolerance), return; end
keep=norms>=1e-12; A=A(keep,:)./norms(keep); b=b(keep)./norms(keep);
[~,order]=sort(A*request-b,'descend'); A=A(order,:); b=b(order);
n=size(A,1);
for i=1:n
    if candidate(i), return; end
end
for i=1:n-1
    for j=i+1:n
        if candidate([i j]), return; end
    end
end
for i=1:n-2
    for j=i+1:n-1
        for k=j+1:n
            if candidate([i j k]), return; end
        end
    end
end
    function yes=candidate(indices)
        yes=false; active=A(indices,:); gram=active*active';
        if rcond(gram)<1e-11, return; end
        multiplier=gram\(active*request-b(indices));
        if any(multiplier<-tolerance), return; end
        value=request-active'*multiplier;
        if any(A*value>b+tolerance), return; end
        solution=value; feasible=true; yes=true;
    end
end
