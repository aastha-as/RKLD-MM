function [objVal,x,nrse] = rkld_mm(y,A,xtrue,x1,totIter,thresh2)
%{
 Robust phase retrieval algorithm based on minimization of Reverse Kullback-Leibler Divergence (RKLD), 
 with Primal Dual Majorization Minimization Primal variable : x updated with update of dual variable z.
%}
%** Input parameters:**%
%{
 y (M x 1): M observations or measurements of magnitude or intensity of signal,
 A (M x k): system matrix with each row as a^H (1 x k),
 xtrue (k x 1): true k-dimensional signal x
 x1 (k x 1): initialized signal vector x
 thresh1: threshold to check convergence of x
 thresh2: threshold to check convergence of z
%}
y(y==0)=eps;

%% Algorithm RKLD-MM
                                       
xt=x1;                                                                      % xt used to store previous iteration x value
ax = abs(A*xt).^2;
z1 = log(ax./y) +1;                                                         % z1 initial estimate of auxiliary variable z
zk = z1;                                                                    % zk used to store previous iteration z value
iter =1; 
objVal(iter) = sum(ax.*(z1-1)-ax+y);        
nrse(iter) = norm(xtrue - exp(-1i*angle(trace(xtrue'*xt))) * xt, 'fro')/norm(xtrue,'fro');

    AA=(A'*A);
    while iter<totIter
        iter = iter +1;    
        bt = (AA)*xt;  
        ztA = zk.*A;
        xInn = (A'*ztA)\bt;
        eta2=1;                                
        while eta2> thresh2          
            Ck_diag=abs((ztA)*(xInn)).^2;
            z = 2*lambertw(0.5*sqrt((Ck_diag)./y)*sqrt(exp(1)));            % z update using previous iterate of x
            eta2 = norm(z-zk,'fro')/norm(zk,'fro');                     
            zk=z; 
            ztA = zk.*A;
            Rk = A'*ztA;
            xInn = Rk\bt;
            axInn = abs(A*xInn).^2;
            objInn = sum(axInn.*log(axInn./y)-axInn+y);
            if (objInn <= objVal(iter-1))                     
                break;                                                      
            end
        end             
        x = xInn;                                                           % x update
        objVal(iter) = objInn;
        nrse(iter) = norm(xtrue - exp(-1i*angle(trace(xtrue'*x))) * x, 'fro')/norm(xtrue,'fro');
        xt = x;                                     
    end
end