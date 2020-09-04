
daily_stock_price = SNP(:,5); 
n = height(daily_stock_price);
R_t = zeros(n-1,1);
for num = 1:n-1
    R_t(num) = log(daily_stock_price{num+1,1}/daily_stock_price{num,1});
end
paramInitial = [0.0001,0.1,0.1,0.1];
signLev = 0.05;
hyp = [0,0,0,0];
[paramH, low_CI, up_CI, pVal] = MaxLikelihood(R_t, paramInitial, signLev, hyp, 1, 1);

p = testing(100,1,1,0.9248,0.1894,-0.2426,1000,-0.8734,[0.0001,0.1,0.1,0.1],0.05,[-0.8734,0.9248,0.1894,-0.2426])

% Post Estimation (RESIDUALS)
Mdl = egarch('Offset',NaN','GARCHLags',1,'ARCHLags',1,'LeverageLags',1);
EstMdl = estimate(Mdl,R_t);
v = infer(EstMdl,R_t);
res = (R_t-EstMdl.Offset)./sqrt(v);
plot(res)
title('Standardized Residuals')
autocorr(res)
parcorr(res)

function df = denFun(param, ySeries, pNum ,qNum)
    maximumOfpq = max(pNum,qNum);
    beta = param(pNum+2:end-1); 
    alpha = param(2:pNum+1);  
    gamma = param(length(param));
    sigm = zeros(length(ySeries),1);
    sigm(1:maximumOfpq) = std(ySeries);
    for j=(maximumOfpq+1):length(ySeries)
        sigm(j) = sqrt(exp(param(1)...
            + sum(log(sigm(j-pNum:j-1).*sigm(j-pNum:j-1)).*alpha)...
            + sum(abs(ySeries(j-qNum:j-1)./sigm(j-qNum:j-1)).*beta)...
            + gamma * ySeries(j-1)/sigm(j-1)));
    end
    df = 1/sqrt(2*pi)./sigm.*exp(-ySeries.^2./(2*(sigm.^2)));
end


function like = likelihood(funct, param, ySeries, pNum, qNum) 
    like = -sum(log(funct(param, ySeries, pNum ,qNum)))/(length(ySeries)); 
end

function [paramH, low_CI, up_CI, pVal] = MaxLikelihood(ySeries, paramInitial, signLev, hyp, pNum, qNum)
    [paramH,~,~,~,~,hessian] = fminunc(@(param) likelihood(@denFun, param, ySeries, pNum, qNum),paramInitial);
    A = length(paramH);
    N = length(ySeries);
    variance_mat = inv(hessian);
    sigm_estim = zeros(1,A);
    for j=1:A
        sigm_estim(j) = sqrt(variance_mat(j,j))/sqrt(N);
    end
    z_stat = (paramH-hyp)./sigm_estim; 
    pVal = 2*normcdf(-abs(z_stat),0,1);
    L = abs(icdf('Normal',signLev/2,0,1)); 
    low_CI = hyp - L * sigm_estim; 
    up_CI = hyp + L * sigm_estim;
end







function [ySeries,sigm] = simuGARCH(pNum,alpha,qNum,beta,gamma,size,const) 
    maximumOfpq = max(pNum,qNum);
    ySeries = zeros(size+1000,1);
    sigm = zeros(size+1000,1); 
    for j = 1:maximumOfpq
        ySeries(j) = normrnd(0,1);
        sigm(j) = normrnd(0,1);
    end
    for j = (maximumOfpq+1):(size+1000)
        sigm(j) = sqrt(exp(const...
            + sum(log(sigm(j-pNum:j-1).*sigm(j-pNum:j-1)).*alpha)...
            + sum(abs(ySeries(j-qNum:j-1)./sigm(j-qNum:j-1)).*beta)...
            + gamma * ySeries(j-1)/ sigm(j-1))); 
        ySeries(j) = sigm(j)*normrnd(0,1);
    end
    ySeries = ySeries(1001:end);
    sigm = sigm(1001:end); 
end

function [p] = testing(n,p,q,alpha,beta,gamma,size,const,paramInitial,signLev,hyp) 
    pVal_store = zeros(p+q+2,n);
    for j = 1:n
        [y,~] = simuGARCH(p,alpha,q,beta,gamma,size,const);
        [~, ~, ~, pVal] = MaxLikelihood(y, paramInitial, signLev, hyp, p, q);
        pVal_store(:,j) = 1-pVal;
    end
    decision=pVal_store<signLev; 
    p = sum(decision,2)/n;
end
