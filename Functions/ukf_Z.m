function [xEst,PEst,xPred,PxxPred]=ukf_Z(xEst,PEst,Q,R,Z,alpha,beta,kappa,fname,const)
% Compute one full step of the unscented Kalman filter
%
% INPUTS:    xEst    : state mean estimate at time k: 4x1 cell
%            PEst    : state covariance at time: k 12x12 matrix
%            Q       : process noise covariance at time k
%            R       : measurement noise covariance at k+1
%            alpha   : sigma point scaling parameter.
%            beta    : higher order error scaling parameter.
%            kappa   : scalar tuning parameter 1.
%
% OUTPUTS:   xEst    : updated estimate of state mean at time k+1
%            PEst    : updated state covariance at time k+1
%            xPred   : prediction of state mean at time k+1
%            PxxPred : prediction of state covariance at time k+1
%% Part 1: State prediction through motion model
fname = str2func(fname);

% Calculate the dimensions of the problem 
sDim = 12; %State dimension
vDim = 12; %Process noise dimension
wDim = 6;  %Measurement noise dimension

%Augment the state mean and covariance with the noise parameters.
PA=[                PEst, zeros(sDim,vDim), zeros(sDim,wDim);
    zeros(vDim,sDim),                    Q, zeros(vDim,wDim);
    zeros(wDim,sDim), zeros(wDim,vDim),                    R];
 
xA=[zeros(sDim,1);zeros(vDim,1);zeros(wDim,1)];

%Calculate the sigma points and their weights using the Scaled Unscented
%Transformation (on tangent space of augmented state)
[sigX,sigM,sigN,w,nsp]=CompSigmaPnts(xA,PA,alpha,beta,kappa,sDim,vDim,wDim); 

surface{1}.dim = [2.3 1];
surface{1}.speed=[0.4; 0; 0];
surface{1}.transform=[Ry(-0.5) [0.5; 0; 0]; zeros(1,3),1];

box.B_M_B=const.B_M_B;
box.mass = const.mass;
box.vertices = const.vertices;

for ii = 1:nsp % For each sigma point
    %Map the sigma point of the state to the group
    sigx = xprod(xEst,expx(sigX(:,ii)));

    x.releasePosition   = sigx{2};
    x.releaseOrientation= sigx{1};
    x.releaseLinVel     = sigx{3};
    x.releaseAngVel     = sigx{4};

    [MH_B, BV_MB] = feval(fname,x,const,box,surface);

    res{1}=MH_B{6}(1:3,1:3);
    res{2}=MH_B{6}(1:3,4);
    res{3}=BV_MB(1:3,6);
    res{4}=BV_MB(4:6,6);


    % res = feval(fname,sigx,const);
    %Compute the predicted sigma points on the group: f(x)*exp(n)
    xPredSig(:,ii) = xprod(res,expx(sigM(:,ii)));
    xmean = xPredSig(:,1); %Take zeroth sigma point as initial mean on group
    XPredSig(:,ii) = logx(xprod(invx(xmean),xPredSig(:,ii)));
end 

% Compute the mean in the tangent space
Xmean = XPredSig*w(1:nsp)';

% Check update and compute new mean
while norm(Xmean) > 1e-4 %If mean is not at the origin
    xmean = xprod(xmean,expx(Xmean)); %New mean on Lie group
    for ii = 1:nsp
        XPredSig(:,ii) = logx(xprod(invx(xmean),xPredSig(:,ii)));
    end
    Xmean = XPredSig*w(1:nsp)';
end

wnew=(flipud(w'))'; %To ensure zeroth weight of covariance is at index 1

%Final mean and covariance of the predicted state
PxxPred = (wnew(1:nsp).*XPredSig)*XPredSig';
xPred = xmean;     

%% Part 2: State update through measurement
for jj = 1:length(xPredSig(1,:)) %For each sigma point
    hpred(:,jj) = Hprod(xPredSig(1:2,jj),expH(sigN(:,jj))); %Hpred_j = Hpred_j*exp(Hsigma): The predicted output is perturbed with gaussian noise
    hmean = hpred(:,1);
    HPredSigma(:,jj) = logH(Hprod(invH(hmean),hpred(:,jj)));
end

% Compute the mean
Hmean = HPredSigma*w(1:nsp)';

% Check update and compute new mean
while norm(Hmean) > 1e-4 %Might be too computational heavy
    hmean = Hprod(hmean,expH(Hmean)); %New mean on Lie group
    for jj = 1:nsp
        HPredSigma(:,jj) = logH(Hprod(invH(hmean),hpred(:,jj)));
    end
    Hmean = HPredSigma*w(1:nsp)';
end

%Final mean and covariance of the predicted state
PyyPred = (wnew(1:nsp).*HPredSigma)*HPredSigma';
PxyPred = (wnew(1:nsp).*XPredSig)*HPredSigma';

inn = logH(Hprod(invH(hmean),Z)); %innovation = log(hmean^-1*Z)
xEst = xprod(xPred,expx(PxyPred/(PyyPred)*inn)); %Estimated state
PEst = PxxPred - PxyPred*(PxyPred/(PyyPred))'; %Estimated covariance
