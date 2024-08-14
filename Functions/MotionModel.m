function [X1] = MotionModel(X0,const)
% Computes the state of the cuboid for the next time step
% INPUTS:    X0        : State at t0 cell(R;o;v;w)
%
% OUTPUTS:   X1        : State at t1 cell(R;o;v;w)
%% Compute the state of the next time step
%% Initial pose and velocity
MR_B    = X0{1};
Mo_B    = X0{2};
MH_B = [MR_B, Mo_B; zeros(1,3), 1];

BV_MB = [X0{3}; X0{4}];

LambdaNfull = zeros(8,1);   %Initial guess for LambdaN        
LambdaTfull(1:8,1) = {zeros(2,1)}; %initial guess for LambdaT

B_M_B = const.B_M_B;
%% Dynamics
for t = 1:const.N %For each time step
    %Kinematics: Compute the configuration at the mid-time
    MH_Bm = MH_B*expm(0.5*const.dt*hat(BV_MB));
    MR_Bm = MH_Bm(1:3,1:3);
    Mo_Bm = MH_Bm(1:3,4);
    MR_B  = MH_B(1:3,1:3);
    
    %Compute the wrench at the mid-time
    B_fM = ([MR_Bm zeros(3); zeros(3) MR_Bm])'*const.BM_f;
    
    %And compute the gap-functions at tM column of normal contact distances
    gN = (const.MR_C(:,3)'*(Mo_Bm + MR_Bm*const.vertices-const.Mo_C))';
   
    %Obtain the linear and angular velocity at tA
    vA = BV_MB;
    
    IN = find(gN<0);
    if  IN > 0
        %Compute the matrix containing force directions
        [WNA, WTA] = CompW(MR_B,const.MR_C,const.vertices(:,IN));
        [WNM, WTM] = CompW(MR_Bm,const.MR_C,const.vertices(:,IN));

        converged = 0;
        LambdaN=LambdaNfull(IN);
        LambdaT=cell2mat(LambdaTfull(IN));
        term1 = B_fM*const.dt - [hat(vA(4:6)), zeros(3); hat(vA(1:3)), hat(vA(4:6))]*B_M_B*vA*const.dt;
        while converged==0
            %Decompose the system to write the linear and angular velocity
            %in different equations
            vE = vA + B_M_B\(term1 + WNM*LambdaN + WTM*LambdaT);
            
            %Define the normal velocities at the beginning and end of the
            %time step
            gammaNA = WNA'*vA;
            gammaNE = WNM'*vE;
            
            gammaTA = WTA'*vA;
            gammaTE = WTM'*vE;
            
            %Newtons restitution law
            xiN = gammaNE+const.eN*gammaNA;
            xiT = gammaTE+const.eT*gammaTA;
            
            %Find LambdaN using the proximal point function
            LambdaNold = LambdaN;
            LambdaTold = LambdaT;
            LambdaN = proxCN(LambdaN-const.a*xiN);
            LambdaT = proxCT(LambdaT-const.a*xiT,const.mu*LambdaN);
            
            error= norm(LambdaN-LambdaNold)+norm(LambdaT-LambdaTold);
            converged = error<const.tol;
        end
        BV_MB = vE;
    else
        %Update the velocity to the next time step
        vE = B_M_B\(B_fM*const.dt - [hat(vA(4:6)), zeros(3); hat(vA(1:3)), hat(vA(4:6))]*B_M_B*vA*const.dt) + vA;
        BV_MB = vE;
        LambdaN = 0;
        LambdaT = [0;0];
    end
    %Update Lambda for next estimate
    if IN ~= 0
        LambdaNfull(IN)=LambdaN;
        cnt=1;
        for ii = length(IN)
            LambdaTfull(IN(ii)) = {LambdaT(cnt:cnt+1)};
            cnt=cnt+2;
        end
    end
    
    %Complete the time step
    MH_B  = MH_Bm*expm(0.5*const.dt*hat(BV_MB));
end
    
    X1{1,1} = MH_B(1:3,1:3);
    X1{2,1} = MH_B(1:3,4);
    X1{3,1} = BV_MB(1:3);
    X1{4,1} = BV_MB(4:6);
end

%% Matrix with force directions
function [WN,WT] = CompW(MR_B,MR_C,vertices)
% Compute the matrix containing the tangential force directions.
tel = 1;
for ii = 1:length(vertices(1,:))
    w = (MR_C'*[MR_B -MR_B*hat(vertices(:,ii))])';
    WN(:,ii) = w(:,3);
    WT(:,tel:tel+1) = w(:,1:2);
    tel = tel+2;
end
end

%% Proximal point Normal
function y=proxCN(x)
% Proximal point formulation for CN. See thesis for reference.
% prox_CN(x) = 0 for x=< 0
%            = x for x > 0
y=max(x,0);
end

%% Proximal point Tangential
function y=proxCT(x,a)
% Proximal point formulation for CT. See thesis for reference.
%
% prox_CT(x) = x           for ||x|| =< a
%            = a*(x/||x||) for ||x||  > a
% for CT = {x in R^n| ||x|| =< a}
cnt = 1;
for ii = 1:length(a) %For each point in contact
    if norm(x(cnt:cnt+1)) <= a(ii)
        y(cnt:cnt+1,1) = x(cnt:cnt+1); %Stick
    else
        y(cnt:cnt+1,1) = a(ii)*x(cnt:cnt+1)/norm(x(cnt:cnt+1)); %Slip
    end
    cnt = cnt+2;
end
end