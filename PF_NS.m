close all; clearvars; clc; set(groot,'defaulttextinterpreter','latex'); set(groot,'defaultAxesTickLabelInterpreter','latex'); set(groot,'defaultLegendInterpreter','latex');
addpath('Functions');

%% Cuboid Tracking using Particle Filter with Non Smooth model
% Settings
object = "Box007";              %Choose Box006, Box007, or Box009
viewpoint = "VP1";      %Choose VP1 or VP2
DoSave = false;                 %Choose true if you want to save, else false 
PlotProjections = true;        %Choose true if you want to plot the projections on the images, else false
Randomness = false;             %Choose true if you want to have randomness, else false

if ~Randomness 
    if object == "Box006"
        randn('seed',0);  rand('seed',0);
    elseif object == "Box007"
        randn('seed',0);  rand('seed',0)
    elseif object == "Box009"
        randn('seed',17);  rand('seed',17)
    end
end

%% Load data
%Load images and initial settings from dataset
[GT,Z,K,AH_M,Bw_MB,Bv_MB,sel_img,box,impacts] = Processing_script(object,viewpoint,false,false);
%Load reference image
Href = HIS(imread('RefImage.png'),12,12,12); %Reference histograms

%% Settings
Npart  = 50;                                           %Number of particles used
maxt   = length(sel_img);                               %Run to this frame
g     = 9.81;                                           %Gravitational acceleration [m/s^2]
te    = 0.0167;                                         %End time of the simulation (1/60 s)

%Constant parameters
if object == "Box006" && viewpoint == "VP1"
    const.eN    = 0.6; %0.25;                                     %Coefficient of restitution in normal direction [-]
    const.eT    = 0;                                        %Coefficient of restitution in tangential direction [-]
    const.mu    = 0.4;                                      %Coefficient of friction
    const.N     = 5;                                        %Number of discretization points [-]
    const.dt    = te/const.N;                               %Time between two time steps [s]
    const.a     = 0.001;                                     %Prox point parameter [-]
    const.tol   = 1e-5;                                     %Error tol for fixed-point [-]
    const.endtime = te;

    surface{1}.dim = [2.3 1];
    surface{1}.speed=[0.4; 0; 0];
    surface{1}.transform=[Ry(-0.5) [0.5; 0; 0]; zeros(1,3),1];
    abb = 'VP1';

elseif object == "Box006" && viewpoint == "VP2"
    const.eN    = 0.6;                                     %Coefficient of restitution in normal direction [-]
    const.eT    = 0.5;                                        %Coefficient of restitution in tangential direction [-]
    const.mu    = 0.6;                                      %Coefficient of friction
    const.N     = 5;                                        %Number of discretization points [-]
    const.dt    = te/const.N;                               %Time between two time steps [s]
    const.a     = 0.01;                                     %Prox point parameter [-]
    const.tol   = 1e-5;                                     %Error tol for fixed-point [-]
    const.endtime = te;
    abb = 'VP2';
elseif object == "Box007" && viewpoint == "VP1"
    const.eN    = 0.4; %0.58;                                      %Coefficient of restitution in normal direction [-]
    const.eT    = 0.4;  %0.5                                       %Coefficient of restitution in tangential direction [-]
    const.mu    = 0.4;  %0.6                                     %Coefficient of friction
    const.N     = 5;                                        %Number of discretization points [-]
    const.dt    = te/const.N;                               %Time between two time steps [s]
    const.a     = 0.001;                                     %Prox point parameter [-]
    const.tol   = 1e-5;                                     %Error tol for fixed-point [-]
    const.endtime = te;

    surface{1}.dim = [2.3 1];
    surface{1}.speed=[0.4; 0; 0];
    surface{1}.transform=[Ry(-0.5) [0.5; 0; 0]; zeros(1,3),1];
    abb = 'VP1';
elseif object == "Box007" && viewpoint == "VP2"
    const.eN    = 0.6; %0.25;                                     %Coefficient of restitution in normal direction [-]
    const.eT    = 0.5; %0;                                        %Coefficient of restitution in tangential direction [-]
    const.mu    = 0.6; %0.4;                                      %Coefficient of friction
    const.N     = 5;                                        %Number of discretization points [-]
    const.dt    = te/const.N;                               %Time between two time steps [s]
    const.a     = 0.01;                                     %Prox point parameter [-]
    const.tol   = 1e-5;                                     %Error tol for fixed-point [-]
    const.endtime = te;
    surface{1}.dim = [2.3 1];
    surface{1}.speed=[0.4; 0; 0];
    surface{1}.transform=[Ry(-0.5) [0.5; 0; 0]; zeros(1,3),1];
    abb = 'VP2';
elseif object == "Box009" && viewpoint == "VP1" 
    const.eN    = 0.5; %0.5;                                     %Coefficient of restitution in normal direction [-]
    const.eT    = 0.4; %0.4;                                        %Coefficient of restitution in tangential direction [-]
    const.mu    = 0.4; %0.4;                                      %Coefficient of friction
    const.N     = 5;                                        %Number of discretization points [-]
    const.dt    = te/const.N;                               %Time between two time steps [s]
    const.a     = 0.01;                                     %Prox point parameter [-]
    const.tol   = 1e-5;                                     %Error tol for fixed-point [-]     
    const.endtime = te;

    surface{1}.dim = [2.3 1];
    surface{1}.speed=[0.4; 0; 0];
    surface{1}.transform=[Ry(-0.5) [0.5; 0; 0]; zeros(1,3),1];
    abb = 'VP1';
elseif object == "Box009" && viewpoint == "VP2" 
    const.eN    = 0.6; %0.25;                                     %Coefficient of restitution in normal direction [-]
    const.eT    = 0.5; %0;                                        %Coefficient of restitution in tangential direction [-]
    const.mu    = 0.6; %0.4;                                      %Coefficient of friction
    const.N     = 5;                                        %Number of discretization points [-]
    const.dt    = te/const.N;                               %Time between two time steps [s]
    const.a     = 0.01;                                     %Prox point parameter [-]
    const.tol   = 1e-5;                                     %Error tol for fixed-point [-]     
    const.endtime = te;
    surface{1}.dim = [2.3 1];
    surface{1}.speed=[0.4; 0; 0];
    surface{1}.transform=[Ry(-0.5) [0.5; 0; 0]; zeros(1,3),1];
    abb = 'VP2';
end

%For box009
% PR = 1e-3*diag([1 1 1]);    Po = 1e-4*diag([1 1 1]);
% Pv = 1e-6*diag([1 1 1]);    Pw = 1e-3*diag([1 1 1]);

% %Initial state covariance
PR = 1e-3*diag([1 1 1]);    Po = 1e-4*diag([1 1 1]);
Pv = 1e-9*diag([1 1 1]);    Pw = 1e-5*diag([1 1 1]);

% For box007
% PR = 1e-5*diag([1 1 1]);    Po = 1e-4*diag([1 1 1]);
% Pv = 1e-6*diag([1 1 1]);    Pw = 1e-4*diag([1 1 1]);

%Initial guess for LambdaN and LambdaT
LambdaNfull = zeros(8,1);       LambdaTfull(1:8,1) = {zeros(2,1)}; 

%Force acting on body B: expressed in B with orientation of M:
BM_fo  = [0; 0; -box.mass*g];          BM_Tau = [0; 0; 0];         const.BM_f = [BM_fo; BM_Tau];

%Inertia tensor
const.B_M_B = box.B_M_B;

%Coordinate frame of the contact surface
const.MR_C  = eye(3);           const.Mo_C  = zeros(3,1);

%Vertices of the box
const.vertices  = box.vertices;

%Initial pose and velocity
MR_B = GT{1}(1:3,1:3);                                  %Initial rotation
Mo_B = GT{1}(1:3,4);                                    %Initial position [m]

%% 1: INITIALIZATION
time = tic();

%Sample initial set of particles
hsi = rgb2hsi(sel_img{1},12,12,12);         %Convert image to HSI bins (12x12x12)

for i = 1:Npart
    xiR = zeros(3,1) + sqrtm(PR)*randn(3,1);
    xio = zeros(3,1) + sqrtm(Po)*randn(3,1);
    xiv = zeros(3,1) + sqrtm(Pv)*randn(3,1);
    xiw = zeros(3,1) + sqrtm(Pw)*randn(3,1);
    
    X{1}{1,i}  = MR_B*expm(hat(xiR));
    X{1}{2,i}  = Mo_B+xio;
    X{1}{3,i}  = Bv_MB+xiv;
    X{1}{4,i}  = Bw_MB+xiw;
             
    [wK(1,i),~,~,~]=likelihood(X{1}(:,i),Href,hsi,K,box,AH_M); 
end 

%Compute the weighted average of the initial frame
wK=wK./sum(wK);       %Normalize the weights
[~,indx] = max(wK);   %Index of the particle with highest weight
wmean = X{1}(:,indx); %Take particle with highest weight as initial mean
% Map the other particles to a tangent space at wmean
for ii = 1:Npart
    XSampled(:,ii) = logx(xprod(invx(wmean),X{1}(:,ii)));
end
%Compute the mean in the tangent space, check if it is at the origin
Wmean = XSampled*wK';
if norm(Wmean) > 0.01 %If not at the origin, an update is executed
    wmean = xprod(wmean,expx(Wmean)); %New mean on Lie group
end
%Obtain the pose of the weighted mean as output
Y(:,1)= wmean(1:2,:); 


% 2: PARTICLE FILTER
for t=2:maxt
    hsi = rgb2hsi(sel_img{t},12,12,12);         %Convert image to HSI bins (12x12x12)
        
    % PROPAGATE THE PARTICLES (CREATE PRIOR/PROPOSAL) AND COMPUTE LIKELIHOOD
    for i=1:Npart %For each particle
%         X{t}(:,i) = MotionModel(X{t-1}(:,i),const);
        x.releasePosition    = X{t-1}{2,i};
        x.releaseOrientation = X{t-1}{1,i};
        x.releaseLinVel      = X{t-1}{3,i};
        x.releaseAngVel      = X{t-1}{4,i};
        [AH_B, BV_AB, FN, FT] = BoxSimulator(x,const,box,surface);
        X{t}{1,i} = AH_B{end}(1:3,1:3) * expm(hat(sqrtm(PR)*randn(3,1)));
        X{t}{2,i} = AH_B{end}(1:3,4) + sqrtm(Po)*randn(3,1);
        X{t}{3,i} = BV_AB(1:3,end) + sqrtm(Pv)*randn(3,1);
        X{t}{4,i} = BV_AB(4:6,end) + sqrtm(Pw)*randn(3,1);
        [wK(1,i),~,~,~]=likelihood(X{t}(:,i),Href,hsi,K,box,AH_M);
    end
    
    % COMPUTE WEIGHTED MEAN
    wK=wK./sum(wK);       %Normalize the weights
    [~,indx] = max(wK);   %Index of the particle with highest weight
    wmean = X{t}(:,indx); %Take particle with highest weight as initial mean
    
    % Map the other particles to a tangent space at wmean
    for ii = 1:Npart
        XSampled(:,ii) = logx(xprod(invx(wmean),X{t}(:,ii)));
    end
    
    %Compute the mean in the tangent space, check if it is at the origin
    Wmean = XSampled*wK'; 
    while norm(Wmean) > 1e-5 %If not at the origin, an update is executed
        wmean = xprod(wmean,expx(Wmean)); %New mean on Lie group
        for ii = 1:Npart
            % Map the other particles to a tangent space at wmean
            XSampled(:,ii) = logx(xprod(invx(wmean),X{t}(:,ii)));
        end
        Wmean = XSampled*wK';
    end
    %Obtain the pose of the weighted mean as output
    Y(:,t)= wmean(1:2,:); 
    
    % RESAMPLE
    [X{t},count] = Resample_systematic(X{t},wK);
    %Print progress in command window
    textwaitbar(t, maxt,"Progress   ")
end

%% 3: POST PROCESSING
%Print runtime to command window
runtime = toc(time);
runstring = sprintf('Runtime [s]: %d', runtime);
disp(runstring);

%Save the data
if DoSave
    PF_NS_Y = Y;
    save(sprintf(append('Data/Res_',object,abb,'/PF_NS_P%dF%d'),Npart,maxt),'PF_NS_Y','runtime','GT','Z','K','AH_M','box','impacts','sel_img');
end

%Compute the errors
for ii = 1:maxt
    XEY(ii)  = Y{2,ii}(1)-GT{ii}(1,4); %X-error of the output
    YEY(ii)  = Y{2,ii}(2)-GT{ii}(2,4); %Y-error of the output
    ZEY(ii)  = Y{2,ii}(3)-GT{ii}(3,4); %Z-error of the output
    
    EY = GT{ii}(1:3,4)-Y{2,ii};   
    
    NEY(ii) = norm(EY,1);  %Norm of the error of the output 
    NYR(ii) = rad2deg(norm(logm((GT{ii}(1:3,1:3))\Y{1,ii}))); %||log(R_GT^-1 * R_Y)||%
end 

%% 1: PLOT FIGURES
if PlotProjections
    figure;
    for ii = 1:length(Y(1,:))
        imshow(sel_img{ii}); 
        H_state_out = AH_M*[Y{1,ii} Y{2,ii}; 0 0 0 1];
        H_Z = AH_M*[Z{1,ii} Z{2,ii}; 0 0 0 1];
        projectBox(H_state_out,box,K,[0,1,0],1);
        hold on;
        projectBox(AH_M*GT{ii},box,K,[1,0,0],1);
%         projectBox(H_Z,box,K,[0,0,1],1);
        for kk = 1:5%Npart
            H_state_ptcls = AH_M*[X{ii}{1,kk} X{ii}{2,kk}; 0 0 0 1];
            projectBox(H_state_ptcls,box,K,[1,1,0],1);   
        end
pause()
%         pause(0.01);
    end
end

%% Figures to plot positions and errors
figure('pos',[200 400 250 200]); 
    g2 = plot(XEY,'linewidth',1,'color',[0.9290 0.6940 0.1250]);
    hold on; grid on;
    xlabel('Frame [-]');
    ylabel('x-error [m]')
    axis([1 maxt -0.15 0.15]);
    for ii = 1:length(impacts)
    xline(impacts(ii),':','linewidth',1.2,'color',[0 0 0 1],'alpha',1);
    end
    legend('PF\_NS','location','southwest');

figure('pos',[455 400 250 200]); 
    plot(YEY,'linewidth',1,'color',[0.9290 0.6940 0.1250]);
    hold on; grid on;
    xlabel('Frame [-]');
    ylabel('y-error [m]')
    axis([1 maxt -0.15 0.15]);
    for ii = 1:length(impacts)
    xline(impacts(ii),':','linewidth',1.2,'color',[0 0 0 1],'alpha',1);
    end
    legend('PF\_NS','location','southwest');

figure('pos',[710 400 250 200]); 
    plot(ZEY,'linewidth',1,'color',[0.9290 0.6940 0.1250]);
    hold on; grid on;
    xlabel('Frame [-]');
    ylabel('z-error [m]')
    axis([1 maxt -0.15 0.15]);
    for ii = 1:length(impacts)
    xline(impacts(ii),':','linewidth',1.2,'color',[0 0 0 1],'alpha',1);
    end
    legend('PF\_NS','location','southwest');

figure('pos',[965 400 250 200]); 
    plot(NEY,'linewidth',1,'color',[0.9290 0.6940 0.1250]);
    hold on; grid on;
    xlabel('Frame [-]');
    ylabel('$\|e_{\mathbf{o}}\|$ [m]')
    axis([1 maxt -0.1 0.2]);
    for ii = 1:length(impacts)
    xline(impacts(ii),':','linewidth',1.2,'color',[0 0 0 1],'alpha',1);
    end
    legend('PF\_NS','location','southwest');

figure('pos',[1220 400 250 200]); 
    plot(NYR,'linewidth',1,'color',[0.9290 0.6940 0.1250]);
    hold on; grid on;
    xlabel('Frame [-]');
    ylabel('$\|e_{\mathbf{R}}\|$ [deg]')
    axis([1 maxt -20 40]);
    for ii = 1:length(impacts)
    xline(impacts(ii),':','linewidth',1.2,'color',[0 0 0 1],'alpha',1);
    end
    legend('PF\_NS','location','southwest');