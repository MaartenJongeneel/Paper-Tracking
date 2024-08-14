close all; clearvars; clc; set(groot,'defaulttextinterpreter','latex'); set(groot,'defaultAxesTickLabelInterpreter','latex'); set(groot,'defaultLegendInterpreter','latex');
addpath('Functions');

%% Cuboid Tracking using Particle Filter with Non Smooth model and measurement Z
% Settings
object = "Box009";              %Choose Box006, Box007, or Box009
viewpoint = 'VP1';         %Choose VP1 or VP2
DoSave = false;                 %Choose true if you want to save, else false 
PlotProjections = true;        %Choose true if you want to plot the projections on the images, else false
Randomness = false;             %Choose true if you want to have randomness, else false

if ~Randomness 
    if object == "Box006"
        randn('seed',0);  rand('seed',0);
    elseif object == "Box007"
        randn('seed',1);  rand('seed',1)
    elseif object == "Box009"
        randn('seed',2);  rand('seed',2)
    end
end

%% Load data
%Load images and initial settings from dataset
[GT,Z,K,AH_M,Bw_MB,Bv_MB,sel_img,box,impacts,plotdata,bounding_boxes] = Processing_script(object,viewpoint,false,false);
Pixel_coordinates = bounding_boxes.Pixel_coordinates;
bb_centers = bounding_boxes.bb_centers';
%Load reference image
Href = HIS(imread('RefImage.png'),12,12,12); %Reference histograms

%Use this for Christiaan
%Load measurement
% load(append('Data/Res_',object,'VP',viewpoint(end),'/Z.mat'));
% Zold = Z;
% for ii = 1:length(Zold)
%     Z(:,:,ii) = AH_M\Zold(:,:,ii);
% end


%% Settings
Npart  = 500;                                           %Number of particles used
maxt   = length(sel_img);                          %Run to this frame
alpha  = 0.9;                                           %UKF : point scaling parameter
beta   = 1;                                             %UKF : scaling parameter for higher order terms of Taylor series expansion
kappa  = 0.5;                                           %UKF : sigma point selection scaling parameter 
g     = 9.81;                                           %Gravitational acceleration [m/s^2]
te    = 0.0167;                                         %End time of the simulation (1/60 s)


if object == "Box006" && viewpoint == "VP1"
    %Process noise covariance Qv and measurement noise covariance Rv
    %Use this for Christiaan
    %Qv = 1e-5*diag([10 10 10 1 1 1 1 1 1 1 1 1]);           
    %Rv = 1e-5*diag([10 10 10 1 1 1]);    
    Qv = 1e-5*diag([10 10 10 1 1 1 1 1 1 1 1 1]);
    Rv = 1e-3*diag([10 10 10 1 1 1]); 

    %Constant parameters
    const.eN    = 0.6; %0.25;                                     %Coefficient of restitution in normal direction [-]
    const.eT    = 0;                                        %Coefficient of restitution in tangential direction [-]
    const.mu    = 0.4;                                      %Coefficient of friction
    const.N     = 5;                                        %Number of discretization points [-]
    const.dt    = te/const.N;                               %Time between two time steps [s]
%     const.a     = 0.005;                                     %Prox point parameter [-]
    const.a     = 0.01;                                     %Prox point parameter [-]
    const.tol   = 1e-5;                                     %Error tol for fixed-point [-]
    const.mass  = box.mass;
    const.endtime = te;
    abb = 'VP1';

elseif object == "Box006" && viewpoint == "VP2"
    %Process noise covariance Qv and measurement noise covariance Rv
    Qv = 1e-5*diag([10 10 10 1 1 1 1 1 1 1 1 1]);           
    Rv = 1e-3*diag([10 10 10 1 1 1]);                       

    %Constant parameters
    const.eN    = 0.6;% 0.25;                                     %Coefficient of restitution in normal direction [-]
    const.eT    = 0;                                        %Coefficient of restitution in tangential direction [-]
    const.mu    = 0.4;                                      %Coefficient of friction
    const.N     = 5;                                        %Number of discretization points [-]
    const.dt    = te/const.N;                               %Time between two time steps [s]
    const.a     = 0.01;                                     %Prox point parameter [-]
    const.tol   = 1e-5;                                     %Error tol for fixed-point [-]
    const.mass  = box.mass;
    const.endtime = te;
    abb = 'VP2';
    
elseif object == "Box007" && viewpoint == "VP1" 
    %Use this for Christiaan
    Qv = 1e-5*diag([10 10 10 1 1 1 1 1 1 1 1 1]);           
    Rv = 1e-6*diag([1 1 1 1 1 1]);  
    %Process noise covariance Qv and measurement noise covariance Rv
%     Qv = 1e-5*diag([10 10 10 1 1 1 1 1 1 1 1 1]);           
%     Rv = 1e-3*diag([10 10 10 1 1 1]);                       

    %Constant parameters
    const.eN    = 0.6; %0.25;                                     %Coefficient of restitution in normal direction [-]
    const.eT    = 0.3;                                        %Coefficient of restitution in tangential direction [-]
    const.mu    = 0.4;                                      %Coefficient of friction
    const.N     = 5;                                        %Number of discretization points [-]
    const.dt    = te/const.N;                               %Time between two time steps [s]
    const.a     = 0.01;                                     %Prox point parameter [-]
    const.tol   = 1e-5;                                     %Error tol for fixed-point [-]
    const.mass  = box.mass;
    const.endtime = te;
%     Z(:,:,24) = Z(:,:,23);
    abb = 'VP1';

elseif object == "Box007" && viewpoint == "VP2" 
    %Process noise covariance Qv and measurement noise covariance Rv
    Qv = 1e-5*diag([10 10 10 1 1 1 1 1 1 1 1 1]);           
    Rv = 1e-5*diag([10 10 10 1 1 1]);                       

    %Constant parameters
    const.eN    = 0.25;                                     %Coefficient of restitution in normal direction [-]
    const.eT    = 0;                                        %Coefficient of restitution in tangential direction [-]
    const.mu    = 0.4;                                      %Coefficient of friction
    const.N     = 5;                                        %Number of discretization points [-]
    const.dt    = te/const.N;                               %Time between two time steps [s]
    const.a     = 0.01;                                     %Prox point parameter [-]
    const.tol   = 1e-5;                                     %Error tol for fixed-point [-]
    const.mass  = box.mass;

elseif object == "Box009" && viewpoint == "VP1"
    %Use this for Christiaan
%     Qv = 1e-5*diag([10 10 10 1 1 1 1 1 1 1 1 1]);           
%     Rv = 1e-5*diag([10 10 10 1 1 1]); 
    %Process noise covariance Qv and measurement noise covariance Rv
    Qv = 1e-5*diag([10 10 10 1 1 1 1 1 1 1 1 1]);           
    Rv = 1e-4*diag([10 10 10 1 1 1]);                     

    %Constant parameters
    const.eN    = 0.5; %0.25;                                     %Coefficient of restitution in normal direction [-]
    const.eT    = 0.4; %0;                                        %Coefficient of restitution in tangential direction [-]
    const.mu    = 0.4; %0.4;                                      %Coefficient of friction
    const.N     = 5;                                        %Number of discretization points [-]
    const.dt    = te/const.N;                               %Time between two time steps [s]
    const.a     = 0.01;                                     %Prox point parameter [-]
    const.tol   = 1e-5;                                     %Error tol for fixed-point [-]     
    const.mass  = box.mass;
    const.endtime = te;
    abb = 'VP1';

elseif object == "Box009" && viewpoint == "VP2"
    %Process noise covariance Qv and measurement noise covariance Rv
    Qv = 1e-5*diag([10 10 10 1 1 1 1 1 1 1 1 1]);           
    Rv = 1e-3*diag([10 10 10 1 1 1]);                       

    %Constant parameters
    const.eN    = 0.25;                                     %Coefficient of restitution in normal direction [-]
    const.eT    = 0;                                        %Coefficient of restitution in tangential direction [-]
    const.mu    = 0.4;                                      %Coefficient of friction
    const.N     = 5;                                        %Number of discretization points [-]
    const.dt    = te/const.N;                               %Time between two time steps [s]
    const.a     = 0.01;                                     %Prox point parameter [-]
    const.tol   = 1e-5;                                     %Error tol for fixed-point [-] 
    const.mass  = box.mass;
end

%Simulation time for each frame
% const.endtime = 6*const.dt;  %For 60fps
% const.endtime = 12*const.dt; %For 30fps

%Initial state covariance
%Use this for Christiaan
% PR = 1e-5*diag([1 1 1]);    Po = 1e-5*diag([1 1 1]);
% Pv = 1e-3*diag([1 1 1]);    Pw = 1e-3*diag([1 1 1]);

PR = 1e-5*diag([1 1 1]);    Po = 1e-5*diag([1 1 1]);
Pv = 1e-3*diag([1 1 1]);    Pw = 1e-3*diag([1 1 1]);

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
    
    P{1}{1,i}  = blkdiag(PR,Po,Pv,Pw);             
    [wK(i),~,~,~]=likelihood(X{1}(:,i),Href,hsi,K,box,AH_M); 
end 

%Compute the weighted average of the initial frame
wK=wK./sum(wK);             %Normalize the weights
[~,indx] = max(wK);         %Index of the particle with highest weight
wmean = X{1}(:,indx);       %Take particle with highest weight as initial mean
    
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

%% 2: UNSCENTED PARTICLE FILTER
% for t=2:maxt%length(FileList) %For 60fps
% for t=3:2:maxt%length(FileList) %For 30fps
ind = 1;
t_vec(1) = 1;
for t=2:maxt%length(FileList) 
    ind = ind+1;
    t_vec(ind)=t;
    hsi = rgb2hsi(sel_img{t},12,12,12);         %Convert image to HSI bins (12x12x12)

    %Obtain a measurement
%     AH_Best=AH_M*[Y{1,t-1},Y{2,t-1};zeros(1,3),1];
%     measurement = PEfun(bb_centers(t,:),AH_Best,sel_img{t},K,Pixel_coordinates(:,:,t),box);
%     measurement = AH_M\measurement;
%     Z(:,t)={measurement(1:3,1:3);measurement(1:3,4)};

    %CREATE THE PROPOSAL DISTRIBUTION
    for i=1:Npart %For each particle

        %Run the UKF to create a proposal distribution and the transition prior
%         measurement = {Z(1:3,1:3,t);Z(1:3,4,t)};
        measurement = Z(:,t);
        [xEst(:,i),PEst{1,i},xPred,PxxPred]=ukf_Z(X{ind-1}(:,i),P{ind-1}{:,i},Qv,Rv,measurement,alpha,beta,kappa,"BoxSimulator",const); 
        
        %Sample from the proposal distribution
        xSampled(:,i) = xprod(xEst(:,i),expx(sqrtm(PEst{1,i})*randn(12,1))); 
        
        %Evaluate the sampled particle at the proposal distribution:
        proposal = LieProb(xSampled(:,i),xEst(:,i),PEst{1,i});
        
        %Evaluate the sampled particle at the transition prior:
        prior = LieProb(xSampled(:,i),xPred,PxxPred);
        
        %Evaluate the sampled particle at the likelihood function:
        [lik,~,~,~]=likelihood(xSampled(:,i),Href,hsi,K,box,AH_M);
        
        %Compute the weight of the sampled particle:
        wK(1,i) = lik*prior/proposal;
    end
    
    % COMPUTE WEIGHTED MEAN
    wK=wK./sum(wK);     %Normalize the weights
    [~,indx] = max(wK); %Index of the particle with highest weight    
    wmean = xSampled(:,indx); %Take particle with highest weight as initial mean
    
    % Map the other particles to a tangent space at wmean
    for ii = 1:Npart
        XSampled(:,ii) = logx(xprod(invx(wmean),xSampled(:,ii)));
    end
    
    %Compute the mean in the tangent space, check if it is at the origin
    Wmean = XSampled*wK';     
    while norm(Wmean) > 1e-5 %If not at the origin, an update is executed
        wmean = xprod(wmean,expx(Wmean)); %New mean on Lie group
        for ii = 1:Npart
            % Map the other particles to a tangent space at wmean
            XSampled(:,ii) = logx(xprod(invx(wmean),xSampled(:,ii)));
        end
        Wmean = XSampled*wK';
    end
    Y(:,ind)= wmean(1:2,:); %Obtain the pose of the weighted mean as output
    
    % RESAMPLE
    [X{ind},count] = Resample_systematic(xSampled,wK);
    P{ind} = PEst(1,count);
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
    GUPF_NS_Z_Y = Y;
    save(sprintf(append('Data/Res_',object,abb,'/GUPF_NS_Z_P%dF%d'),Npart,maxt),'GUPF_NS_Z_Y','runtime','GT','Z','K','AH_M','box','impacts');
end

%Compute the errors
for ii = 1:length(Y)
    XEY(ii)  = Y{2,ii}(1)-GT{ii}(1,4); %X-error of the output
    YEY(ii)  = Y{2,ii}(2)-GT{ii}(2,4); %Y-error of the output
    ZEY(ii)  = Y{2,ii}(3)-GT{ii}(3,4); %Z-error of the output
    
%     XEZ(ii) = Z(1,4,ii)-GT{ii}(1,4);  %X-error of the measurement
%     YEZ(ii) = Z(2,4,ii)-GT{ii}(2,4);  %Y-error of the measurement
%     ZEZ(ii) = Z(3,4,ii)-GT{ii}(3,4);  %Z-error of the measurement
    XEZ(ii) = Z{2,ii}(1)-GT{ii}(1,4);  %X-error of the measurement
    YEZ(ii) = Z{2,ii}(2)-GT{ii}(2,4);  %Y-error of the measurement
    ZEZ(ii) = Z{2,ii}(3)-GT{ii}(3,4);  %Z-error of the measurement
    
    EY = (GT{ii}(1:3,4)-Y{2,ii});   
%     EZ = GT{ii}(1:3,4)-Z(1:3,4,ii);
    EZ = GT{ii}(1:3,4)-Z{2,ii};
%     EM = GT{ii}(1:3,4)-MH_Bsim{(ii-1)*6+1}(1:3,4);
    
    NEY(ii) = norm(EY);  %Norm of the error of the output 
    NYR(ii) = rad2deg(norm(logm((GT{ii}(1:3,1:3))\Y{1,ii}))); %||log(R_GT^-1 * R_Y)||%
    NEZ(ii) = norm(EZ);  %Norm of the error of the measurement
%     NZR(ii) = rad2deg(norm(logm((GT{ii}(1:3,1:3))\Z(1:3,1:3,ii)))); %||log(R_GT^-1 * R_Z)||%
    NZR(ii) = rad2deg(norm(logm((GT{ii}(1:3,1:3))\Z{1,ii}))); %||log(R_GT^-1 * R_Z)||%
%     NEM(ii) = norm(EM,1);  %Norm of the error of the model
%     NMR(ii) = rad2deg(norm(logm((GT{ii}(1:3,1:3))\MH_Bsim{(ii-1)*6+1}(1:3,1:3)))); %||log(R_GT^-1 * R_Z)||%
end 

%% 4: PLOT FIGURES
if PlotProjections
    figure;
    for ii = 1:length(t_vec)
        hold on;
        imshow(sel_img{t_vec(ii)}); 
        H_state_out = AH_M*[Y{1,ii} Y{2,ii}; 0 0 0 1];

        projectBox(H_state_out,box,K,[0,1,0],true);
        projectBox(AH_M*GT{t_vec(ii)},box,K,[1,0,0],true);
%         projectBox(AH_M*Z(:,:,ii),box,K,[0,0,1],true);
        projectBox(AH_M*[Z{1,ii} Z{2,ii}; 0 0 0 1],box,K,[0,0,1],true);
%         projectBox(AH_M*MH_Bsim{(ii-1)*6+1},box,K,[0,0,1],true)
% 
%         for kk = 1:50%Npart
%             H_state_ptcls = AH_M*[X{ii}{1,kk} X{ii}{2,kk}; 0 0 0 1];
%             projectBox(H_state_ptcls,box,K,[0,1,0],true);
%         end
        pause();
    end
end
%%
% video = VideoWriter('GUPF_NS_Z.avi'); %create the video object
% open(video); %open the file for writing
% 
% for i = 1:length(sel_img)
%     figure(1);
%     axis([0 720 0 540])
%     imshow(sel_img{i})
%     hold on
%     H_state_out = AH_M*[Y{1,i} Y{2,i}; 0 0 0 1];
%     projectBox(H_state_out,box,K,[0,0,1],true)
%     I = getframe(gcf);
%     writeVideo(video,I); %write the image to file
%     hold off;
% end
% close(video); %close the file
%% Figures to plot positions and errors
maxt = length(Y)
figure('pos',[200 400 250 200]); 
    plot(XEY,'linewidth',1,'color',[0.9290 0.6940 0.1250]);
    hold on; grid on;
    plot(XEZ,'-','linewidth',1,'color','r');
    xlabel('Frame [-]');
    ylabel('x-error [m]')
    axis([1 maxt -0.15 0.15]);
    for ii = 1:length(impacts)
    xline(impacts(ii),':','linewidth',1.2,'color',[0 0 0 1],'alpha',1);
    end
    legend('GUPF\_NS\_Z','Z','location','southwest');

figure('pos',[455 400 250 200]); 
    plot(YEY,'linewidth',1,'color',[0.9290 0.6940 0.1250]);
    hold on; grid on;
    plot(YEZ,'-','linewidth',1,'color','r');
    xlabel('Frame [-]');
    ylabel('y-error [m]')
    axis([1 maxt -0.15 0.15]);
    for ii = 1:length(impacts)
    xline(impacts(ii),':','linewidth',1.2,'color',[0 0 0 1],'alpha',1);
    end
    legend('GUPF\_NS\_Z','Z','location','southwest');

figure('pos',[710 400 250 200]); 
    plot(ZEY,'linewidth',1,'color',[0.9290 0.6940 0.1250]);
    hold on; grid on;
    plot(ZEZ,'-','linewidth',1,'color','r');
    xlabel('Frame [-]');
    ylabel('z-error [m]')
    axis([1 maxt -0.15 0.15]);
    for ii = 1:length(impacts)
    xline(impacts(ii),':','linewidth',1.2,'color',[0 0 0 1],'alpha',1);
    end
    legend('GUPF\_NS\_Z','Z','location','southwest');

figure('pos',[965 400 250 200]); 
    plot(NEY,'linewidth',1,'color',[0.9290 0.6940 0.1250]);
    hold on; grid on;
    plot(NEZ,'-','linewidth',1,'color','r');
%     plot(NEM,'-','linewidth',1,'color','k');
    xlabel('Frame [-]');
    ylabel('$\|e_{\mathbf{o}}\|$ [m]')
    axis([1 maxt -0.1 0.6]);
    for ii = 1:length(impacts)
    xline(impacts(ii),':','linewidth',1.2,'color',[0 0 0 1],'alpha',1);
    end
    exportgraphics(gcf,'poserror.png',Resolution=600)
%     legend('GUPF\_NS\_Z','Z','location','southwest');

figure('pos',[1220 400 250 200]); 
    plot(NYR,'linewidth',1,'color',[0.9290 0.6940 0.1250]);
    hold on; grid on;
    plot(NZR,'-','linewidth',1,'color','r');
%     plot(NMR,'-','linewidth',1,'color','k');
    xlabel('Frame [-]');
    ylabel('$\|e_{\mathbf{R}}\|$ [deg]')
    axis([1 maxt -20 100]);
    for ii = 1:length(impacts)
    xline(impacts(ii),':','linewidth',1.2,'color',[0 0 0 1],'alpha',1);
    end
    exportgraphics(gcf,'roterror.png',Resolution=600)
%     legend('GUPF\_NS\_Z','Z','location','southwest');

