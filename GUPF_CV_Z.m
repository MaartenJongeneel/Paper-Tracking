close all; clearvars; clc; set(groot,'defaulttextinterpreter','latex'); set(groot,'defaultAxesTickLabelInterpreter','latex'); set(groot,'defaultLegendInterpreter','latex');
addpath('Functions');

%% Cuboid Tracking using Particle Filter with Constant Velocity model and measurement Z
% Settings
object = "Box006";              %Choose Box006, Box007, or Box009
viewpoint = "Viewpoint 1";      %Choose Viewpoint 1 or Viewpoint 2
DoSave = false;                 %Choose true if you want to save, else false 
PlotProjections = true;        %Choose true if you want to plot the projections on the images, else false
Randomness = false;             %Choose true if you want to have randomness, else false

if ~Randomness 
    randn('seed',0)
    rand('seed',0)
end

%% Load data
%Load images and initial settings from dataset
[GT,Z,K,AH_M,Bw_MB,Bv_MB,sel_img,box,impacts] = Processing_script(object,viewpoint,false,false);
%Load reference image
Href = HIS(imread('RefImage.png'),12,12,12); %Reference histograms

%% Settings
Npart  = 10;                                           %Number of particles used
maxt   = length(sel_img);                               %Run to this frame
alpha  = 0.9;                                           %UKF : point scaling parameter
beta   = 1;                                             %UKF : scaling parameter for higher order terms of Taylor series expansion
kappa  = 0.5;                                           %UKF : sigma point selection scaling parameter 
const.Noise = 0.035*diag([1.1 1.1 0.85 0.9 0.9 0.9]);

%Process noise covariance Qv and measurement noise covariance Rv
if object == "Box006" && viewpoint == "Viewpoint 1"
    Qv = 1e-5*diag([10 10 10 1 1 1 1 1 1 1 1 1]);           
    Rv = 1e-3*diag([10 10 10 1 1 1]);                       
 
elseif object == "Box006" && viewpoint == "Viewpoint 2" 
    Qv = 1e-5*diag([10 10 10 1 1 1 1 1 1 1 1 1]);           
    Rv = 1e-4*diag([10 10 10 1 1 1]);                       

elseif object == "Box007" && viewpoint == "Viewpoint 1"
    Qv = 1e-5*diag([10 10 10 1 1 1 1 1 1 1 1 1]);           
    Rv = 1e-3*diag([10 10 10 1 1 1]);                      

elseif object == "Box007" && viewpoint == "Viewpoint 2"
    Qv = 1e-5*diag([10 10 10 1 1 1 1 1 1 1 1 1]);          
    Rv = 1e-4*diag([10 10 10 1 1 1]);                       

elseif object == "Box009" && viewpoint == "Viewpoint 1" 
    Qv = 1e-5*diag([10 10 10 1 1 1 1 1 1 1 1 1]);           
    Rv = 1e-3*diag([10 10 10 1 1 1]);                       

elseif object == "Box009" && viewpoint == "Viewpoint 2" 
    Qv = 1e-5*diag([10 10 10 1 1 1 1 1 1 1 1 1]);          
    Rv = 1e-4*diag([10 10 10 1 1 1]);                       
end

%Initial state covariance
PR = 1e-5*diag([1 1 1]);    Po = 1e-5*diag([1 1 1]);
Pv = 1e-3*diag([1 1 1]);    Pw = 1e-3*diag([1 1 1]);

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
for t=2:maxt%length(FileList)
    hsi = rgb2hsi(sel_img{t},12,12,12);         %Convert image to HSI bins (12x12x12)

    %CREATE THE PROPOSAL DISTRIBUTION
    for i=1:Npart %For each particle
        %Run the UKF to create a proposal distribution and the transition prior
        [xEst(:,i),PEst{1,i},xPred,PxxPred]=ukf_Z(X{t-1}(:,i),P{t-1}{:,i},Qv,Rv,Z(:,t),alpha,beta,kappa,"CVModel",const); 
        
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
    
    %% COMPUTE WEIGHTED MEAN
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
    Y(:,t)= wmean(1:2,:); %Obtain the pose of the weighted mean as output
    
    %% RESAMPLE
    [X{t},count] = Resample_systematic(xSampled,wK);
    P{t} = PEst(1,count);
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
    GUPF_CV_Z_Y = Y;
    save(sprintf(append('Results/','/GUPF_CV_Z_P%dF%d'),Npart,maxt),'GUPF_CV_Z_Y','runtime','GT','Z','K','AH_M','box','impacts');
end

%Compute the errors
for ii = 1:maxt
    XEY(ii)  = Y{2,ii}(1)-GT{ii}(1,4); %X-error of the output
    YEY(ii)  = Y{2,ii}(2)-GT{ii}(2,4); %Y-error of the output
    ZEY(ii)  = Y{2,ii}(3)-GT{ii}(3,4); %Z-error of the output
    
    XEZ(ii) = Z{2,ii}(1)-GT{ii}(1,4);  %X-error of the measurement
    YEZ(ii) = Z{2,ii}(2)-GT{ii}(2,4);  %Y-error of the measurement
    ZEZ(ii) = Z{2,ii}(3)-GT{ii}(3,4);  %Z-error of the measurement
    
    EY = GT{ii}(1:3,4)-Y{2,ii};   
    EZ = GT{ii}(1:3,4)-Z{2,ii};
    
    NEY(ii) = norm(EY,1);  %Norm of the error of the output 
    NYR(ii) = rad2deg(norm(logm((GT{ii}(1:3,1:3))\Y{1,ii}))); %||log(R_GT^-1 * R_Y)||%
    NEZ(ii) = norm(EZ,1);  %Norm of the error of the measurement
    NZR(ii) = rad2deg(norm(logm((GT{ii}(1:3,1:3))\Z{1,ii}))); %||log(R_GT^-1 * R_Z)||%
end 

%% 4: PLOT FIGURES
if PlotProjections
    figure;
    for ii = 1:length(Y(1,:))
        hold on;
        imshow(sel_img{ii}); 
        for kk = 1:Npart
            H_state_ptcls = AH_M*[X{ii}{1,kk} X{ii}{2,kk}; 0 0 0 1];
            state_ptcls = {H_state_ptcls(1:3,1:3); H_state_ptcls(1:3,4)};
            H_state_out = AH_M*[Y{1,ii} Y{2,ii}; 0 0 0 1];
            state_out = {H_state_out(1:3,1:3); H_state_out(1:3,4)};
            H_state_meas = AH_M*[Z{1,ii} Z{2,ii}; 0 0 0 1];
            state_meas = {H_state_meas(1:3,1:3); H_state_meas(1:3,4)};

            %Determine which edges are visible and which faces are visible
            [~,~,~,ed_ptcls,~] = compute_points(box,state_ptcls,K,AH_M);
            [~,~,~,ed_out,~] = compute_points(box,state_out,K,AH_M);
            [~,~,~,ed_meas,~] = compute_points(box,state_meas,K,AH_M);

            %Obtain the corner points corresponding to the found edges
            cornerpoints_ptcls = box.edges(:,ed_ptcls);
            cornerpoints_out = box.edges(:,ed_out);
            cornerpoints_meas = box.edges(:,ed_meas);           

            %Obtain the first and second point of these edges
            first_points_ptcls = box.vertices(:,cornerpoints_ptcls(1,:));
            second_points_ptcls = box.vertices(:,cornerpoints_ptcls(2,:));
            first_points_out = box.vertices(:,cornerpoints_out(1,:));
            second_points_out = box.vertices(:,cornerpoints_out(2,:));
            first_points_meas = box.vertices(:,cornerpoints_meas(1,:));
            second_points_meas = box.vertices(:,cornerpoints_meas(2,:));

            %Transform these points to their true xyz position
            first_ptcls = hom_trans(first_points_ptcls,state_ptcls);
            second_ptcls = hom_trans(second_points_ptcls,state_ptcls);
            first_out = hom_trans(first_points_out,state_out);
            second_out = hom_trans(second_points_out,state_out);
            first_meas = hom_trans(first_points_meas,state_meas);
            second_meas = hom_trans(second_points_meas,state_meas);

            %Compute their representation in the image plane
            pts1_ptcls = persProj(first_ptcls,K,eye(4));
            pts2_ptcls = persProj(second_ptcls,K,eye(4));
            pts1_out = persProj(first_out,K,eye(4));
            pts2_out = persProj(second_out,K,eye(4));
            pts1_meas = persProj(first_meas,K,eye(4));
            pts2_meas = persProj(second_meas,K,eye(4));

            for jj = 1:length(pts1_ptcls)
                line([pts1_ptcls(1,jj);pts2_ptcls(1,jj)],[pts1_ptcls(2,jj);pts2_ptcls(2,jj)],'color','r','LineWidth',1.5);
            end

            for jj = 1:length(pts1_out)
                line([pts1_out(1,jj);pts2_out(1,jj)],[pts1_out(2,jj);pts2_out(2,jj)],'color','g','LineWidth',1.5);
            end

            for jj = 1:length(pts1_meas)
                line([pts1_meas(1,jj);pts2_meas(1,jj)],[pts1_meas(2,jj);pts2_meas(2,jj)],'color','b','LineWidth',1.5);
            end
        end
        pause();
    end
end

% Figures to plot positions and errors
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
    legend('GUPF\_CV\_Z','Z','location','southwest');

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
    legend('GUPF\_CV\_Z','Z','location','southwest');

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
    legend('GUPF\_CV\_Z','Z','location','southwest');

figure('pos',[965 400 250 200]); 
    plot(NEY,'linewidth',1,'color',[0.9290 0.6940 0.1250]);
    hold on; grid on;
    plot(NEZ,'-','linewidth',1,'color','r');
    xlabel('Frame [-]');
    ylabel('$\|e_{\mathbf{o}}\|$ [m]')
    axis([1 maxt -0.1 0.2]);
    for ii = 1:length(impacts)
    xline(impacts(ii),':','linewidth',1.2,'color',[0 0 0 1],'alpha',1);
    end
    legend('GUPF\_CV\_Z','Z','location','southwest');

figure('pos',[1220 400 250 200]); 
    plot(NYR,'linewidth',1,'color',[0.9290 0.6940 0.1250]);
    hold on; grid on;
    plot(NZR,'-','linewidth',1,'color','r');
    xlabel('Frame [-]');
    ylabel('$\|e_{\mathbf{R}}\|$ [deg]')
    axis([1 maxt -20 40]);
    for ii = 1:length(impacts)
    xline(impacts(ii),':','linewidth',1.2,'color',[0 0 0 1],'alpha',1);
    end
    legend('GUPF\_CV\_Z','Z','location','southwest');
