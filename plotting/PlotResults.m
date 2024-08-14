close all; clearvars; clc; 
set(groot,'defaulttextinterpreter','latex'); set(groot,'defaultAxesTickLabelInterpreter','latex'); set(groot,'defaultLegendInterpreter','latex');
%% Plot the results to figures
% This script is used to load the tracking result data and plot some nice
% figures to illustrate the results. It also computes the errors with
% respect to the ground truth data.

%% General stuff
addpath(genpath('Functions'));

%% Settings
object = "Box006";              %Choose Box006, Box007, or Box009
ViewPnt = "VP1";      %Choose VP1 or VP2
doSave = false;
meas = "synth";        %Choose "chris" for Christiaans method or "synth" for synthetic

% doSave       = 0;                %Decide if you want to save the plots [-]
createvideo  = 0;                %Decide if you want to make a video   [-]
ws           = 1;                %Width of the contact surface         [m]
ls           = 1;                %Length of the contact surface        [m]
R      = 5;  %radius of the manifold
plottext = 0;

if object == "Box006" && ViewPnt=="VP1"
    impacts = [11 13 15 19 27 35];
    Res_NS    = load('data/Res_Box006VP1/NS.mat');             %Nonsmooth motion model only
    Res_PF_CV = load('data/Res_Box006VP1/PF_CV_P500F60.mat');  %Particle Filter, CV model
    Res_PF_NS = load('data/Res_Box006VP1/PF_NS_P500F60.mat');  %Particle Filter, NS model
    if meas == "chris"
        Res_GUPF_NS = load('data/Res_Box006VP1/GUPF_NS_Z_P50F60.mat');  %Particle Filter, NS model christiaan
        Res_Z     = load('data/Res_Box006VP1/Z.mat');              %Measurement
    elseif meas == "synth"
        Res_GUPF_NS = load('data/Res_Box006VP1/GUPF_NS_Z_P500F60_synthetic_rmspos0p027rmsrot5p57.mat');  %Particle Filter, NS model
        Res_Z     = load('data/Res_Box006VP1/Z_synthetic.mat');              %Measurement
    end
elseif object == "Box006" && ViewPnt=="VP2"
    impacts = [11 13 15 21 28 37 45];  
elseif object == "Box007" && ViewPnt=="VP1"
    impacts = [12 13 14 24 32];
    Res_NS    = load('data/Res_Box007VP1/NS.mat');             %Nonsmooth motion model only
    Res_PF_CV = load('data/Res_Box007VP1/PF_CV_P500F60.mat');  %Particle Filter, CV model
    Res_PF_NS = load('data/Res_Box007VP1/PF_NS_P50F60_rmspos0p0634rmsrot20p16.mat');  %Particle Filter, NS model
    
%     Res_PF_NS.PF_NS_Y=Res_PF_NS.Y; %did something stupid

    Res_Z     = load('data/Res_Box007VP1/Z_synthetic.mat');              %Measurement
    if meas == "chris"
        Res_GUPF_NS = load('data/Res_Box007VP1/GUPF_NS_Z_P50F60.mat');  %Particle Filter, NS model christiaan
        Res_Z     = load('data/Res_Box007VP1/Z.mat');              %Measurement
    elseif meas == "synth"
        Res_GUPF_NS = load('data/Res_Box007VP1/GUPF_NS_Z_P100F60_synthetic_rmspos0p039rmsrot5p99.mat');  %Particle Filter, NS model
        Res_Z     = load('data/Res_Box007VP1/Z_synthetic.mat');              %Measurement
    end
elseif object == "Box007" && ViewPnt=="VP2"
    impacts = [12 13 14 23 29];  
elseif object == "Box009" && ViewPnt=="VP1"
    impacts = [12 14 23 27 33]; 
    Res_NS    = load('data/Res_Box009VP1/NS.mat');             %Nonsmooth motion model only
    Res_PF_CV = load('data/Res_Box009VP1/PF_CV_P500F63.mat');  %Particle Filter, CV model
    Res_PF_NS = load('data/Res_Box009VP1/PF_NS_P500F63.mat');  %Particle Filter, NS model
    if meas == "chris"
        Res_GUPF_NS = load('data/Res_Box009VP1/GUPF_NS_Z_P50F63.mat');  %Particle Filter, NS model christiaan
        Res_Z       = load('data/Res_Box009VP1/Z.mat');              %Measurement
    elseif meas == "synth"
        Res_GUPF_NS = load('data/Res_Box009VP1/GUPF_NS_Z_P500F63_synthetic_rmspos0p013rmsrot6p10.mat');  %Particle Filter, NS model
        Res_Z       = load('data/Res_Box009VP1/Z_synthetic.mat');              %Measurement
    end
elseif object == "Box009" && ViewPnt=="VP2"
    impacts = [11 12 22 28 37]; 
end

wrtimg = false;
doPlot = false;
[GT,~,K,AH_M,~,~,sel_img,boxmodel,~,plotdata,bounding_boxes] = Processing_script(object,ViewPnt,wrtimg,doPlot);


tue = struct('r',[0.784 0.098 0.098],...
            'cb',[0.063 0.063 0.451],...
             'b',[0 0.4 0.8],...
             'c',[0 0.635 0.871],...
             'g',[0.518 0.824 0],...
             'y',[0.808 0.875 0],...
             'delft',[0 0.651 0.8392],...
             'dg',[0 151/255 57/255],...
             'tb',[4/255 51/255 83/255],...
             'herfstblad',[122/255,70/255,57/255],...
             'origami',[220/255,217/255,208/255],...
             'middadel',[201/255,174/255,145/255],...
             'webtext',[54/255,54/255,54/255]);
% tue.colorNS = [94 80 63]/255;
% tue.colorPFCV = [201 174 145]/255;
% tue.colorPFNS=[10 9 8]/255;
% tue.colorGUPFNS=[48 68 75]/255;
% tue.colorZ=[198 189 180]/255;

% tue.colorNS = [33 67 77]/255;
tue.colorNS = [82 109 130]/255;
tue.colorPFCV = [84 11 14]/255;
tue.colorPFNS=[201 174 145]/255;
tue.colorGUPFNS=[158 42 43]/255;
tue.colorZ=[224 159 62]/255;


%% Plot results from simulation
close all;

%Compute simulation error
for jj = 1:length(GT)
    tel = (jj-1)*6+1;

    o_GT = GT{jj}(1:3,4);
    o_SIM(:,jj) = Res_NS.MH_Bsim{tel}(1:3,4);

    R_GT = GT{jj}(1:3,1:3);
    R_SIM(:,:,jj) = Res_NS.MH_Bsim{tel}(1:3,1:3);

    e_o(jj) = norm(o_SIM(:,jj) - o_GT);
    e_R(jj) = rad2deg(norm(vee(logm(R_GT\R_SIM(:,:,jj)))));

    if meas == "chris"
        Z(:,:,jj) = AH_M\Res_Z.Z(:,:,jj);
    elseif meas == "synth"
        Z(:,:,jj) = Res_Z.Z(:,:,jj);
    end
end

%Compute the errors
for ii = 1:length(GT)
    XEY_CV(ii)  = Res_PF_CV.PF_CV_Y{2,ii}(1)-GT{ii}(1,4); %X-error of the output
    YEY_CV(ii)  = Res_PF_CV.PF_CV_Y{2,ii}(2)-GT{ii}(2,4); %Y-error of the output
    ZEY_CV(ii)  = Res_PF_CV.PF_CV_Y{2,ii}(3)-GT{ii}(3,4); %Z-error of the output
%     XEY_NS(ii)  = Res_PF_NS.PF_NS_Y{2,ii}(1)-GT{ii}(1,4); %X-error of the output
%     YEY_NS(ii)  = Res_PF_NS.PF_NS_Y{2,ii}(2)-GT{ii}(2,4); %Y-error of the output
%     ZEY_NS(ii)  = Res_PF_NS.PF_NS_Y{2,ii}(3)-GT{ii}(3,4); %Z-error of the output
    XEY_Z(ii)   = Z(1,4,ii)-GT{ii}(1,4); %X-error of the output
    YEY_Z(ii)   = Z(2,4,ii)-GT{ii}(2,4); %Y-error of the output
    ZEY_Z(ii)   = Z(3,4,ii)-GT{ii}(3,4); %Z-error of the output
    
    EY_CV = GT{ii}(1:3,4)-Res_PF_CV.PF_CV_Y{2,ii};   
    EY_NS = GT{ii}(1:3,4)-Res_PF_NS.PF_NS_Y{2,ii}; 
    if meas == "chris"
        EY_GUPF = GT{ii}(1:3,4)-Res_GUPF_NS.GUPF_NS_Z_Y{2,ii};    %Christiaan
        o_GUPF(:,ii) = Res_GUPF_NS.GUPF_NS_Z_Y{2,ii}; %Christiaan
        R_GUPF(:,:,ii) = Res_GUPF_NS.GUPF_NS_Z_Y{1,ii}; %Christiaan
        NYR_GUPF(ii) = rad2deg(norm(logm((GT{ii}(1:3,1:3))\Res_GUPF_NS.GUPF_NS_Z_Y{1,ii}))); %||log(R_GT^-1 * R_Y)||%
    elseif meas == "synth"
        EY_GUPF = GT{ii}(1:3,4)-Res_GUPF_NS.Y{2,ii}; %Synthetic
        o_GUPF(:,ii) = Res_GUPF_NS.Y{2,ii}; %Synthetic
        R_GUPF(:,:,ii) = Res_GUPF_NS.Y{1,ii}; %Synthetic
        NYR_GUPF(ii) = rad2deg(norm(logm((GT{ii}(1:3,1:3))\Res_GUPF_NS.Y{1,ii}))); %||log(R_GT^-1 * R_Y)||%
    end
    EY_Z  = GT{ii}(1:3,4)-Z(1:3,4,ii);

    o_GT(:,ii) = GT{ii}(1:3,4);
    o_CV(:,ii) = Res_PF_CV.PF_CV_Y{2,ii};
    o_NS(:,ii) = Res_PF_NS.PF_NS_Y{2,ii};
    
    R_GT(:,:,ii) = GT{ii}(1:3,1:3);
    R_CV(:,:,ii) = Res_PF_CV.PF_CV_Y{1,ii};
    R_NS(:,:,ii) = Res_PF_NS.PF_NS_Y{1,ii};
      
    
    NEY_CV(ii) = norm(EY_CV);  %Norm of the error of the output 
    NYR_CV(ii) = rad2deg(norm(logm((GT{ii}(1:3,1:3))\Res_PF_CV.PF_CV_Y{1,ii}))); %||log(R_GT^-1 * R_Y)||%
    NEY_NS(ii) = norm(EY_NS);  %Norm of the error of the output 
    NYR_NS(ii) = rad2deg(norm(logm((GT{ii}(1:3,1:3))\Res_PF_NS.PF_NS_Y{1,ii}))); %||log(R_GT^-1 * R_Y)||%
    NEY_GUPF(ii) = norm(EY_GUPF);  %Norm of the error of the output   
    NEY_Z(ii) = norm(EY_Z);  %Norm of the error of the output 
    NYR_Z(ii) = rad2deg(norm(logm((GT{ii}(1:3,1:3))\Z(1:3,1:3,ii)))); %||log(R_GT^-1 * R_Y)||%
end 

%% Compute RMS values
RMS_pos_NS    = rms(e_o);
RMS_pos_PF_CV = rms(NEY_CV);
RMS_pos_PF_NS = rms(NEY_NS);
RMS_pos_GUPF  = rms(NEY_GUPF);
RMS_pos_Z     = rms(NEY_Z);

RMS_rot_NS    = rms(e_R);
RMS_rot_PF_CV = rms(NYR_CV);
RMS_rot_PF_NS = rms(NYR_NS);
RMS_rot_GUPF  = rms(NYR_GUPF);
RMS_rot_Z     = rms(NYR_Z);

T= table([RMS_pos_NS; RMS_pos_PF_CV; RMS_pos_PF_NS; RMS_pos_GUPF; RMS_pos_Z], [RMS_rot_NS;RMS_rot_PF_CV;RMS_rot_PF_NS;RMS_rot_GUPF; RMS_rot_Z], ...
    'VariableNames',{'$\|e_{o}\|_{RMS} [m]$','$\|e_{R}\|_{RMS} [deg]$'},'RowNames',{'MM','PF_CV','PF_NS','GUPF','Z'});
disp(T)


%% Plot figures
%Create a grid, plot figures at these xy coordinates
im_width = 240;
im_height = 150;
px = 5:(im_width+5):2000;
py = 45:(im_height+90):2000;
for  ii = 1:length(px)
    for jj = 1:length(py)
        pp{jj,ii} = [px(ii) py(jj)];
    end 
end 

J = customcolormap(linspace(0,1,22), fliplr({'#DCD9D0','#D2D1CA', '#C7C9C4','#BDC1BE','#B3B9B8', '#A9B1B2', '#9EAAAC', '#94A2A6', '#8A9AA0', '#7F929A', '#758A94', '#6B828F', '#617A89', '#567283', '#4C6A7D', '#426277', '#375B71', '#2D536B', '#234B65', '#19435F', '#0E3B59', '#043353'}));

%% 
% Position error over time
figure('pos',[pp{1,4} im_width im_height]); 
if meas == "synth"
    plot(e_o,'linewidth',1.5,'color',tue.colorNS); hold on;
    plot(NEY_NS,'-.','linewidth',1.5,'color',tue.colorPFNS);
    plot(NEY_CV,'linewidth',1.5,'color',tue.colorPFCV);
end
plot(NEY_GUPF,'-.','linewidth',1.5,'color',tue.colorGUPFNS);hold on;
plot(NEY_Z,'-','linewidth',1,'color',tue.colorZ);
xline(impacts,'-.');
xlabel('Frame [-]');
xlim([1 60]);
ylim([0 0.15]);
ylabel('Position error [m]');
grid on;
fontsize(gcf,9,"points")
if doSave ==1; fig = gcf; fig.PaperPositionMode = 'auto'; fig_pos = fig.PaperPosition; fig.PaperSize = [fig_pos(3) fig_pos(4)];
    if meas == "chris"
        print(fig,append('plotting/paper_figures/pos-error-',object,'-christiaan.pdf'),'-dpdf','-painters')
    elseif meas == "synth"
        print(fig,append('plotting/paper_figures/pos-error-',object,'.pdf'),'-dpdf','-painters')
    end    
end

% Orientation error over time
figure('pos',[pp{1,5} im_width im_height]);
if meas == "synth"
    plot(e_R,'linewidth',1.5,'color',tue.colorNS); hold on;
    plot(NYR_NS,'-.','linewidth',1.5,'color',tue.colorPFNS);
    plot(NYR_CV,'linewidth',1.5,'color',tue.colorPFCV);
end
plot(NYR_GUPF,'-.','linewidth',1.5,'color',tue.colorGUPFNS); hold on;
plot(NYR_Z,'-','linewidth',1,'color',tue.colorZ);
xline(impacts,'-.');
xlabel('Frame [-]');
xlim([1 60]);
ylim([0 40]);
ylabel('Orientation error [deg]');
grid on;
fontsize(gcf,9,"points")
if doSave ==1; fig = gcf; fig.PaperPositionMode = 'auto'; fig_pos = fig.PaperPosition; fig.PaperSize = [fig_pos(3) fig_pos(4)]; 
    if meas == "chris"
        print(fig,append('plotting/paper_figures/rot-error-',object,'-christiaan.pdf'),'-dpdf','-painters')
    elseif meas == "synth"
        print(fig,append('plotting/paper_figures/rot-error-',object,'.pdf'),'-dpdf','-painters')
    end    
end

%%
% Show position trajectories over time
figure('rend','painters','pos',[pp{1,1} im_width im_height]); 
    ha = tight_subplot(1,1,[.08 .09],[.21 .05],[0.17 0.03]);  %[gap_h gap_w] [lower upper] [left right]
    axes(ha(1));
    plot(o_GT(1,:),'k','linewidth',2);hold on;
    plot(o_SIM(1,:),'linewidth',1.5,'color',tue.colorNS);
    plot(o_CV(1,:),'linewidth',1.5,'color',tue.colorPFCV);
    plot(o_NS(1,:),'-.','linewidth',1.5,'color',tue.colorPFNS);
    plot(o_GUPF(1,:),'-.','linewidth',1.5,'color',tue.colorGUPFNS);
    grid on;
    xline(impacts,':','color',[0.1 0.1 0.1],'LineWidth',1);
    xlim([1 60]);
    ylim([-0.25 0.5]);
    set(gca,'XTickLabel',[1 10 20 30 40 50 60],'XTick',[1 10 20 30 40 50 60]);
    xlabel('Frame [-]');
    ylabel('Position $x$ [m]');
    if doSave ==1; fig = gcf; fig.PaperPositionMode = 'auto'; fig_pos = fig.PaperPosition; fig.PaperSize = [fig_pos(3) fig_pos(4)];
        print(fig,append('plotting/paper_figures/pos-x',object,'.pdf'),'-dpdf','-painters')
    end

figure('rend','painters','pos',[pp{1,2} im_width im_height]); 
    ha = tight_subplot(1,1,[.08 .09],[.21 .05],[0.17 0.03]);  %[gap_h gap_w] [lower upper] [left right]
    axes(ha(1));
    plot(o_GT(2,:),'k','linewidth',2);hold on;
    plot(o_SIM(2,:),'linewidth',1.5,'color',tue.colorNS);
    plot(o_CV(2,1:25),'linewidth',1.5,'color',tue.colorPFCV);
    plot(o_NS(2,:),'-.','linewidth',1.5,'color',tue.colorPFNS);
    plot(o_GUPF(2,:),'-.','linewidth',1.5,'color',tue.colorGUPFNS);
    grid on;
    xline(impacts,':');
    xlim([1 60]);
    ylim([0.27 0.5]);
    set(gca,'XTickLabel',[1 10 20 30 40 50 60],'XTick',[1 10 20 30 40 50 60]);
    xlabel('Frame [-]');
    ylabel('Position $y$ [m]');
    if doSave ==1; fig = gcf; fig.PaperPositionMode = 'auto'; fig_pos = fig.PaperPosition; fig.PaperSize = [fig_pos(3) fig_pos(4)];
        print(fig,append('plotting/paper_figures/pos-y',object,'.pdf'),'-dpdf','-painters')
    end

figure('rend','painters','pos',[pp{1,3} im_width im_height]); 
    ha = tight_subplot(1,1,[.08 .09],[.21 .05],[0.17 0.03]);  %[gap_h gap_w] [lower upper] [left right]
    axes(ha(1));
    plot(o_GT(3,:),'k','linewidth',2);hold on;
    plot(o_SIM(3,:),'linewidth',1.5,'color',tue.colorNS);
    plot(o_CV(3,1:27),'linewidth',1.5,'color',tue.colorPFCV);
    plot(o_NS(3,:),'-.','linewidth',1.5,'color',tue.colorPFNS);
    plot(o_GUPF(3,:),'-.','linewidth',1.5,'color',tue.colorGUPFNS);
    grid on;
    xline(impacts,':');
    xlim([1 60]);
    ylim([0 0.3]);
    set(gca,'XTickLabel',[1 10 20 30 40 50 60],'XTick',[1 10 20 30 40 50 60]);
    xlabel('Frame [-]');
    ylabel('Position $z$ [m]');
    if doSave ==1; fig = gcf; fig.PaperPositionMode = 'auto'; fig_pos = fig.PaperPosition; fig.PaperSize = [fig_pos(3) fig_pos(4)];
        print(fig,append('plotting/paper_figures/pos-z',object,'.pdf'),'-dpdf','-painters')
    end
%%
% Show orientation trajectories over time
for ii = 1:length(GT)
    xi_GT(:,ii) = vee(logm(R_GT(:,:,ii)));
    xi_SIM(:,ii) = vee(logm(R_SIM(:,:,ii)));
    xi_CV(:,ii) = vee(logm(R_CV(:,:,ii)));
    xi_NS(:,ii) = vee(logm(R_NS(:,:,ii)));
    xi_GUPF(:,ii) = vee(logm(R_GUPF(:,:,ii)));
end

figure('rend','painters','pos',[pp{2,1} im_width im_height]); 
    ha = tight_subplot(1,1,[.08 .09],[.21 .05],[0.17 0.03]);  %[gap_h gap_w] [lower upper] [left right]
    axes(ha(1));
    plot(rad2deg(xi_GT(1,:)),'k','linewidth',2);hold on;
    plot(rad2deg(xi_SIM(1,:)),'linewidth',1.5,'color',tue.colorNS);
    plot(rad2deg(xi_CV(1,1:40)),'linewidth',1.5,'color',tue.colorPFCV);
    plot(rad2deg(xi_NS(1,:)),'-.','linewidth',1.5,'color',tue.colorPFNS);
    plot(rad2deg(xi_GUPF(1,:)),'-.','linewidth',1.5,'color',tue.colorGUPFNS);
    grid on;
    xline(impacts,':','color',[0.1 0.1 0.1],'LineWidth',1);
    xlim([1 60]);
    ylim([-20 40]);
    set(gca,'XTickLabel',[1 10 20 30 40 50 60],'XTick',[1 10 20 30 40 50 60]);
    xlabel('Frame [-]');
    ylabel('Rotation $\xi_x$ [deg]');
    if doSave ==1; fig = gcf; fig.PaperPositionMode = 'auto'; fig_pos = fig.PaperPosition; fig.PaperSize = [fig_pos(3) fig_pos(4)];
        print(fig,append('plotting/paper_figures/rot-x',object,'.pdf'),'-dpdf','-painters')
    end

figure('rend','painters','pos',[pp{2,2} im_width im_height]); 
    ha = tight_subplot(1,1,[.08 .09],[.21 .05],[0.17 0.03]);  %[gap_h gap_w] [lower upper] [left right]
    axes(ha(1));
    plot(rad2deg(xi_GT(2,:)),'k','linewidth',2);hold on;
    plot(rad2deg(xi_SIM(2,:)),'linewidth',1.5,'color',tue.colorNS);
    plot(rad2deg(xi_CV(2,1:40)),'linewidth',1.5,'color',tue.colorPFCV);
    plot(rad2deg(xi_NS(2,:)),'-.','linewidth',1.5,'color',tue.colorPFNS);
    plot(rad2deg(xi_GUPF(2,:)),'-.','linewidth',1.5,'color',tue.colorGUPFNS);
    grid on;
    xline(impacts,':');
    xlim([1 60]);
    ylim([-40 20]);
    set(gca,'XTickLabel',[1 10 20 30 40 50 60],'XTick',[1 10 20 30 40 50 60]);
    xlabel('Frame [-]');
    ylabel('Rotation $\xi_y$ [deg]');
    if doSave ==1; fig = gcf; fig.PaperPositionMode = 'auto'; fig_pos = fig.PaperPosition; fig.PaperSize = [fig_pos(3) fig_pos(4)];
        print(fig,append('plotting/paper_figures/rot-y',object,'.pdf'),'-dpdf','-painters')
    end

figure('rend','painters','pos',[pp{2,3} im_width im_height]); 
    ha = tight_subplot(1,1,[.08 .09],[.21 .05],[0.17 0.03]);  %[gap_h gap_w] [lower upper] [left right]
    axes(ha(1));
    plot(rad2deg(xi_GT(3,:)),'k','linewidth',2);hold on;
    plot(rad2deg(xi_SIM(3,:)),'linewidth',1.5,'color',tue.colorNS);
    plot(rad2deg(xi_CV(3,:)),'linewidth',1.5,'color',tue.colorPFCV);
    plot(rad2deg(xi_NS(3,:)),'-.','linewidth',1.5,'color',tue.colorPFNS);
    plot(rad2deg(xi_GUPF(3,:)),'-.','linewidth',1.5,'color',tue.colorGUPFNS);
    grid on;
    xline(impacts,':');
    xlim([1 60]);
    ylim([-80 0]);
    set(gca,'XTickLabel',[1 10 20 30 40 50 60],'XTick',[1 10 20 30 40 50 60]);
    xlabel('Frame [-]');
    ylabel('Rotation $\xi_z$ [deg]');
    if doSave ==1; fig = gcf; fig.PaperPositionMode = 'auto'; fig_pos = fig.PaperPosition; fig.PaperSize = [fig_pos(3) fig_pos(4)];
        print(fig,append('plotting/paper_figures/rot-z',object,'.pdf'),'-dpdf','-painters')
    end

% Plot 3D trajectory of the box
surface{1}.dim = [2.3 1];
surface{1}.speed=[0.4; 0; 0];
surface{1}.transform=[Ry(-0.5) [0.5; 0; 0]; zeros(1,3),1];

ws    = surface{1}.dim(1);                 %With of the contact surface             [m]
ls    = surface{1}.dim(2);               %Length of the contact surface           [m]
surfacepoints = [0.5*ws -0.5*ws -0.5*ws 0.5*ws 0.5*ws; -0.5*ls -0.5*ls 0.5*ls 0.5*ls -0.5*ls; 0 0 0 0 0;];
spoints{1} = surface{1}.transform(1:3,1:3)*surfacepoints + surface{1}.transform(1:3,4); %Transform the vertices according to position/orientation of the surface

tel =1;
plotidx = [1 3 5 7 9 11 13 15 17 19 21 25 30 40 50 60];
figure('Position',[pp{2,4} 485 290]);
for jj = 1:length(plotidx)
    %Plot the box
%     plotBox(GT{plotidx(jj)},box,tue.colorPFCV,true);
%     plotBox(GT{plotidx(jj)},box,[192 143 79]/255,true);
    plotBox(GT{plotidx(jj)},boxmodel,[205 159 97]/255,true);
    
    %Plot the inclined table C
    table3 = fill3(spoints{1}(1,1:4),spoints{1}(2,1:4),spoints{1}(3,1:4),1);hold on;
    set(table3,'FaceColor',0.8*[1 1 1],'FaceAlpha',1);
       
    axis([-1 1.0 -0.2 0.5 -0.2 0.5]);
    axis equal
    zoom(gcf,3.5)
    view(30,17)
    xlim([0,0.5769]);
    ylim([-0.0142,0.2369]);
    zlim([0.1809,0.2752]);
    drawnow
    camproj('perspective')
    axis off;
    set(gca,'clipping','off')
end
if doSave
    exportgraphics(gcf,append('plotting/paper_figures/',object,'trajectory.png'),Resolution=800)
end

% % show rgb image with projected box from sim results
% figure;
% for ii = 1:length(GT)
%     axis([0 720 0 540])
%     imshow(sel_img{ii});
%     H_state_cv = AH_M*[Res_Box006VP1_PF_CV.PF_CV_Y{1,ii} Res_Box006VP1_PF_CV.PF_CV_Y{2,ii}; 0 0 0 1];
%     H_state_gt = AH_M*GT{ii};
%     H_state_z = AH_M*Z(:,:,ii);
%     hold on;
%     pixel_cv(:,ii) = K*(H_state_cv(1:3,4)./H_state_cv(3,4));
%     pixel_gt(:,ii) = K*(H_state_gt(1:3,4)./H_state_gt(3,4));
%     pixel_z(:,ii) = K*(H_state_z(1:3,4)./H_state_z(3,4));
% %     plot(pixel_cv(1,1:ii),pixel_cv(2,1:ii),'color', [0,1,0])
% %     plot(pixel_gt(1,1:ii),pixel_gt(2,1:ii),'color', [1,0,0])
% %     plot(pixel_z(1,1:ii),pixel_z(2,1:ii),'color', [0,0,1])
% %     projectBox(H_state_gt,box,K,[1,0,0],1);
%     projectBox(H_state_cv,box,K,[0,1,0],1);
% %     projectBox(H_state_z,box,K,[0,0,1],1);
%     pause();
% %     pause(0.01);
% end

% Plot likelihood
%% Figures
load('plotdata.mat');
for ii = 1:3
    %%
    figure('rend','painters','pos',[pp{4,ii} im_width 200]); 
        ha = tight_subplot(1,1,[.08 .09],[.17 .01],[0.1 0.0]);  %[gap_h gap_w] [lower upper] [left right]
        axes(ha(1));
        b = bar3(lik{ii}.L_xy,1,'detached');
        bb = get(b(3),'parent');
        set(bb,'XTickLabel',trans_x_mm,'YTickLabel',trans_y_mm,'xtick',x_axis,'ytick',y_axis)
    %     set(bb,'xticklabel',[trans_x_mm(21:10:end-20)],'yticklabel',[trans_y_mm(21:10:end-20)], 'xtick',[x_axis(21:10:end-20)],'ytick',[y_axis(21:10:end-20)]);
        colormap(jet(256));
    
        for k = 1:length(b)
            zdata = b(k).ZData;
            b(k).CData = zdata;
            b(k).FaceColor = 'interp';
        end
    
        xlabel('y [mm]')
        ylabel('x [mm]')
        zlabel('L [-]')
        ylim([0.5 length(trans_x_mm)+0.5]) % depends on size translation vector
    %         colorbar(gca,"eastoutside",'TickLabelInterpreter','latex')
%         J = customcolormap([0 0.2 0.4 0.6 0.8 1], {'#FF0000','#FFA500', '#FFFF00','#00FF00','#00BFFF', '#00FFFF'});
        J = customcolormap(linspace(0,1,22), fliplr({'#DCD9D0','#D2D1CA', '#C7C9C4','#BDC1BE','#B3B9B8', '#A9B1B2', '#9EAAAC', '#94A2A6', '#8A9AA0', '#7F929A', '#758A94', '#6B828F', '#617A89', '#567283', '#4C6A7D', '#426277', '#375B71', '#2D536B', '#234B65', '#19435F', '#0E3B59', '#043353'}));
        colormap(J);
        axis square;
        view(0,90);
    %     axis([21 81 21 81 0 1])
        fontsize(gcf,9,"points")
    %     if doSave ==1;
    %         exportgraphics(gcf,append('likelihood_validation/Photo',string(ii),'_Likelihood.png'),Resolution=600)
    %     end
        if doSave ==1; fig = gcf; fig.PaperPositionMode = 'auto'; fig_pos = fig.PaperPosition;
            fig.PaperSize = [fig_pos(3) fig_pos(4)]; print(fig,append('plotting/paper_figures/Photo',string(ii),'_Likelihood.pdf'),'-dpdf','-painters')
        end
    
    
    %% Figure2: Likelihood in z-direction
    figure('pos',[pp{3,ii} im_width im_height]);
        plot(trans_z_mm, lik{ii}.L_z,'LineWidth',1.3,'color',tue.tb)
%         hold on
%         plot(trans_z_mm, lik{ii}.S1_z, 'LineWidth',1)
%         hold on
%         plot(trans_z_mm, lik{ii}.S2_z, 'LineWidth',1)
%         hold on
%         plot(trans_z_mm, lik{ii}.S3_z, 'LineWidth',1)
        xlabel('z [mm]')
        ylabel('$L$ [-], $S_1$ [-], $S_2$ [-], $S_3$ [-]')
        ylabel('$L$ [-]')
        xticks([-50:10:50])
        yticks([0:0.2:1])
        fontsize(gcf,9,"points")
        grid on
        if doSave ==1; fig = gcf; fig.PaperPositionMode = 'auto'; fig_pos = fig.PaperPosition;
            fig.PaperSize = [fig_pos(3) fig_pos(4)]; print(fig,append('plotting/paper_figures/Photo',string(ii),'_Likelihood_Z.pdf'),'-dpdf','-painters')
        end
     
    %Figure 4: Plot particle in image
    fig = figure();
        ImageShow = imshow(img{ii});
        hold on
        projectBox(AH_B(:,:,ii),box{ii},K,[0 0 0],1);
        for i = 1:length(lik{ii}.see_face)
            plot(lik{ii}.pin{1,lik{ii}.see_face(i)}(1,:), lik{ii}.pin{1,lik{ii}.see_face(i)}(2,:),'.','color',tue.tb,'MarkerSize',16);
            plot(lik{ii}.pface{1,lik{ii}.see_face(i)}(1,:), lik{ii}.pface{1,lik{ii}.see_face(i)}(2,:),'.','color',tue.origami,'MarkerSize',16);
            plot(lik{ii}.pout(1,:), lik{ii}.pout(2,:),'.','color',tue.middadel,'MarkerSize',16);
        end
        xlim(lik{ii}.xlims);
        ylim(lik{ii}.ylims);
        fig.Position = [pp{1,6} 650 470];
        if doSave ==1; 
            exportgraphics(fig,append('plotting/paper_figures/Photo',string(ii),'.png'),'Resolution',600)
        end
end

%% Plotting points on the manifold
for ii = 1:3
    figure('rend','painters','pos',[pp{1,ii}+[180*(ii-1) 0] 420 300]);
    ha = tight_subplot(1,1,[.08 .07],[-.5 -.25],[-0.6 -0.6]); %[gap_h gap_w] [lower upper] [left right]
    axes(ha(1));
    %Create the manifold
    [x1,y1,z1]=sphere;
    hSurface=surf(R*x1,R*y1,R*z1);hold on
%     set(hSurface,'FaceColor',tue.b,'FaceLighting','gouraud','FaceAlpha',0.05,'EdgeAlpha',0.2)%,'EdgeColor','none')
    set(hSurface,'FaceColor','#043353','FaceLighting','gouraud','FaceAlpha',0.05,'EdgeAlpha',0.1)%,'EdgeColor','none')
    daspect([1 1 1]);

    %Create the tangent space
    T = fill3(1.8*[-R R R -R],1.8*[-R -R R R],[R R R R],10);
%     set(T,'FaceColor',tue.r,'FaceAlpha',0.2,'EdgeColor','none');
    set(T,'FaceColor','#043353','FaceAlpha',0.3,'EdgeColor','none');

    %plot the lines on the manifold
    plot3([0 -0.7439 -1.469 -2.159 -2.795 -3.362 -3.847 -4.237 -4.523],...
        [0 0.2417 0.4775 0.7015 0.9082 1.093 1.25 1.377 1.469],...
        [5 4.940 4.757 4.457 4.047 3.538 2.941 2.272 1.547],'linewidth',1,'color',tue.tb);
    plot3([0  0.2417  0.4775 0.7015 0.9082 1.093  1.25],...
        [0 -0.7439 -1.469 -2.159 -2.795 -3.362 -3.847],...
        [5  4.940   4.757  4.457  4.047  3.538  2.941],'linewidth',1,'color',tue.tb);

    %Plot the points on the manifold
    plot3(-4.523,1.469,1.547,'.','MarkerSize',18,'color',tue.tb);
    plot3(1.25,-3.847,2.941,'.','MarkerSize',18,'color',tue.tb);

    %plot the lines on the tangent space
    plot3(1.5*[0,-4.523],1.5*[0,1.469],[R,R],'linewidth',1,'color',tue.herfstblad);
    plot3(1.2*[0,1.25],1.2*[0,-3.847],[R,R],'linewidth',1,'color',tue.herfstblad);

    if ii == 2 || ii ==3
        %Compute mean point on the tangent space
        mean = (1.5*[-4.697;-1.526;R/1.5]+1.2*[1.25;-3.847;R/1.2])./2;
        plot3(mean(1),mean(2),mean(3),'.','MarkerSize',18,'color',tue.origami);
        plot3([0 mean(1)],[0 mean(2)],[5 mean(3)],'linewidth',1,'color',tue.origami);
        text(-3.5,-3.8,4,'$\bar{\xi}$','FontSize',20);
    end

    if ii ==3
        %plot the mean on the manifold
        plot3([0 -0.633 -1.25 -1.836 -2.377 -2.860],...
            [0 -0.460 -0.908 -1.334 -1.727 -2.078],...
            [5 4.940 4.757 4.457 4.047 3.538],'linewidth',1,'color',tue.origami);
        plot3(-2.860,-2.078,3.538,'p','MarkerSize',10,'color',tue.origami,'MarkerFaceColor',tue.origami); %Plot the mean
        text(-4,-2,2.5,'$\bar{g}$','FontSize',20);
    end

    %plot the points on the tangent space
    plot3(-0.05,-0.05,R,'p','MarkerSize',10,'color',tue.herfstblad,'MarkerFaceColor',tue.herfstblad); %Plot the mean
    plot3(1.5*-4.523,1.5*1.469,R,'.','MarkerSize',18,'color',tue.herfstblad);
    plot3(1.2*1.25,1.2*-3.847,R,'.','MarkerSize',18,'color',tue.herfstblad);

    text(2,1.2,5.5,'$\tilde{g}$','FontSize',20);
    text(0.5,6.9,5.5,'$T_{\tilde{g}}\mathcal{G}$','FontSize',20,'Rotation',32);
    text(-7,3.6,5,'$\xi_i$','FontSize',20);
    text(1.5,-5.1,5,'$\xi_i$','FontSize',20);
    text(1.25,-4.2,3,'$g_i$','FontSize',20);
    text(-4.5,3,1.6,'$g_i$','FontSize',20);

    %Further plot options
    ax = gca;
    ax.SortMethod = 'childorder';
%     set(gca, 'Children', flipud(get(gca, 'Children')) )
    axis([-(R+5) (R+5) -(R+5) (R+5) -(R+5) (R+5)]);
    axis off
    set(gca,'xtick',[])
    set(gca,'ytick',[])
    set(gca,'ztick',[])
    view(-66,16)
    %    tightfig;
    if doSave
        fig = gcf;
        fig.Color = [0.94 0.94 0.94];
        set(gcf, 'InvertHardcopy', 'off');
        fig.PaperPositionMode = 'auto';
        fig_pos = fig.PaperPosition;
        fig.PaperSize = [fig_pos(3) fig_pos(4)];
        print(fig,sprintf('plotting/paper_figures/ManifoldMatlab%d.pdf',ii),'-dpdf','-vector')
    end
end
%%
   
%Plot the pi-ball
for ii = 1:3
    load(append('Photo',string(ii),'_Likelihood_rot.mat'));
    doSave = true;
    XvecX = [-3.5, 3.5];
    YvecX = [0, 0];
    ZvecX = [0, 0];
    
    XvecY = [0, 0];
    YvecY = [-3.5, 3.5];
    ZvecY = [0, 0];
    
    XvecZ = [0, 0];
    YvecZ = [0, 0];
    ZvecZ = [-3.5, 3.5];
    alpha_map = [ones(1,16)*0.02 ones(1,48)];
    
    figure('pos',[500 200 250 310]); 
    ha = tight_subplot(1,1,[.001 .001],[-.02 .12],[0.09 0.05]); %[gap_h gap_w] [low up] [lft rght]
    axes(ha(1));
    for i = 1:length(points)
        h(i) = surf(x2{i},y2{i},z2{i},Lnl{i});hold on
        alpha(h(i),'color');
        alphamap(alpha_map);
    end
    xlabel('x','position',[0 -5 -3.5],'Interpreter','Latex')
    ylabel('y','position',[-5 0 -3.5],'Interpreter','Latex')
    zlabel('z','position',[-3.9 4.5 0],'Interpreter','Latex')
    xlim([-3.5 3.5])
    ylim([-3.5 3.5])
    zlim([-3.5 3.5])
    xticks([-pi 0 pi])
    yticks([-pi 0 pi])
    zticks([-pi 0 pi])
    xticklabels({'$-\pi$', 0, '$\pi$'})
    yticklabels({'$-\pi$', 0, '$\pi$'})
    zticklabels({'$-\pi$', 0, '$\pi$'})
    
    shading interp;
    axis equal;
    fontsize(gcf,9,"points")
    % J = customcolormap([0 0.2 0.4 0.6 0.8 1], {'#FF0000','#FFA500', '#FFFF00','#00FF00','#00FFFF','#0000FF'});
%     J = customcolormap([0 0.2 0.4 0.6 0.8 1], {'#FF0000','#FFA500', '#FFFF00','#00FF00','#00BFFF', '#00FFFF'});
    colormap(J);
    q = colorbar('northoutside','TickLabelInterpreter','latex');
    q.Position = [0.05 0.8302 0.9 0.066]
    ylabel(q,'$L$','Interpreter','Latex')
    hold on
    line(XvecX,YvecX,ZvecX,'linewidth',1.5)
    hold on
    line(XvecY,YvecY,ZvecY,'linewidth',1.5)
    hold on
    line(XvecZ,YvecZ,ZvecZ,'linewidth',1.5)

    if doSave
        exportgraphics(gcf,append('plotting/paper_figures/Photo',string(ii),'_Likelihood_rot.png'),Resolution=800)
    end
end



% %% Figure(1): Plotting the manifold 
% figure('rend','painters','pos',[500 300 680 680]);
%    ha = tight_subplot(1,1,[.08 .07],[.003 .03],[.05 .05]); %[gap_h gap_w] [lower upper] [left right]
%    axes(ha(1));
%    %Create the manifold
%    [x1,y1,z1]=sphere;
%    hSurface=surf(R*x1,R*y1,R*z1);hold on
%    set(hSurface,'FaceColor',tue.b,'FaceLighting','gouraud','FaceAlpha',0.05,'EdgeAlpha',0.2)%,'EdgeColor','none')
%    daspect([1 1 1]);
%    
%    %Create the tangent space
%    T = fill3(1.8*[-R R R -R],1.8*[-R -R R R],[R R R R],10);
%    set(T,'FaceColor',tue.r,'FaceAlpha',0.2,'EdgeColor','none');
%    
%    %Plot the points on the manifold
%    plot3(-4.697,1.526,0.7822,'.','MarkerSize',18,'color',tue.b);
%    
%    %plot the points on the tangent space
%    plot3(0,0,R,'p','MarkerSize',10,'color',tue.r,'MarkerFaceColor',tue.r); %Plot the mean
%    plot3(1.5*-4.697,1.5*1.526,R,'.','MarkerSize',18,'color',tue.r);
% 
%    %Further plot options
%    axis([-(R+5) (R+5) -(R+5) (R+5) -(R+5) (R+5)])
%    set(gca,'xtick',[])
%    set(gca,'ytick',[])
%    set(gca,'ztick',[])
%    view(-66,16)
% %    tightfig;
%    if doSave ==1
%       print -painters -dpdf Figures/Mappings.pdf
%    end
%    
% %% Plotting the sigma points
% figure('rend','painters','pos',[500 300 680 680]);
%    ha = tight_subplot(1,1,[.08 .07],[.003 .03],[.05 .05]); %[gap_h gap_w] [lower upper] [left right]
%    axes(ha(1));
%    %Create the manifold
%    [x1,y1,z1]=sphere;
%    hSurface=surf(R*x1,R*y1,R*z1);hold on
%    set(hSurface,'FaceColor',tue.b,'FaceLighting','gouraud','FaceAlpha',0.05,'EdgeAlpha',0.2)%'EdgeColor','none')
%    daspect([1 1 1]);
%    
%    %Create the tangent space
%    T = fill3(1.8*[-R R R -R],1.8*[-R -R R R],[R R R R],10);
%    set(T,'FaceColor',tue.r,'FaceAlpha',0.2,'EdgeColor','none');
% 
%    %plot the covariance circle
%    theta = linspace(0,2*pi,1000);
%    for ii = 1:length(theta)
%        x(ii) = 4*cos(theta(ii));
%        y(ii) = 4*sin(theta(ii));
%        z(ii) = R;
%    end
%    plot3(x,y,z,'linewidth',0.2,'color','k');
%    
%    %Plot the mean and sigma points 
%    plot3(0,0,R,'p','MarkerSize',10,'color',tue.r,'MarkerFaceColor',tue.r);  %Plot the mean
%    plot3(3.159,-2.454,5,'.','MarkerSize',18,'color',tue.r);%Plot sigma points
% %    plot3([0 4 0 -4],[-4 0 4 0],[5 5 5 5],'.','MarkerSize',18,'color',tue.r);%Plot sigma points
% %    plot3([0 4 0 -4],[-4 0 4 0],[2.939*ones(1,4)],'.','MarkerSize',18,'color',tue.b);%Plot sigma points
%    
%    %Further plot options
%    axis([-(R+5) (R+5) -(R+5) (R+5) -(R+5) (R+5)])
%    set(gca,'xtick',[])
%    set(gca,'ytick',[])
%    set(gca,'ztick',[])
%    view(-66,16);
% %    camlight;
% %    tightfig;
%    if doSave ==1
%       print -painters -dpdf Figures/SigmaPointsMatlab.pdf
%    end

%% Synchronization plots
% figure('rend','painters','pos',[pp{1,1} 0.6*im_width im_height]); 
%     ha = tight_subplot(1,1,[.08 .09],[.2 .05],[0.21 0.03]);  %[gap_h gap_w] [lower upper] [left right]
%     axes(ha(1));
%     plot(plotdata.t_Aruco,plotdata.y_Aruco);
%     hold on;
%     plot(plotdata.t_Mocap,plotdata.y_Mocap);
%     ylabel("$(^M\mathbf{o}_D)_z$ [m]");
%     xlabel("Time [s]");
%     x = [min(plotdata.y_Mocap(1:1000)) max(plotdata.y_Mocap(1:1000))];
%     ylim((x-mean(x))*1.2+mean(x));
%     xlim([0 plotdata.t_Aruco(find(~isnan(plotdata.y_Aruco),1,'last'))]);
%     
% %     legend('Aruco data','Mocap data');
%     if doSave ==1; fig = gcf; fig.PaperPositionMode = 'auto'; fig_pos = fig.PaperPosition;
%         fig.PaperSize = [fig_pos(3) fig_pos(4)]; print(fig,'Images/ArucoMocap.pdf','-dpdf','-painters')
%     end
% 
% %Frame rate RGB
% figure('rend','painters','pos',[pp{1,2} 0.6*im_width im_height]); 
%     ha = tight_subplot(1,1,[.08 .09],[.21 .05],[0.21 0.03]);  %[gap_h gap_w] [lower upper] [left right]
%     axes(ha(1));
%     plot(plotdata.t_Aruco(1:end-1),1./diff(plotdata.t_Aruco));
%     ylabel("Framerate [FPS]");
%     xlabel("Time [s]");
%     ylim([58 62]);
%     xlim([0 plotdata.t_Aruco(end)]);
%     % legend('Framerate RGB');
%     if doSave ==1; fig = gcf; fig.PaperPositionMode = 'auto'; fig_pos = fig.PaperPosition;
%         fig.PaperSize = [fig_pos(3) fig_pos(4)]; print(fig,'Images/FrameRate.pdf','-dpdf','-painters')
%     end
% 
% 
% %Frame rate Mocap
% figure('rend','painters','pos',[pp{1,3} 0.6*im_width im_height]); 
%     ha = tight_subplot(1,1,[.08 .09],[.21 .05],[0.21 0.03]);  %[gap_h gap_w] [lower upper] [left right]
%     axes(ha(1));
%     plot(plotdata.t_Mocap(1:end-1),1./diff(plotdata.t_Mocap));
%     ylabel("Framerate [FPS]");
%     xlabel("Time [s]");
%     ylim([358 362]);
%     xlim([0 plotdata.t_Mocap(end)]);
%     % legend('Framerate Mocap');
%     if doSave ==1; fig = gcf; fig.PaperPositionMode = 'auto'; fig_pos = fig.PaperPosition;
%         fig.PaperSize = [fig_pos(3) fig_pos(4)]; print(fig,'Images/FrameRateMocap.pdf','-dpdf','-painters')
%     end
% 
% %Errorplot
% [yp,xp] =min(plotdata.Nerror);
% figure('rend','painters','pos',[pp{1,4} im_width im_height]); 
%     ha = tight_subplot(1,1,[.08 .09],[.2 .05],[0.13 0.01]);  %[gap_h gap_w] [lower upper] [left right]
%     axes(ha(1));
%     plot(plotdata.t_vec,plotdata.Nerror);
%     hold on;
%     plot(plotdata.t_vec(xp),yp,'o');
%     ylabel("$e_{pos}$ [m]");
%     xlabel("Time shift [s]");
%     ylim([0 4]);
%     xlim([0 plotdata.t_vec(end)]);
%     %     legend('$\sqrt{\sum_{i=i}^N |y_{Aruco}(i)-y_{Matlab}(i)|^2}$','Minimum at $t_{shift} = 871$ [ms]','Location','SouthWest');
%     if doSave ==1; fig = gcf; fig.PaperPositionMode = 'auto'; fig_pos = fig.PaperPosition;
%         fig.PaperSize = [fig_pos(3) fig_pos(4)]; print(fig,'Images/Error.pdf','-dpdf','-painters')
%     end
% 
% %Synchronization complete
% figure('rend','painters','pos',[pp{1,5} 0.6*im_width im_height]); 
%     ha = tight_subplot(1,1,[.08 .09],[.2 .05],[0.21 0.03]);  %[gap_h gap_w] [lower upper] [left right]
%     axes(ha(1));
%     plot(plotdata.t_Aruco,plotdata.y_Aruco);
%     hold on;
%     plot(plotdata.t_Aruco(plotdata.indx_con),plotdata.y_Mocap(plotdata.indx_mocap(plotdata.indx_ts_opt,plotdata.indx_con)));
%     ylabel("$(^M\mathbf{o}_D)_z$ [m]");
%     xlabel("Time [s]");
%     ylim((x-mean(x))*1.2+mean(x));
%     xlim([0 plotdata.t_Aruco(find(~isnan(plotdata.y_Aruco),1,'last'))]);
%     % legend('Aruco data','Mocap data');
%     if doSave ==1; fig = gcf; fig.PaperPositionMode = 'auto'; fig_pos = fig.PaperPosition;
%         fig.PaperSize = [fig_pos(3) fig_pos(4)]; print(fig,'Images/MocapArucoSynchronized.pdf','-dpdf','-painters')
%     end
% 
% %%
% 
ii=3
figure('rend','painters','pos',[pp{1,6} 0.93*im_width im_height]); 
    ha = tight_subplot(1,1,[.08 .09],[.2 .05],[0.13 -0.01]);  %[gap_h gap_w] [lower upper] [left right]
    hold on;
    for fi = 1:6
        plot3(box{ii}.Epoints{fi}(1,:), box{ii}.Epoints{fi}(2,:), box{ii}.Epoints{fi}(3,:),'.','color',tue.tb,'MarkerSize',6);
        plot3(box{ii}.Spoints{fi}(1,:), box{ii}.Spoints{fi}(2,:), box{ii}.Spoints{fi}(3,:),'.','color',tue.origami,'MarkerSize',6);
    end
    for jj=1:12
        plot3(box{ii}.Opoints{jj}(1,:),box{ii}.Opoints{jj}(2,:),box{ii}.Opoints{jj}(3,:),'+','color',tue.middadel,'MarkerSize',6);
        %Plot the black contour of the box
        two_vertices=[box{ii}.vertices(:,box{ii}.edges(1,jj)),box{ii}.vertices(:,box{ii}.edges(2,jj))];
        line(two_vertices(1,:),two_vertices(2,:),two_vertices(3,:),'color','k','LineWidth',1);
    end
    axis equal
    view(-37,23)
    axis([-box{ii}.dim(1)/2-0.01 box{ii}.dim(1)/2+0.01 -box{ii}.dim(2)/2-0.01 box{ii}.dim(2)/2+0.01 -box{ii}.dim(3)/2-0.01 box{ii}.dim(3)/2+0.01]);
    xlabel('x [m]','position',[  -0.01 -0.180 -0.05]);
    ylabel('y [m]','position',[-0.17    0 -0.1]);
    zlabel('z [m]','position',[-0.19  0.100   0.03]);
    ax = gca;
    ax.XTick = [-0.1,0,0.1];
    ax.XTickLabel = {'-0.1','0','0.1'};
    ax.YTick = [-0.05,0,0.05];
    ax.YTickLabel = {'-0.05','0','0.05'};
    grid on;
    set(gca,"FontSize",10)
    if doSave ==1; fig = gcf; fig.PaperPositionMode = 'auto'; fig_pos = fig.PaperPosition;
        fig.PaperSize = [fig_pos(3) fig_pos(4)]; print(fig,'Images/BoxGeoModel.pdf','-dpdf','-painters')
    end
% 
% %% Plot detected box in the image
% figure; 
%     imshow(sel_img{1}); hold on; 
%     plot([bounding_boxes.Pixel_coordinates(1,:,1) bounding_boxes.Pixel_coordinates(1,1,1)],[bounding_boxes.Pixel_coordinates(2,:,1) bounding_boxes.Pixel_coordinates(2,1,1)],'color',[0 1 0],'linewidth',2)


%% Write video of aruco data
% video = VideoWriter('yourvideo.avi'); %create the video object
% open(video); %open the file for writing
% 
% imagefiles = dir('./Images/ArucoDetectImages/*.jpg');
% nfiles = length(imagefiles);    % Number of files found
% for ii=1:nfiles
%     currentfilename = append(imagefiles(ii).folder,'/',imagefiles(ii).name);
%     I = imread(currentfilename); %read the next image
%     writeVideo(video,I); %write the image to file
% end
% close(video); %close the file

%% Write video of ground truth images
% video = VideoWriter('yourvideo1.avi'); %create the video object
% open(video); %open the file for writing
% 
% for i = 1:length(sel_img)
%     figure(1);
%     imshow(sel_img{i})
%     hold on
%     for k = 1:length(box.edges)
%         two_vertices=[pixel_coordinates{mocap_points(1,img_vec(i)),1}(1:2,box.edges(1,k)), pixel_coordinates{mocap_points(1,img_vec(i)),1}(1:2,box.edges(2,k))];
%         line(two_vertices(1,:),two_vertices(2,:),'color','r','LineWidth',1.5);
%     end
%     I = getframe(gcf);
%     writeVideo(video,I); %write the image to file
%     hold off;
% end
% close(video); %close the file

%% ------------------------------- FIGURE 1 ------------------------------- %%
% figure('rend','painters','pos',[pp{3,2} 1.2*450 200]); 
%     ha = tight_subplot(1,3,[.08 .09],[.18 .18],[0.08 0.02]);  %[gap_h gap_w] [lower upper] [left right]
%     axes(ha(1));
%     plot(EY{ple(1)}(1,:),'-','linewidth',1,'color',[0.518 0.824 0]);
%     hold on; grid on;
%     plot(EY{ple(2)}(1,:),'-','linewidth',1,'color','b');
%     plot(EY{ple(3)}(1,:),'-','linewidth',1,'color',[0.9290 0.6940 0.1250]);
%     plot(EY{ple(4)}(1,:),'-','linewidth',1,'color','r');
%     xlabel('Frame [-]');
%     ylabel('$x$-error, $e_x$ [m]')
%     axis([1 maxt -0.1 0.1]);
%     ha(1).XTick = [1 impacts(1) impacts(2) impacts(3) impacts(4) impacts(5) 40];
%     ha(1).XTickLabel = ({'1';num2str(impacts(1));num2str(impacts(2));num2str(impacts(3));num2str(impacts(4));num2str(impacts(5));'40'});
%     for ii = 1:length(impacts)
%     xline(impacts(ii),':','linewidth',1.2,'color',[0 0 0 1],'alpha',1);
%     end
%     
%     axes(ha(2));
%     plot(EY{ple(1)}(2,:),'-','linewidth',1,'color',[0.518 0.824 0]);
%     hold on; grid on;
%     plot(EY{ple(2)}(2,:),'-','linewidth',1,'color','b');
%     plot(EY{ple(3)}(2,:),'-','linewidth',1,'color',[0.9290 0.6940 0.1250]);
%     plot(EY{ple(4)}(2,:),'-','linewidth',1,'color','r');
%     xlabel('Frame [-]');
%     ylabel('$y$-error, $e_y$ [m]')
%     set(gca, 'YDir','reverse')
%     axis([1 maxt -0.1 0.1]);
%     ha(2).XTick = [1 impacts(1) impacts(2) impacts(3) impacts(4) impacts(5) 40];
%     ha(2).XTickLabel = ({'1';num2str(impacts(1));num2str(impacts(2));num2str(impacts(3));num2str(impacts(4));num2str(impacts(5));'40'});
%     for ii = 1:length(impacts)
%     xline(impacts(ii),':','linewidth',1.2,'color',[0 0 0 1],'alpha',1);
%     end
%    
%     axes(ha(3));
%     plot(EY{ple(1)}(3,:),'-','linewidth',1,'color',[0.518 0.824 0]);
%     hold on; grid on;
%     plot(EY{ple(2)}(3,:),'-','linewidth',1,'color','b');
%     plot(EY{ple(3)}(3,:),'-','linewidth',1,'color',[0.9290 0.6940 0.1250]);
%     plot(EY{ple(4)}(3,:),'-','linewidth',1,'color','r');
%     xlabel('Frame [-]');
%     ylabel('$z$-error, $e_z$ [m]')
%     axis([1 maxt -0.10 0.1]);   
%     ha(3).XTick = [1 impacts(1) impacts(2) impacts(3) impacts(4) impacts(5) 40];
%     ha(3).XTickLabel = ({'1';num2str(impacts(1));num2str(impacts(2));num2str(impacts(3));num2str(impacts(4));num2str(impacts(5));'40'});
%     for ii = 1:length(impacts)
%     xline(impacts(ii),':','linewidth',1.2,'color',[0 0 0 1],'alpha',1);
%     end
%     
%     L1 = legend('PF\_CV','PF\_NS','GUPF\_CV\_Z','GUPF\_NS\_Z','Impact Time','NumColumns',5,'location','northeast');
%     L1.Position(2) = 0.88;
%     L1.Position(1) = 0.5-(L1.Position(3)/2);
%     if doSave ==1
%         fig = gcf;
%         fig.PaperPositionMode = 'auto';
%         fig_pos = fig.PaperPosition;
%         fig.PaperSize = [fig_pos(3) fig_pos(4)];
%         print(fig,'Figures/POSErrors.pdf','-dpdf','-painters')
%     end   
    
%% ------------------------------- FIGURE 2 ------------------------------- %%
% figure('rend','painters','pos',[pp{2,3} 1.2*450 200]);
%     ha = tight_subplot(1,2,[0 .08],[.18 .18],[0.09 0.02]);%[gap_h gap_w] [low up ] [lft rght]
%     
%     axes(ha(1));
%     plot(NEY{ple(1)},'-','linewidth',1,'color',[0.518 0.824 0]);
%     hold on; grid on;
%     plot(NEY{ple(2)},'-','linewidth',1,'color','b');
%     plot(NEY{ple(3)},'-','linewidth',1,'color',[0.9290 0.6940 0.1250]);
%     plot(NEY{ple(4)},'-','linewidth',1,'color','r');
%     plot([1 40],[0 0],'-','linewidth',1,'color','k');
%     xlabel('Frame [-]');
%     ylabel('Position error $\|e_{\mathbf{o}}\|$ [m]')
%     axis([1 maxt -0.05 0.25]);    
%     ha(1).XTick = [1 impacts(1) impacts(2) impacts(3) impacts(4) impacts(5) 40];
%     ha(1).XTickLabel = ({'1';num2str(impacts(1));num2str(impacts(2));num2str(impacts(3));num2str(impacts(4));num2str(impacts(5));'40'});
%     for ii = 1:length(impacts)
%     xline(impacts(ii),':','linewidth',1.2,'color',[0 0 0 1],'alpha',1);
%     end
%     
%     axes(ha(2));
%     g1= plot(NYR{ple(1)},'-','linewidth',1,'color',[0.518 0.824 0]);
%     hold on; grid on;
%     g2= plot(NYR{ple(2)},'-','linewidth',1,'color','b');
%     g3= plot(NYR{ple(3)},'-','linewidth',1,'color',[0.9290 0.6940 0.1250]);
%     g4= plot(NYR{ple(4)},'-','linewidth',1,'color','r');
%     plot([1 40],[0 0],'-','linewidth',1,'color','k');
%     xlabel('Frame [-]');
%     ylabel('Rotation error $\|e_{\mathbf{R}}\|$ [deg]')
%     axis([1 maxt -4 30]);
%     ha(2).XTick = [1 impacts(1) impacts(2) impacts(3) impacts(4) impacts(5) 40];
%     ha(2).XTickLabel = ({'1';num2str(impacts(1));num2str(impacts(2));num2str(impacts(3));num2str(impacts(4));num2str(impacts(5));'40'});
%     for ii = 1:length(impacts)
%     g5 = xline(impacts(ii),':','linewidth',1.2,'color',[0 0 0 1],'alpha',1);
%     end
%     
%     L1 = legend([g1 g2 g3 g4 g5],'PF\_CV','PF\_NS','GUPF\_CV\_Z','GUPF\_NS\_Z','Impact Time','NumColumns',6,'location','northeast');
%     L1.Position(2) = 0.88;
%     L1.Position(1) = 0.5-(L1.Position(3)/2);
%     if doSave ==1
%         fig = gcf;
%         fig.PaperPositionMode = 'auto';
%         fig_pos = fig.PaperPosition;
%         fig.PaperSize = [fig_pos(3) fig_pos(4)];
%         print(fig,'Figures/POS_ROT_Errors.pdf','-dpdf','-painters')
%     end
    
%% ------------------------------- FIGURE 3 ------------------------------- %%

% figure('rend','painters','pos',[pp{3,4} 1.2*450 200]); 
%     ha = tight_subplot(1,3,[.08 .09],[.18 .18],[0.08 0.02]);  %[gap_h gap_w] [lower upper] [left right]
%     axes(ha(1));
%     plot(GTo(1,:),'-','linewidth',2,'color','k');
%     hold on; grid on;
%     plot(q{ple(1)}(1,:),'-','linewidth',1,'color',[0.518 0.824 0]);
%     plot(q{ple(2)}(1,:),'-','linewidth',1,'color','b');
%     plot(q{ple(3)}(1,:),'-','linewidth',1,'color',[0.9290 0.6940 0.1250]);
%     plot(q{ple(4)}(1,:),'-','linewidth',1,'color','r');
%     xlabel('Frame [-]');
%     ylabel('Position x [m]')
%     axis([1 maxt 0 0.85]);
%     ha(1).XTick = [1 impacts(1) impacts(2) impacts(3) impacts(4) impacts(5) 40];
%     ha(1).XTickLabel = ({'1';num2str(impacts(1));num2str(impacts(2));num2str(impacts(3));num2str(impacts(4));num2str(impacts(5));'40'});
%     for ii = 1:length(impacts)
%     xline(impacts(ii),':','linewidth',1.2,'color',[0 0 0 1],'alpha',1);
%     end
%     
%     axes(ha(2));
%     plot(GTo(2,:),'-','linewidth',2,'color','k');
%     hold on; grid on;
%     plot(q{ple(1)}(2,:),'-','linewidth',1,'color',[0.518 0.824 0]);
%     plot(q{ple(2)}(2,:),'-','linewidth',1,'color','b');
%     plot(q{ple(3)}(2,:),'-','linewidth',1,'color',[0.9290 0.6940 0.1250]);
%     plot(q{ple(4)}(2,:),'-','linewidth',1,'color','r');
%     xlabel('Frame [-]');
%     ylabel('Position y [m]')
%     set(gca, 'YDir','reverse')
%     axis([1 maxt 0.2 0.6]);
%     ha(2).XTick = [1 impacts(1) impacts(2) impacts(3) impacts(4) impacts(5) 40];
%     ha(2).XTickLabel = ({'1';num2str(impacts(1));num2str(impacts(2));num2str(impacts(3));num2str(impacts(4));num2str(impacts(5));'40'});
%     for ii = 1:length(impacts)
%     xline(impacts(ii),':','linewidth',1.2,'color',[0 0 0 1],'alpha',1);
%     end
%    
%     axes(ha(3));
%     plot(GTo(3,:),'-','linewidth',2,'color','k');
%     hold on; grid on;
%     plot(q{ple(1)}(3,:),'-','linewidth',1,'color',[0.518 0.824 0]);
%     plot(q{ple(2)}(3,:),'-','linewidth',1,'color','b');
%     plot(q{ple(3)}(3,:),'-','linewidth',1,'color',[0.9290 0.6940 0.1250]);
%     plot(q{ple(4)}(3,:),'-','linewidth',1,'color','r');
%     xlabel('Frame [-]');
%     ylabel('Position z [m]')
%     axis([1 maxt -0.1 0.5]);
%     ha(3).XTick = [1 impacts(1) impacts(2) impacts(3) impacts(4) impacts(5) 40];
%     ha(3).XTickLabel = ({'1';num2str(impacts(1));num2str(impacts(2));num2str(impacts(3));num2str(impacts(4));num2str(impacts(5));'40'});
%     for ii = 1:length(impacts)
%     xline(impacts(ii),':','linewidth',1.2,'color',[0 0 0 1],'alpha',1);
%     end   
%     
%     L1 = legend('GT','PF\_CV','PF\_NS','GUPF\_CV\_Z','GUPF\_NS\_Z','Impact Time','NumColumns',6,'location','northeast');
%     L1.Position(2) = 0.88;
%     L1.Position(1) = 0.5-(L1.Position(3)/2);
%     if doSave ==1
%         fig = gcf;
%         fig.PaperPositionMode = 'auto';
%         fig_pos = fig.PaperPosition;
%         fig.PaperSize = [fig_pos(3) fig_pos(4)];
%         print(fig,'Figures/ResultingTrajectory.pdf','-dpdf','-painters')
%     end

