clear all; close all; clc;
set(groot,'defaulttextinterpreter','latex'); set(groot,'defaultAxesTickLabelInterpreter','latex'); set(groot,'defaultLegendInterpreter','latex');
%% Add to path
addpath('Functions')
addpath('Data')
doSave = true;

%% Load data and parameters
folder = "likelihood_validation/data";
fndir = dir(folder);

cnt1 = 1;
cnt2 = 1;
for ii = 1:length(fndir)
    if endsWith(fndir(ii).name,'.txt') 
        fileID = fopen(append(fndir(ii).folder,'/',fndir(ii).name),'r');
        if ~startsWith(fndir(ii).name,'intrinsics')            
            AH_B(:,:,cnt1) = str2num(fscanf(fileID,'%c'));
            cnt1 = cnt1+1;
        else
            K = str2num(fscanf(fileID,'%c'));
        end
        fclose(fileID);
        
    elseif endsWith(fndir(ii).name,'.png')
        img{cnt2} = imread(fndir(ii).name);
        cnt2 = cnt2+1;
    end
end

%Create the box models
% Box006 = create_box_model(0.205,0.155,0.100);
% Box007 = create_box_model(0.207,0.158,0.099);
% Box009 = create_box_model(164,125,120);

box{1} = create_box_model(0.205,0.155,0.100); %Box006
box{2} = create_box_model(0.164,0.125,0.120);     %Box009
box{3} = create_box_model(0.207,0.158,0.099); %Box007

AH_B(:,:,1) = AH_B(:,:,1)*[eye(3),[-0.03;0.03;0]; zeros(1,3),1];
AH_B(:,:,2) = AH_B(:,:,2)*[eye(3),[-0.01;0.0;0]; zeros(1,3),1];
AH_B(:,:,3) = AH_B(:,:,3)*[eye(3),[0.01;0.003;0.005]; zeros(1,3),1];

%% Extract the data
for ii =3%:length(img)
    AH_M = eye(4); %we already express the state in the camera frame
    % Particle computation
    bins_hue = 12;
    bins_saturation = 12;
    bins_intensity = 12;

    Href = HIS(imread('RefImage.png'),bins_hue,bins_saturation,bins_intensity);
    hsi = rgb2hsi(img{ii},bins_hue,bins_saturation,bins_intensity);

    [x,y,z] = sphere(100);
    
%     points = [0:pi/12:pi];
    points = [0:pi/180:pi];
    
    for i = 1:length(points)
        for j = 1:length(x)
            for k = 1:length(y)
            x2{i}(j,k) = x(j,k)*points(i);
            y2{i}(j,k) = y(j,k)*points(i);
            z2{i}(j,k) = z(j,k)*points(i);
            end
        end
    end
    
    for i = 1:length(points)
        for j = 1:length(x)
            for k = 1:length(y)
            w_norm = sqrt(x2{i}(j,k)^2+y2{i}(j,k)^2+z2{i}(j,k)^2);
            w_hat = [0 -z2{i}(j,k) y2{i}(j,k); ...
                     z2{i}(j,k) 0 -x2{i}(j,k); ...
                     -y2{i}(j,k) x2{i}(j,k) 0];
                    
            R = eye(3)+(sin(w_norm)/w_norm)*w_hat+((1-cos(w_norm))/(w_norm)^2)*w_hat^2;
            R_FrameA{i}{j,k} = AH_B(1:3,1:3,ii)*R;
            end
        end
    end
    
    for j = 1:length(x)
        for k = 1:length(y)
            R_FrameA{1}{j,k} = eye(3);
        end
    end
    
    
    for i = 1:length(points)
        for j = 1:length(x)
            for k = 1:length(y)
            state{1,1}  = R_FrameA{i}{j,k};
            state{2,1}  = AH_B(1:3,4,ii);
            state{3,1}  = [0 0 0]';
            state{4,1}  = [0 0 0]';
            
            [L{i}(j,k),S1{i}(j,k),S2{i}(j,k),S3{i}(j,k)]=likelihood(state,Href,hsi,K,box{ii},eye(4));
            end
        end
        i
    end
    
    alpha_map = [ones(1,12)*0.02 ones(1,52)];
    
    %normalize L
    for jj = 1:length(L)
        maxL(jj) = max(L{jj}(:));
    end
    for jj = 1:length(L)
        Lnl{jj} = L{jj}/max(maxL);
    end

    %Figure 1
    figure('pos',[500 100 700 600]); 
        ha = tight_subplot(1,1,[.05 .05],[.05 .05],[0.08 0.03]); %[gap_h gap_w] [low up] [lft rght]
        axes(ha(1));
        for i = 1:length(points)
            h(i) = surf(x2{i},y2{i},z2{i},Lnl{i});
            xlabel('x','position',[0 -4 -3],'Interpreter','Latex','Fontsize',14)
            ylabel('y','position',[-4 0 -3],'Interpreter','Latex','Fontsize',14)
            zlabel('z','position',[-3.5 3.5 0],'Interpreter','Latex','Fontsize',14)
            
            alpha(h(i),'color');
            alphamap(alpha_map);
            shading interp;
            axis equal;
            % J = customcolormap([0 0.2 0.4 0.6 0.8 1], {'#FF0000','#FFA500', '#FFFF00','#00FF00','#00FFFF','#0000FF'});
            J = customcolormap([0 0.2 0.4 0.6 0.8 1], {'#FF0000','#FFA500', '#FFFF00','#00FF00','#00BFFF', '#00FFFF'});
            colormap(J);
            colorbar;
            hold on
        end

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
    ha = tight_subplot(1,1,[.001 .001],[.05 .015],[0.05 0.05]); %[gap_h gap_w] [low up] [lft rght]
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
    J = customcolormap([0 0.2 0.4 0.6 0.8 1], {'#FF0000','#FFA500', '#FFFF00','#00FF00','#00BFFF', '#00FFFF'});
    colormap(J);
    q = colorbar('northoutside','TickLabelInterpreter','latex');
    ylabel(q,'$L$','Interpreter','Latex')
    hold on
    line(XvecX,YvecX,ZvecX,'linewidth',1.5)
    hold on
    line(XvecY,YvecY,ZvecY,'linewidth',1.5)
    hold on
    line(XvecZ,YvecZ,ZvecZ,'linewidth',1.5)
    
%     if doSave ==1; fig = gcf; fig.PaperPositionMode = 'auto'; fig_pos = fig.PaperPosition;
%         fig.PaperSize = [fig_pos(3) fig_pos(4)]; print(fig,append('likelihood_validation/results/Photo',string(ii),'_Likelihood_rot.pdf'),'-dpdf','-painters')
%     end
    if doSave ==1;
        exportgraphics(gcf,append('likelihood_validation/results/Photo',string(ii),'_Likelihood_rot.png'),Resolution=800)
    end
end