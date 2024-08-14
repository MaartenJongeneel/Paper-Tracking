clear all; close all; clc;
set(groot,'defaulttextinterpreter','latex'); set(groot,'defaultAxesTickLabelInterpreter','latex'); set(groot,'defaultLegendInterpreter','latex');
%% Add to path
addpath('Functions')
addpath('Data')
doSave = false;

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

%%
AH_B(:,:,1) = AH_B(:,:,1)*[eye(3),[-0.03;0.03;0]; zeros(1,3),1];
AH_B(:,:,2) = AH_B(:,:,2)*[eye(3),[-0.01;0.0;0]; zeros(1,3),1];
AH_B(:,:,3) = AH_B(:,:,3)*[eye(3),[0.01;0.003;0.005]; zeros(1,3),1];

% ii = 3;
% figure;
% imshow(img{ii}); hold on;
% projectBox(AH_Bnew(:,:,ii),box{ii},K,[1 0 0],1)

%% Setting some colors
tue = struct('r',[0.784 0.098 0.098],...
    'cb',[0.063 0.063 0.451],...
    'b',[0 0.4 0.8],...
    'c',[0 0.635 0.871],...
    'g',[0.518 0.824 0],...
    'y',[0.808 0.875 0],...
    'delft',[0 0.651 0.8392],...
    'dg',[0 151/255 57/255]);

%% Extract the data
for ii =1:length(img)
    AH_M = eye(4); %we already express the state in the camera frame
    % Particle computation
    bins_hue = 12;
    bins_saturation = 12;
    bins_intensity = 12;

    Href = HIS(imread('RefImage.png'),bins_hue,bins_saturation,bins_intensity);
    hsi = rgb2hsi(img{ii},bins_hue,bins_saturation,bins_intensity);

    trans_x = [-0.05:0.01:0.05];     %[m]
    trans_y = [-0.05:0.01:0.05];     %[m]
    trans_z = [-0.05:0.01:0.05];     %[m]
    rot = [-200:1:200];               %[deg]

    for i = 1:length(trans_x)
        for j = 1:length(trans_y)
            particles{1}{i,j} = [AH_B(1,4,ii)+trans_x(i); AH_B(2,4,ii)+trans_y(j);  AH_B(3,4,ii)];
        end
    end

    for i = 1:length(trans_z)
        particles{2}{i} = [AH_B(1,4,ii); AH_B(2,4,ii); AH_B(3,4,ii)+trans_z(i)];
    end

    for i = 1:length(rot)
        particles{3}{i} = AH_B(1:3,1:3,ii)*Rx(rot(i));
        particles{4}{i} = AH_B(1:3,1:3,ii)*Ry(rot(i));
        particles{5}{i} = AH_B(1:3,1:3,ii)*Rz(rot(i));
    end

    %Construct H matrices
    for i = 1:length(particles)
        if i == 1
            for j = 1:length(trans_x)
                for k = 1:length(trans_y)
                    AH_B_particles = [AH_B(1:3,1:3,ii), particles{1}{j,k}; 0 0 0 1];
                    MH_B_particles{1}{j,k} = inv(AH_M)*AH_B_particles;
                end
            end
        end

        if i ~= 1
            for j = 1:length(particles{i})
                if i == 2
                    AH_B_particles = [AH_B(1:3,1:3,ii), particles{i}{j}; 0 0 0 1];
                    MH_B_particles{i}{j} = inv(AH_M)*AH_B_particles;
                end

                if i == 3 || i == 4 || i == 5
                    AH_B_particles = [particles{i}{j}, AH_B(1:3,4,ii); 0 0 0 1];
                    MH_B_particles{i}{j} = inv(AH_M)*AH_B_particles;
                end
            end
        end
    end

    %Construct state cells
    for i = 1:length(particles)
        if i == 1
            for j = 1:length(trans_x)
                for k = 1:length(trans_y)
                    state_xy{j,k}{1,1}  = MH_B_particles{1}{j,k}(1:3,1:3);
                    state_xy{j,k}{2,1}  = MH_B_particles{1}{j,k}(1:3,4);
                    state_xy{j,k}{3,1}  = [0 0 0]';
                    state_xy{j,k}{4,1}  = [0 0 0]';
                end
            end
        end

        if i ~= 1
            k=i-1;
            for j = 1:length(particles{i})
                state{k}{1,j}  = MH_B_particles{i}{j}(1:3,1:3);
                state{k}{2,j}  = MH_B_particles{i}{j}(1:3,4);
                state{k}{3,j}  = [0 0 0]';
                state{k}{4,j}  = [0 0 0]';
            end
        end
    end

    % Likelihood computation
    for i = 1:length(trans_x)
        for j = 1:length(trans_y)
            [lik{ii}.L_xy(i,j),lik{ii}.S1_xy(i,j),lik{ii}.S2_xy(i,j),lik{ii}.S3_xy(i,j)]=likelihood(state_xy{i,j},Href,hsi,K,box{ii},AH_M);
        end
    end
    lik{ii}.L_xy = lik{ii}.L_xy/(max(max(lik{ii}.L_xy)));

    for i = 1:length(state{1})
        [lik{ii}.L_z(i),lik{ii}.S1_z(i),lik{ii}.S2_z(i),lik{ii}.S3_z(i)]=likelihood(state{1,1}(:,i),Href,hsi,K,box{ii},AH_M);
    end
    lik{ii}.L_z = lik{ii}.L_z/(max(lik{ii}.L_z));

    for i = 2:4
        k = i-1;
        for j = 1:length(state{2})
            [lik{ii}.L_rot(k,j),lik{ii}.S1_rot(k,j),lik{ii}.S2_rot(k,j),lik{ii}.S3_rot(k,j)]=likelihood(state{1,i}(:,j),Href,hsi,K,box{ii},AH_M);
        end
    end
    lik{ii}.L_rot = lik{ii}.L_rot./max(lik{ii}.L_rot,[],2);

    trans_x_mm = trans_x*1000;
    trans_y_mm = trans_y*1000;
    trans_z_mm = trans_z*1000;

    x_axis = 1:length(trans_x_mm);
    y_axis = 1:length(trans_y_mm);

    particle_xy = true;
    particle_z = false;
    particle_rotx = false;
    particle_roty = false;
    particle_rotz = false;

    sel_x = 0;
    sel_y = 0;
    sel_z = 0;
    sel_rot = 0;

    id_x = find(abs(trans_x-sel_x) < 0.0001);
    id_y = find(abs(trans_y-sel_y) < 0.0001);
    id_z = find(abs(trans_z-sel_z) < 0.0001);
    id_rot = find(abs(rot-sel_rot) < 0.0001);

    if particle_xy
        for i = 1:length(box{ii}.vertices)
            [~,~,~,~,Hin,Hobj,Hout,lik{ii}.see_face,lik{ii}.pin,lik{ii}.pface,lik{ii}.pout] = likelihood(state_xy{id_x,id_y},Href,hsi,K,box{ii},AH_M);
            vc_particle(:,i) = particles{1}{id_x,id_y}+AH_B(1:3,1:3)*box{ii}.vertices(:,i);
            pc_particle(:,i) = K*(vc_particle(1:3,i)./vc_particle(3,i));
        end
    end

    if particle_z
        for i = 1:length(box{ii}.vertices)
            [~,~,~,~,Hin,Hobj,Hout,lik{ii}.see_face,lik{ii}.pin,lik{ii}.pface,lik{ii}.pout] = likelihood(state{1}{id_z},Href,hsi,K,box{ii},AH_M);
            vc_particle(:,i) = particles{2}{id_z}+AH_B(1:3,1:3)*box{ii}.vertices(:,i);
            pc_particle(:,i) = K*(vc_particle(1:3,i)./vc_particle(3,i));
        end
    end

    if particle_rotx
        for i = 1:length(box{ii}.vertices)
            [~,~,~,~,Hin,Hobj,Hout,lik{ii}.see_face,lik{ii}.pin,lik{ii}.pface,lik{ii}.pout] = likelihood(state{2}{id_rot},Href,hsi,K,box{ii},AH_M);
            vc_particle(:,i) = AH_B(1:3,4)+particles{3}{id_rot}*box{ii}.vertices(:,i);
            pc_particle(:,i) = K*(vc_particle(1:3,i)./vc_particle(3,i));
        end
    end

    if particle_roty
        for i = 1:length(box{ii}.vertices)
            [~,~,~,~,Hin,Hobj,Hout,lik{ii}.see_face,lik{ii}.pin,lik{ii}.pface,lik{ii}.pout] = likelihood(state{3}{id_rot},Href,hsi,K,box{ii},AH_M);
            vc_particle(:,i) = AH_B(1:3,4)+particles{4}{id_rot}*box{ii}.vertices(:,i);
            pc_particle(:,i) = K*(vc_particle(1:3,i)./vc_particle(3,i));
        end
    end

    if particle_rotz
        for i = 1:length(box{ii}.vertices)
            [~,~,~,~,Hin,Hobj,Hout,lik{ii}.see_face,lik{ii}.pin,lik{ii}.pface,lik{ii}.pout] = likelihood(state{4}{id_rot},Href,hsi,K,box{ii},AH_M);
            vc_particle(:,i) = AH_B(1:3,4)+particles{5}{id_rot}*box{ii}.vertices(:,i);
            pc_particle(:,i) = K*(vc_particle(1:3,i)./vc_particle(3,i));
        end
    end
end
lik{1}.xlims = [340.8313073916391,458.5570607867608];
lik{1}.ylims = [300.5204180564458,388.8147331027868];
lik{2}.xlims = [401.9906783160586,544.4388399241559];
lik{2}.ylims = [313.7117526158187,420.5478738218913];
lik{3}.xlims = [568.9715739447585,686.6973273398802];
lik{3}.ylims = [323.0660452107083,411.3603602570493];
clearvars -except lik trans_x_mm trans_y_mm x_axis y_axis trans_z_mm rot img AH_B box K tue doSave ii
%% Figures
% Figure 1: Likelihood
for ii = 1:3
    figure('rend','painters','pos',[50 50 250 200]);
        ha = tight_subplot(1,1,[.08 .09],[.16 .01],[0.15 0.0]);  %[gap_h gap_w] [lower upper] [left right]
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
        J = customcolormap([0 0.2 0.4 0.6 0.8 1], {'#FF0000','#FFA500', '#FFFF00','#00FF00','#00BFFF', '#00FFFF'});
        colormap(J);
        axis square;
        view(0,90);
    %     axis([21 81 21 81 0 1])
        fontsize(gcf,9,"points")
    %     if doSave ==1;
    %         exportgraphics(gcf,append('likelihood_validation/Photo',string(ii),'_Likelihood.png'),Resolution=600)
    %     end
        if doSave ==1; fig = gcf; fig.PaperPositionMode = 'auto'; fig_pos = fig.PaperPosition;
            fig.PaperSize = [fig_pos(3) fig_pos(4)]; print(fig,append('likelihood_validation/Photo',string(ii),'_Likelihood.pdf'),'-dpdf','-painters')
        end
    
    
    % Figure2: Likelihood in z-direction
    figure('pos',[310 50 250 200]);
        plot(trans_z_mm, lik{ii}.L_z,'LineWidth',1.3)
        hold on
        plot(trans_z_mm, lik{ii}.S1_z, 'LineWidth',1)
        hold on
        plot(trans_z_mm, lik{ii}.S2_z, 'LineWidth',1)
        hold on
        plot(trans_z_mm, lik{ii}.S3_z, 'LineWidth',1)
        xlabel('z [mm]')
        ylabel('$L$ [-], $S_1$ [-], $S_2$ [-], $S_3$ [-]')
        ylabel('$L$ [-]')
        xticks([-50:10:50])
        yticks([0:0.2:1])
        fontsize(gcf,9,"points")
        grid on
    %     if doSave ==1;
    %         exportgraphics(gcf,append('likelihood_validation/Photo',string(ii),'_Likelihood_Z.png'),Resolution=600)
    %     end
        if doSave ==1; fig = gcf; fig.PaperPositionMode = 'auto'; fig_pos = fig.PaperPosition;
            fig.PaperSize = [fig_pos(3) fig_pos(4)]; print(fig,append('likelihood_validation/Photo',string(ii),'_Likelihood_Z.pdf'),'-dpdf','-painters')
        end
    
    %Figure 3: Rotational Likelihood
    figure('pos',[710 50 1000 300]);
        ha = tight_subplot(1,3,[.08 .07],[.13 .13],[0.05 0.05]); %[gap_h gap_w] [low up] [lft rght]
        axes(ha(1));
        plot(rot, lik{ii}.L_rot(1,:),'LineWidth',1)
        hold on
        plot(rot, lik{ii}.S1_rot(1,:),'LineWidth',0.8)
        hold on
        plot(rot, lik{ii}.S2_rot(1,:),'LineWidth',0.8)
        hold on
        plot(rot, lik{ii}.S3_rot(1,:),'LineWidth',0.8)
        xlabel('Angle [deg]','Interpreter','Latex','Fontsize',14)
        ylabel('$L$ [-], $S_1$ [-], $S_2$ [-], $S_3$ [-]','Interpreter','Latex','Fontsize',14)
        grid on
    
        axes(ha(2));
        plot(rot, lik{ii}.L_rot(2,:),'LineWidth',1)
        hold on
        plot(rot, lik{ii}.S1_rot(2,:),'LineWidth',0.8)
        hold on
        plot(rot, lik{ii}.S2_rot(2,:),'LineWidth',0.8)
        hold on
        plot(rot, lik{ii}.S3_rot(2,:),'LineWidth',0.8)
        xlabel('Angle [deg]','Interpreter','Latex','Fontsize',14)
        ylabel('$L$ [-], $S_1$ [-], $S_2$ [-], $S_3$ [-]','Interpreter','Latex','Fontsize',14)
        grid on
    
        axes(ha(3));
        g1 = plot(rot, lik{ii}.L_rot(3,:),'LineWidth',1);
        hold on
        g2 = plot(rot, lik{ii}.S1_rot(3,:),'LineWidth',0.8);
        hold on
        g3 = plot(rot, lik{ii}.S2_rot(3,:),'LineWidth',0.8);
        hold on
        g4 = plot(rot, lik{ii}.S3_rot(3,:),'LineWidth',0.8);
        xlabel('Angle [deg]','Interpreter','Latex','Fontsize',14)
        ylabel('$L$ [-], $S_1$ [-], $S_2$ [-], $S_3$ [-]','Interpreter','Latex','Fontsize',14)
        grid on
    
        L1 = legend([g1 g2 g3 g4],{'$L$', '$S_1$', '$S_2$', '$S_3$'},'NumColumns',4,'location','northeast');
        L1.Position(2) = 0.91;
        L1.Position(1) = 0.5-(L1.Position(3)/2);
        L1.FontSize = 12;
    
    
    %Figure 4: Plot particle in image
    figure()
        ImageShow = imshow(img{ii});
        hold on
        projectBox(AH_B(:,:,ii),box{ii},K,[0 0 0],1);
        for i = 1:length(lik{ii}.see_face)
            plot(lik{ii}.pin{1,lik{ii}.see_face(i)}(1,:), lik{ii}.pin{1,lik{ii}.see_face(i)}(2,:),'.','color',tue.b,'MarkerSize',20);
            plot(lik{ii}.pface{1,lik{ii}.see_face(i)}(1,:), lik{ii}.pface{1,lik{ii}.see_face(i)}(2,:),'.','color',tue.dg,'MarkerSize',20);
            plot(lik{ii}.pout(1,:), lik{ii}.pout(2,:),'.','color',tue.r,'MarkerSize',20);
        end
        xlim(lik{ii}.xlims);
        ylim(lik{ii}.ylims);
end
%     for i = 1:length(box{ii}.edges)
%         two_vertices=[pc_particle(1:2,box{ii}.edges(1,i)), pc_particle(1:2,box{ii}.edges(2,i))];
%         line(two_vertices(1,:),two_vertices(2,:),'color','k','LineWidth',1.5);
%     end

%     % Figure 2: Similarities S1
%     figure('pos',[400 50 300 300]);
%     b = bar3(S1_xy,1,'detached');
%     bb = get(b(3),'parent');
%     set(bb,'xticklabel',[trans_x_mm(1:2:end)],'yticklabel',[trans_y_mm(1:2:end)], 'xtick',[x_axis(1:2:end)],'ytick',[y_axis(1:2:end)]);
%     colormap(jet(256));
% 
%     for k = 1:length(b)
%         zdata = b(k).ZData;
%         b(k).CData = zdata;
%         b(k).FaceColor = 'interp';
%     end
% 
%     xlabel('y [mm]','Interpreter','Latex','Fontsize',14)
%     ylabel('x [mm]','Interpreter','Latex','Fontsize',14)
%     zlabel('$S_1$ [-]','Interpreter','Latex','Fontsize',14)
%     ylim([0.5 length(trans_x_mm)+0.5]) % depends on size translation vector
%     colorbar
% 
% 
%     %Figure 3: Similarities S2
%     figure('pos',[750 50 300 300]);
%     b = bar3(S2_xy,1,'detached');
%     bb = get(b(3),'parent');
%     set(bb,'xticklabel',[trans_x_mm(1:2:end)],'yticklabel',[trans_y_mm(1:2:end)], 'xtick',[x_axis(1:2:end)],'ytick',[y_axis(1:2:end)]);
%     colormap(jet(256));
% 
%     for k = 1:length(b)
%         zdata = b(k).ZData;
%         b(k).CData = zdata;
%         b(k).FaceColor = 'interp';
%     end
% 
%     xlabel('y [mm]','Interpreter','Latex','Fontsize',14)
%     ylabel('x [mm]','Interpreter','Latex','Fontsize',14)
%     zlabel('$S_2$ [-]','Interpreter','Latex','Fontsize',14)
%     ylim([0.5 length(trans_x_mm)+0.5]) % depends on size translation vector
%     colorbar
% 
% 
%     %Figure 4: Similarities S3
%     figure('pos',[1100 50 300 300]);
%     b = bar3(S3_xy,1,'detached');
%     bb = get(b(3),'parent');
%     set(bb,'xticklabel',[trans_x_mm(1:2:end)],'yticklabel',[trans_y_mm(1:2:end)], 'xtick',[x_axis(1:2:end)],'ytick',[y_axis(1:2:end)]);
%     colormap(jet(256));
% 
%     for k = 1:length(b)
%         zdata = b(k).ZData;
%         b(k).CData = zdata;
%         b(k).FaceColor = 'interp';
%     end
% 
%     xlabel('y [mm]','Interpreter','Latex','Fontsize',14)
%     ylabel('x [mm]','Interpreter','Latex','Fontsize',14)
%     zlabel('$S_3$ [-]','Interpreter','Latex','Fontsize',14)
%     ylim([0.5 length(trans_x_mm)+0.5]) % depends on size translation vector
%     colorbar

%% Histograms
% locs_hue = [0:1:bins_hue-1];
% locs_sat = [0:1:bins_saturation-1];
% locs_int = [0:1:bins_intensity-1];
%
% %Reference histogram
% for i=1:bins_intensity
%     for j=1:bins_hue
%         Frequencies_Hue(i,j) = sum(Href(j,:,i));
%         Sum_Frequencies_Hue(1,j) = sum(Frequencies_Hue(:,j));
%     end
% end
% SF_Hue_Href = Sum_Frequencies_Hue;
%
% for i=1:bins_intensity
%     for j=1:bins_saturation
%         Frequencies_Saturation(i,j) = sum(Href(:,j,i));
%         Sum_Frequencies_Saturation(1,j) = sum(Frequencies_Saturation(:,j));
%     end
% end
% SF_Saturation_Href = Sum_Frequencies_Saturation;
%
% for i=1:bins_intensity
%     Frequencies_Intensity(1,i) = sum(sum(Href(:,:,i)));
% end
% F_Intensity_Href = Frequencies_Intensity;
%
% %Hout
% for i=1:bins_intensity
%     for j=1:bins_hue
%         Frequencies_Hue(i,j) = sum(Hout(j,:,i));
%         Sum_Frequencies_Hue(1,j) = sum(Frequencies_Hue(:,j));
%     end
% end
% SF_Hue_Hout = Sum_Frequencies_Hue;
%
% for i=1:bins_intensity
%     for j=1:bins_saturation
%         Frequencies_Saturation(i,j) = sum(Hout(:,j,i));
%         Sum_Frequencies_Saturation(1,j) = sum(Frequencies_Saturation(:,j));
%     end
% end
% SF_Saturation_Hout = Sum_Frequencies_Saturation;
%
% for i=1:bins_intensity
%     Frequencies_Intensity(1,i) = sum(sum(Hout(:,:,i)));
% end
% F_Intensity_Hout = Frequencies_Intensity;
%
% for i = 1:length(see_face)
%     %Hin
%     for j=1:bins_intensity
%         for k=1:bins_hue
%             Frequencies_Hue(j,k) = sum(Hin{see_face(i)}(k,:,j));
%             Sum_Frequencies_Hue(1,k) = sum(Frequencies_Hue(:,k));
%         end
%     end
%     SF_Hue_Hin{i} = Sum_Frequencies_Hue;
%
%     for j=1:bins_intensity
%         for k=1:bins_saturation
%             Frequencies_Saturation(j,k) = sum(Hin{see_face(i)}(:,k,j));
%             Sum_Frequencies_Saturation(1,k) = sum(Frequencies_Saturation(:,k));
%         end
%     end
%     SF_Saturation_Hin{i} = Sum_Frequencies_Saturation;
%
%     for j=1:bins_intensity
%         Frequencies_Intensity(1,j) = sum(sum(Hin{see_face(i)}(:,:,j)));
%     end
%     F_Intensity_Hin{i} = Frequencies_Intensity;
%
%     %Hobj
%     for j=1:bins_intensity
%         for k=1:bins_hue
%             Frequencies_Hue(j,k) = sum(Hobj{see_face(i)}(k,:,j));
%             Sum_Frequencies_Hue(1,k) = sum(Frequencies_Hue(:,k));
%         end
%     end
%     SF_Hue_Hobj{i} = Sum_Frequencies_Hue;
%
%     for j=1:bins_intensity
%         for k=1:bins_saturation
%             Frequencies_Saturation(j,k) = sum(Hobj{see_face(i)}(:,k,j));
%             Sum_Frequencies_Saturation(1,k) = sum(Frequencies_Saturation(:,k));
%         end
%     end
%     SF_Saturation_Hobj{i} = Sum_Frequencies_Saturation;
%
%
%     for j=1:bins_intensity
%         Frequencies_Intensity(1,j) = sum(sum(Hobj{see_face(i)}(:,:,j)));
%     end
%     F_Intensity_Hobj{i} = Frequencies_Intensity;
%
%     f = figure(100+i);
%     f.Position(1:4) = [300 200 1500 600];
%     ha = tight_subplot(1,6,[.03 .03],[.08 .05],[0.03 0.01]); %[gap_h gap_w] [low up] [lft rght]
%     axes(ha(1));
%     bar(locs_hue, SF_Hue_Hin{i}, 1, 'FaceColor','b','EdgeColor','k','LineWidth',1.5)
%     xlim([-0.5 bins_hue-0.5])
%     ylim([0 1])
%     xlabel('Bin number [-]','Interpreter','Latex','Fontsize',12)
%     ylabel('Normalized hue value [-]','Interpreter','Latex','Fontsize',12)
%     grid on
%
%     axes(ha(2));
%     bar(locs_sat, SF_Saturation_Hin{i}, 1, 'FaceColor','r','EdgeColor','k','LineWidth',1.5)
%     xlim([-0.5 bins_saturation-0.5])
%     ylim([0 1])
%     xlabel('Bin number [-]','Interpreter','Latex','Fontsize',12)
%     ylabel('Normalized saturation value [-]','Interpreter','Latex','Fontsize',12)
%     title('HSI Hin','Interpreter','Latex','Fontsize',12)
%     grid on
%
%     axes(ha(3));
%     bar(locs_int, F_Intensity_Hin{i}, 1, 'FaceColor','g','EdgeColor','k','LineWidth',1.5)
%     xlim([-0.5 bins_intensity-0.5])
%     ylim([0 1])
%     xlabel('Bin number [-]','Interpreter','Latex','Fontsize',12)
%     ylabel('Normalized intensity value [-]','Interpreter','Latex','Fontsize',12)
%     grid on
%
%     axes(ha(4));
%     bar(locs_hue, SF_Hue_Href, 1, 'FaceColor','b','EdgeColor','k','LineWidth',1.5)
%     xlim([-0.5 bins_hue-0.5])
%     ylim([0 1])
%     xlabel('Bin number [-]','Interpreter','Latex','Fontsize',12)
%     ylabel('Normalized hue value [-]','Interpreter','Latex','Fontsize',12)
%     grid on
%
%     axes(ha(5));
%     bar(locs_sat, SF_Saturation_Href, 1, 'FaceColor','r','EdgeColor','k','LineWidth',1.5)
%     xlim([-0.5 bins_saturation-0.5])
%     ylim([0 1])
%     xlabel('Bin number [-]','Interpreter','Latex','Fontsize',12)
%     ylabel('Normalized saturation value [-]','Interpreter','Latex','Fontsize',12)
%     title('HSI Href','Interpreter','Latex','Fontsize',12)
%     grid on
%
%     axes(ha(6));
%     bar(locs_int, F_Intensity_Href, 1, 'FaceColor','g','EdgeColor','k','LineWidth',1.5)
%     xlim([-0.5 bins_intensity-0.5])
%     ylim([0 1])
%     xlabel('Bin number [-]','Interpreter','Latex','Fontsize',12)
%     ylabel('Normalized intensity value [-]','Interpreter','Latex','Fontsize',12)
%     grid on
%
%     f = figure(200+i);
%     f.Position(1:4) = [300 200 1500 600];
%     ha = tight_subplot(1,6,[.03 .03],[.08 .05],[0.03 0.01]); %[gap_h gap_w] [low up] [lft rght]
%     axes(ha(1));
%     bar(locs_hue, SF_Hue_Hobj{i}, 1, 'FaceColor','b','EdgeColor','k','LineWidth',1.5)
%     xlim([-0.5 bins_hue-0.5])
%     ylim([0 1])
%     xlabel('Bin number [-]','Interpreter','Latex','Fontsize',12)
%     ylabel('Normalized hue value [-]','Interpreter','Latex','Fontsize',12)
%     grid on
%
%     axes(ha(2));
%     bar(locs_sat, SF_Saturation_Hobj{i}, 1, 'FaceColor','r','EdgeColor','k','LineWidth',1.5)
%     xlim([-0.5 bins_saturation-0.5])
%     ylim([0 1])
%     xlabel('Bin number [-]','Interpreter','Latex','Fontsize',12)
%     ylabel('Normalized saturation value [-]','Interpreter','Latex','Fontsize',12)
%     title('HSI Hobj','Interpreter','Latex','Fontsize',12)
%     grid on
%
%     axes(ha(3));
%     bar(locs_int, F_Intensity_Hobj{i}, 1, 'FaceColor','g','EdgeColor','k','LineWidth',1.5)
%     xlim([-0.5 bins_intensity-0.5])
%     ylim([0 1])
%     xlabel('Bin number [-]','Interpreter','Latex','Fontsize',12)
%     ylabel('Normalized intensity value [-]','Interpreter','Latex','Fontsize',12)
%     grid on
%
%     axes(ha(4));
%     bar(locs_hue, SF_Hue_Href, 1, 'FaceColor','b','EdgeColor','k','LineWidth',1.5)
%     xlim([-0.5 bins_hue-0.5])
%     ylim([0 1])
%     xlabel('Bin number [-]','Interpreter','Latex','Fontsize',12)
%     ylabel('Normalized hue value [-]','Interpreter','Latex','Fontsize',12)
%     grid on
%
%     axes(ha(5));
%     bar(locs_sat, SF_Saturation_Href, 1, 'FaceColor','r','EdgeColor','k','LineWidth',1.5)
%     xlim([-0.5 bins_saturation-0.5])
%     ylim([0 1])
%     xlabel('Bin number [-]','Interpreter','Latex','Fontsize',12)
%     ylabel('Normalized saturation value [-]','Interpreter','Latex','Fontsize',12)
%     title('HSI Href','Interpreter','Latex','Fontsize',12)
%     grid on
%
%     axes(ha(6));
%     bar(locs_int, F_Intensity_Href, 1, 'FaceColor','g','EdgeColor','k','LineWidth',1.5)
%     xlim([-0.5 bins_intensity-0.5])
%     ylim([0 1])
%     xlabel('Bin number [-]','Interpreter','Latex','Fontsize',12)
%     ylabel('Normalized intensity value [-]','Interpreter','Latex','Fontsize',12)
%     grid on
%
%     f = figure(300+i);
%     f.Position(1:4) = [300 200 1500 600];
%     ha = tight_subplot(1,6,[.03 .03],[.08 .05],[0.03 0.01]); %[gap_h gap_w] [low up] [lft rght]
%     axes(ha(1));
%     bar(locs_hue, SF_Hue_Hin{i}, 1, 'FaceColor','b','EdgeColor','k','LineWidth',1.5)
%     xlim([-0.5 bins_hue-0.5])
%     ylim([0 1])
%     xlabel('Bin number [-]','Interpreter','Latex','Fontsize',12)
%     ylabel('Normalized hue value [-]','Interpreter','Latex','Fontsize',12)
%     grid on
%
%     axes(ha(2));
%     bar(locs_sat, SF_Saturation_Hin{i}, 1, 'FaceColor','r','EdgeColor','k','LineWidth',1.5)
%     xlim([-0.5 bins_saturation-0.5])
%     ylim([0 1])
%     xlabel('Bin number [-]','Interpreter','Latex','Fontsize',12)
%     ylabel('Normalized saturation value [-]','Interpreter','Latex','Fontsize',12)
%     title('HSI Hin','Interpreter','Latex','Fontsize',12)
%     grid on
%
%     axes(ha(3));
%     bar(locs_int, F_Intensity_Hin{i}, 1, 'FaceColor','g','EdgeColor','k','LineWidth',1.5)
%     xlim([-0.5 bins_intensity-0.5])
%     ylim([0 1])
%     xlabel('Bin number [-]','Interpreter','Latex','Fontsize',12)
%     ylabel('Normalized intensity value [-]','Interpreter','Latex','Fontsize',12)
%     grid on
%
%     axes(ha(4));
%     bar(locs_hue, SF_Hue_Hout, 1, 'FaceColor','b','EdgeColor','k','LineWidth',1.5)
%     xlim([-0.5 bins_hue-0.5])
%     ylim([0 1])
%     xlabel('Bin number [-]','Interpreter','Latex','Fontsize',12)
%     ylabel('Normalized hue value [-]','Interpreter','Latex','Fontsize',12)
%     grid on
%
%     axes(ha(5));
%     bar(locs_sat, SF_Saturation_Hout, 1, 'FaceColor','r','EdgeColor','k','LineWidth',1.5)
%     xlim([-0.5 bins_saturation-0.5])
%     ylim([0 1])
%     xlabel('Bin number [-]','Interpreter','Latex','Fontsize',12)
%     ylabel('Normalized saturation value [-]','Interpreter','Latex','Fontsize',12)
%     title('HSI Hout','Interpreter','Latex','Fontsize',12)
%     grid on
%
%     axes(ha(6));
%     bar(locs_int, F_Intensity_Hout, 1, 'FaceColor','g','EdgeColor','k','LineWidth',1.5)
%     xlim([-0.5 bins_intensity-0.5])
%     ylim([0 1])
%     xlabel('Bin number [-]','Interpreter','Latex','Fontsize',12)
%     ylabel('Normalized intensity value [-]','Interpreter','Latex','Fontsize',12)
%     grid on
% end