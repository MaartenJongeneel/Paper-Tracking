function [Z] = PEfun(bb_centers,AH_Best,img,K,Pixel_coordinates,box)
%% Description
% This script peforms the pose estimation. It takes as input the current
% image, the current bounding box, and the estimate of previous output. It
% then uses the information of the size of the bounding box to determine
% the distance of the box from the camera. After that, it samples particles
% with random positions and orientations around the expected position.
% Using the likelihood computation, it pops out a pose estimate based on a
% weighted average of the particles.

%% Total pose estimation algorithm
%Settings
Npart = 10000;
sigma = 0.2; %0.3

%Obtain the vector for the center of the object
%Vector to bounding box
vec = bb_centers';

%Project vector to meters
vec = inv(K)*[vec;1];

%Bounding box dimensions
boxdim = [max(Pixel_coordinates(1,:))-min(Pixel_coordinates(1,:)); max(Pixel_coordinates(2,:))-min(Pixel_coordinates(2,:))];

%Minimum distance based on fitting in the detected bounding box
for ibb = 1:0.01:3
    [bb_x,bb_y] = projectBox([eye(3) vec*ibb; zeros(1,3),1],box,K,[1,0,0],false);
    if bb_x < boxdim(1) && bb_y < boxdim(2)
        mindis = ibb;
        break;
    end
end

%Maximum distance based on fitting in the detected bounding box
for ibb = 1:0.01:3
    [bb_x bb_y] = projectBox([Rx(45)*Ry(45)*Rz(45) vec*ibb; zeros(1,3),1],box,K,[1,0,0],false);
    if bb_x < boxdim(1) && bb_y < boxdim(2)
        maxdis = ibb;
        break;
    end
end

%So the box position is somewhere along the line of
posvec = [vec*mindis vec*maxdis];

% Now we have an indication, we're gonna sample random particles
dit = AH_Best;
posvecext = [posvec dit(1:3,4)];
for ii = 1:Npart
    randrot(:,:,ii) = randn(3,1);
    dirvec = (posvec(:,2)-posvec(:,1));

    p1 = dirvec;
    p1 = p1/norm(p1);
    p2 = [0;0;1];
    randpos(:,ii) = (posvec(:,1)+0.25*dirvec)+expm(hat([hat(p2)*p1]))*diag([0.02; 0.02; 0.10])*randn(3,1);

    R = AH_Best(1:3,1:3)*expm(hat(sigma*randrot(:,:,ii)));
%     R = expm(hat(randrot(:,:,ii)));
    H(:,:,ii) = [R randpos(:,ii); zeros(1,3),1];
end
% figure;plot3(posvec(1,:),posvec(2,:),posvec(3,:)); hold on; plot3(posvecext(1,:),posvecext(2,:),posvecext(3,:),'o');
% axis equal; grid on;
% axis([-0.2 -0.06 -0.05 0.05 1.4 2.6]);
% hold on; plot3(randpos(1,:),randpos(2,:),randpos(3,:),'x');axis equal; grid on;
% axis([-0.2 -0.06 -0.05 0.05 1.4 2.6]);
% pause();

% Now we're gonna compute the likelihood for each of them
Href = HIS(imread('RefImage.png'),12,12,12);
hsi = rgb2hsi(img,12,12,12);

for ii = 1:length(H)
    state{1,1} = H(1:3,1:3,ii);
    state{2,1} = H(1:3,4,ii);
    H_deze(:,:,ii) = [state{1,1} state{2,1}; zeros(1,3),1];
    [L(ii),~]=likelihood(state,Href,hsi,K,box,eye(4));
end


% Compute weighted mean
%Compute the weighted average of the initial frame
L=L./sum(L);             %Normalize the weights
[~,indx] = max(L);         %Index of the particle with highest weight
wmean_1 = H(:,:,indx);       %Take particle with highest weight as initial mean

% Map the other particles to a tangent space at wmean
for ii = 1:Npart
    XSampled(:,ii) = vee(logm(wmean_1\H(:,:,ii)));
end

%Compute the mean in the tangent space, check if it is at the origin
Wmean = XSampled*L';
if norm(Wmean) > 0.01 %If not at the origin, an update is executed
    wmean_1 = wmean_1*expm(hat(Wmean)); %New mean on Lie group
end

% figure;plot3(posvec(1,:),posvec(2,:),posvec(3,:)); hold on; plot3(posvecext(1,:),posvecext(2,:),posvecext(3,:),'o');
% axis equal; grid on; axis([-0.2 -0.06 -0.05 0.05 1.4 2.6]);
% hold on; plot3(wmean_1(1,4),wmean_1(2,4),wmean_1(3,4),'x');axis equal; grid on; axis([-0.2 -0.06 -0.05 0.05 1.4 2.6]);

% Now we have the position, we're gonna refine the orientation
for ii = 1:2*Npart
    R = AH_Best(1:3,1:3)*expm(hat(sigma*randn(3,1)));
%     R = expm(hat(sigma*randn(3,1)));
    H(:,:,ii) = [R wmean_1(1:3,4); zeros(1,3),1];
end
%     figure;plot3(posvecext(1,:),posvecext(2,:),posvecext(3,:)); hold on; plot3(posvecext(1,:),posvecext(2,:),posvecext(3,:),'o');
%     hold on; plot3(squeeze(H(1,4,:)),squeeze(H(2,4,:)),squeeze(H(3,4,:)),'x');axis equal; grid on;

%Maximum distance based on fitting in the detected bounding box
for ii=1:length(H)
    [bb_x, bb_y] = projectBox(H(:,:,ii),box,K,[1,0,0],false);
    if bb_x < boxdim(1)+5 && bb_y < boxdim(2)+5
        idx(ii) = true;
    else
        idx(ii) = false;
    end
end

clear H_deze XSampled L_rot
%And compute the likelihood
tel = 1;
for ii = 1:length(H)
    if ~idx(ii)
        continue;
    else
        state{1,1} = H(1:3,1:3,ii);
        state{2,1} = H(1:3,4,ii);
        H_deze(:,:,tel) = [state{1,1} state{2,1}; zeros(1,3),1];
        [L_rot(tel),~]=likelihood(state,Href,hsi,K,box,eye(4));
        tel = tel+1;
    end
end

% XX Compute weighted mean
% Order them based on weight, then take top 100
Npart_L = 10;
[a,b] = sort(L_rot,'descend');
L = a./sum(a);
H_sort_1 = H_deze(:,:,b);

L = L(1:Npart_L);
H_sort(:,:,:) = H_sort_1(:,:,1:Npart_L);

%Compute the weighted average of the initial frame
%     L=L_rot./sum(L_rot);       %Normalize the weights
%     [~,indx] = max(L);         %Index of the particle with highest weight
wmean = H_sort(:,:,1);       %Take particle with highest weight as initial mean

% Map the other particles to a tangent space at wmean
XSampled = zeros(6,Npart_L);
for ii = 1:Npart_L
    XSampled(:,ii) = vee(logm(wmean\H_sort(:,:,ii)));
end

%Compute the mean in the tangent space, check if it is at the origin
Wmean = XSampled*L';
while norm(Wmean) > 0.01 %If not at the origin, an update is executed
    wmean = wmean*expm(hat(Wmean)); %New mean on Lie group
    for ii = 1:Npart_L
        XSampled(:,ii) = vee(logm(wmean\H_sort(:,:,ii)));
    end
    Wmean = XSampled*L';
end
% end
Z = wmean;
end