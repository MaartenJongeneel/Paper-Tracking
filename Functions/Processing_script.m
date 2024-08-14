function [GT,Z,K,AH_M,Bomg_MB,Bv_MB,sel_img,box,impacts,plotdata,bounding_boxes,Mocap] = Processing_script(object,vp,wrtimg,doPlot)
%% Processing Script
% This script takes the HDF5 data file and extracts from this file all data
% needed to run the rest of the scripts. Examples are taking the ground
% truth data for error computation and obtaining images from videos.
% --- INPUTS --- %
%   Object       [string] "Box006", "Box007", or "Box009" select box
%   viewpoint    [string] "VP1" or "VP2", select viewpoint
%   wrtimg       [bool]   Determine if you want to write the video images
%                         to a folder
%   doPlot       [bool]   Determine if you want to show the projection of
%                         the mocap data on the image
%
% --- OUTPUTS --- %
%
%% Read h5 file
try
%     data = readH5('230504_Archive_025_ImpactAwareObjectTracking.h5');    %This one has old data..
%     data = readH5('231018_ObjectTracking.h5');                           %This one is good, but authors are incorrect
%     data = readH5('230821_Archive_025_ImpactAwareObjectTracking.h5');    %This one does not have cropped images..
%     data = readH5('230809_Archive_025_ImpactAwareObjectTracking.h5');    %This one is missing the timevector! (because it was not created on Linux PC..)
data = readH5('231106_ObjectTracking.h5');
catch
    error("Cannot read the dataset. Please make sure the HDF5 file '230504_Archive_025_ImpactAwareObjectTracking.h5' is placed in the 'Data' folder and all functions are added to the path.")
end

%% Data selection
%Selecting images and mocap datapoints
%img_vec are the indices of RGB images from the moment the box is released
%until it is at rest on the conveyor. This is done manually
if  vp == "VP1" && object == "Box006"
    viewpoint = "Viewpoint 1";
    img_vec = 781:840;   
    impacts = [11 13 15 19 27 35];
elseif vp == "VP1" && object == "Box007"
    viewpoint = "Viewpoint 1";
    img_vec = 734:793;
    impacts = [12 14 22 27 33]; 
elseif vp == "VP1" && object == "Box009"
    viewpoint = "Viewpoint 1";
    img_vec = 669:731; %img_vec = 667:731;   
    impacts = [12 14 22 27 33]; 
elseif vp == "VP2" && object == "Box006"
    viewpoint = "Viewpoint 2";
    img_vec = 731:790; 
    impacts = [12 14 22 27 33]; 
elseif vp == "VP2" && object == "Box007"
    viewpoint = "Viewpoint 2";
    img_vec = 728:787; 
    impacts = [12 14 22 27 33]; 
elseif vp == "VP2" && object == "Box009"
    viewpoint = "Viewpoint 2";
    img_vec = 665:744;
    impacts = [12 14 22 27 33]; 
end

%We now select the data that is chosen for given object and viewpoint
fn = fieldnames(data);
fn = fn(3:end);
for i = 1:length(fn)
    if contains(data.(fn{i}).attr.note,object,'IgnoreCase',true) && contains(data.(fn{i}).attr.note,viewpoint,'IgnoreCase',true)
        idx(i) = 1;
    else 
        idx(i) = 0;
    end
end
index = find(idx==1);


%Create a box object that contains all info of the box used in experiments
boxdim = data.(fn{index}).OBJECT.(object).dimensions.ds;
box = create_box_model(boxdim(1),boxdim(2),boxdim(3));
box.B_M_B = data.(fn{index}).OBJECT.(object).inertia.ds;
box.mass = data.(fn{index}).OBJECT.(object).mass.ds;
        
%% Load the data
%OptiTrack data
Mocap = data.(fn{index}).SENSOR_MEASUREMENT.Mocap;
MH_B = data.(fn{index}).SENSOR_MEASUREMENT.Mocap.POSTPROCESSING.(object).transforms.ds;
MH_G = data.(fn{index}).SENSOR_MEASUREMENT.Mocap.POSTPROCESSING.RealSense001.transforms.ds;
MH_Db = data.(fn{index}).SENSOR_MEASUREMENT.Mocap.POSTPROCESSING.ArucoTracker001.transforms.ds;

%Obtain the images from the reference video and write them to a folder and struct
fid = fopen('video.mp4','w');
fwrite(fid,double(data.(fn{index}).SENSOR_MEASUREMENT.Reference_camera.datalog.ds));
fclose(fid);

v1 = VideoReader('video.mp4');
vid1Frames = read(v1);
if wrtimg; mkdir(append('Images/',fn{index},'/')); end %Make a directory to drop the images of the video
for ii = 1:length(vid1Frames(1,1,1,:))
    img{ii} = vid1Frames(:,:,:,ii); %Write all images to a struct
    if wrtimg; imwrite(img{ii},sprintf(append('Images/',fn{index},'/%.4d.jpg'),ii)); end %write the images to a folder
end
delete('video.mp4');

%Now we select the images that are of interest for object tracking (from release to rest)
for i = 1:length(img_vec)
    sel_img{i} = img{img_vec(i)};
end

%% Determining mean H matrix that relates camera casing frame (G) to the Motive frame (M)
vec = length(MH_G)-500:length(MH_G);
for i = 1:length(vec)
      MH_G_cell{1,i} = MH_G{vec(i)}(1:3,1:3);
      MH_G_cell{2,i} = MH_G{vec(i)}(1:3,4);
end

%Initial guess mean MH_G matrix
mean_MH_G = MH_G_cell(:,randi([1 length(MH_G_cell)])); 

%Mapping all points to a tangent space
for i = 1:length(vec)
    mapping(:,i) = logH(Hprod(invH(mean_MH_G),MH_G_cell(:,i)));
end

%Compute the mean in the tangent space, check if it is at the origin
mean_tangent = mean(mapping')';     
while norm(mean_tangent) > 1e-5 %If not at the origin, an update is executed
    mean_MH_G = Hprod(mean_MH_G,expH(mean_tangent)); %New mean on Lie group
    for i = 1:length(vec)
        %Map all points to a tangent space at wmean
        mapping(:,i) = logH(Hprod(invH(mean_MH_G),MH_G_cell(:,i)));
    end
    mean_tangent = mean(mapping')';
end

mean_MH_G = [mean_MH_G{1} mean_MH_G{2}; zeros(1,3) 1];

%% Determining H matrices that relate the frames of the boxes (B) to camera casing frame (G)
for i = 1:length(MH_G)
    GH_B{i,1} = inv(mean_MH_G)*MH_B{i,1};            
end

%Translation vectors
for i = 1:length(MH_G)
    trans_GH_B(1:3,i) = GH_B{i,1}(1:3,4);
end

%Rotation matrices
for i = 1:length(MH_G)
    GR_B{i,1} = GH_B{i,1}(1:3,1:3);
end

%% Determining relative transformation from casing frame (G) to camera frame (A)
% w = [-0.0243, 0.0291, -0.0059]; 
% w = [-0.0280, 0.0220, -0.02]; 
% w = [-0.0250 0.0250 0];
% t = [0.0, 0.0, 0.0];
% 
% AR_G = eye(3)+(sin(norm(w))/norm(w))*hat(w)+((1-cos(norm(w)))/(norm(w))^2)*hat(w)^2;   
% % AR_G = [    0.9994    0.0199    0.0269
% %    -0.0199    0.9998   -0.0020
% %    -0.0269    0.0015    0.9996]; %Obtained from intrinsic calibration
% % AR_G = expm(hat(flip(vee(logm(AR_G)))));
% AH_G = [AR_G t'; zeros(1,3) 1];
% AH_G = inv( ...
%     [0.999439300513744	-0.0198950282636966	-0.0269308826254227	 -0.0107964089701748
%      0.0199426374068983	 0.999800000049734	 0.00150037122356354  0.0126931353612391
%      0.0268956465223382	-0.00203660279343584 0.999636172038211	  0.0176445393347407
%      0	0	0	1]);

%With nonlinear LS
AH_G = inv( ...
    [0.9995   -0.0155   -0.0264    0.0002
     0.0151    0.9998   -0.0154   -0.0025
     0.0266    0.0150    0.9995    0.0315
     0         0         0         1.0000]);

% With linear LS
AH_G =inv([   0.999594768933156  -0.020370527963783  -0.019883146429738  -0.006394163483626
   0.020035598194003   0.999656402647903  -0.016901226288771  -0.001115807484294
   0.020220601536011   0.016496006654106   0.999659446530656   0.033640106855218
                   0                   0                   0   1.000000000000000]);

% AH_G = inv([    0.9995   -0.0167   -0.0277   -0.0074
%     0.0161    0.9997   -0.0205    0.0298
%     0.0281    0.0201    0.9994    0.0262
%          0         0         0    1.0000]);

%% Determining H matrix that relates the Motive frame (M) to the camera sensor frame (A)
AH_M = AH_G*inv(mean_MH_G);

%% Convert Aruco data into MH_D world frame data and align with mocap data
Aruco = table2array(data.(fn{index}).SENSOR_MEASUREMENT.ArucoData.datalog.ds);
Aruco(Aruco==0)=NaN;

for ii = 1:length(Aruco)
    AH_D(:,:,ii) = [expm(hat(Aruco(ii,4:6))) Aruco(ii,1:3)'; zeros(1,3) 1]*[eye(3) [0.03;0.015;0]; zeros(1,3), 1];
    MH_Da(:,:,ii) = AH_M\AH_D(:,:,ii); %MH_D from aruco data
end

for ii = 1:length(MH_Db); Mo_Db(:,:,ii) = MH_Db{ii}(1:3,4); end
Mo_Da = squeeze(MH_Da(1:3,4,:));

t_Aruco = data.(fn{index}).SENSOR_MEASUREMENT.Reference_camera.TimeVector.ds;
t_Mocap = table2array(data.(fn{index}).SENSOR_MEASUREMENT.Mocap.datalog.ds(:,2));

y_Aruco = Mo_Da(3,1:length(t_Aruco));
y_Mocap = Mo_Db(3,:);

%Create time vector to shift mocap points
t_vec = 0:0.001:1; %Shifting with 1ms at the time, Mocap as frequency of 360fps
for ii = 1:length(t_vec)
    tel = 1;
    for i = 1:length(t_Aruco)
        abs_dif = abs(t_Aruco(i)-(t_Mocap+t_vec(ii)));
        [~,indx_mocap(ii,tel)] = min(abs_dif);
        tel = tel+1; 
    end
    indx_con = abs(t_Aruco-(t_Mocap(indx_mocap(ii,:))+t_vec(ii)))<((1/360)/2) & ~isnan(y_Aruco)'; %indices to consider are the ones where we have Aruco data, and the time difference between selected frames is not bigger than half times 1/360 sec.
    Nerror(ii) = norm(y_Aruco(indx_con)-y_Mocap(indx_mocap(ii,indx_con)));
end

%Starting index: optimum value for t_s where mocap starts w.r.t. RGB
[~,indx_ts_opt] = min(Nerror);

%mocap points
mocap_points = indx_mocap(indx_ts_opt,:);

%Output for plotting data
plotdata.y_Aruco = y_Aruco;
plotdata.y_Mocap = y_Mocap;
plotdata.t_Aruco = t_Aruco;
plotdata.t_Mocap = t_Mocap;
plotdata.t_vec = t_vec;
plotdata.Nerror = Nerror;
plotdata.indx_ts_opt = indx_ts_opt;
plotdata.indx_con = indx_con;
plotdata.indx_mocap = indx_mocap;
%% Projecting contours to image plane  
%Retreive the Camera Intrinsic Matrix
% K = data.(fn{index}).SENSOR_MEASUREMENT.Reference_camera.IntrinsicMatrix.ds; 

%From RS camera
% K = [681.224456154926	0	0
% 0	686.409753755698	0
% 346.009537937216	284.659961220456	1]';

%With nonlinear LS
% K = [681.224456154926	0	0
% 0	686.409753755698	0
% 358	275	1]';

%With linear LS
K = [681.224456154926	0	0
0	686.409753755698	0
359	277	1]';

% K = [681.224456154926	0	0
% 0	686.409753755698	0
% 352	287	1]';

%Project the motion capture data onto the image plane
for i = 1:length(GH_B)
    for j = 1:length(box.vertices)
        vertices_B_wrt_G{i,1}(:,j) = trans_GH_B(1:3,i)+GR_B{i,1}*box.vertices(:,j); 
    end
    
    vertices_B_wrt_G_aug{i,1} = [vertices_B_wrt_G{i,1}; ones(1,8)];

    for j = 1:length(box.vertices)
        A_p_aug{i,1}(:,j) = AH_G*vertices_B_wrt_G_aug{i,1}(:,j);
        pixel_coordinates{i,1}(:,j) = K*(A_p_aug{i,1}(1:3, j)./A_p_aug{i,1}(3, j));
        u{i,1}(:,j) = pixel_coordinates{i,1}(1,j);
        v{i,1}(:,j) = pixel_coordinates{i,1}(2,j);
    end
end

%Create a figure to show the output
if doPlot
    figure;
    for i = 1:length(sel_img)    
        imshow(sel_img{i})
        hold on
        for k = 1:length(box.edges)
            two_vertices=[pixel_coordinates{mocap_points(1,img_vec(i)),1}(1:2,box.edges(1,k)), pixel_coordinates{mocap_points(1,img_vec(i)),1}(1:2,box.edges(2,k))];
            line(two_vertices(1,:),two_vertices(2,:),'color','r','LineWidth',1.5);
        end
        GT_pixels(:,:,i) = pixel_coordinates{mocap_points(1,img_vec(i)),1}(1:2,:);
        pause();
    end
end

%% Ground truth computation
for i = 1:length(img_vec)
   GT(1,i) = {MH_B{mocap_points(img_vec(i)),1}};
end

%% Initial velocity calculation (central difference)
delta_t = 1/360;

M_o_B_tk = MH_B{mocap_points(img_vec(1))}(1:3,4);
M_o_B_tkp1 = MH_B{mocap_points(img_vec(1))+1}(1:3,4);
M_o_B_tkm1 = MH_B{mocap_points(img_vec(1))-1}(1:3,4);
M_o_dot_B = (1/(delta_t*2))*(M_o_B_tkp1 - M_o_B_tkm1);

MR_B_tk = MH_B{mocap_points(img_vec(1))}(1:3,1:3);
MR_B_tkp1 = MH_B{mocap_points(img_vec(1))+1}(1:3,1:3);
MR_B_tkm1 = MH_B{mocap_points(img_vec(1))-1}(1:3,1:3);
B_omega_hat_M_B = (1/(delta_t*2))*(logm(inv(MR_B_tk)*MR_B_tkp1)-logm(inv(MR_B_tk)*MR_B_tkm1));
Bomg_MB = [B_omega_hat_M_B(3,2); B_omega_hat_M_B(1,3); B_omega_hat_M_B(2,1)];
Bv_MB = inv(MR_B_tk)*M_o_dot_B;

%% Bounding boxes from Single Shot Detector (SSD)
Pixel_coordinates = data.(fn{index}).SENSOR_MEASUREMENT.SSD_data.vertices.ds;
bounding_boxes.Pixel_coordinates = data.(fn{index}).SENSOR_MEASUREMENT.SSD_data.vertices.ds;
bounding_boxes.bb_centers = squeeze([Pixel_coordinates(1,1,:)+(Pixel_coordinates(1,2,:)-Pixel_coordinates(1,1,:))/2 Pixel_coordinates(2,2,:)+(Pixel_coordinates(2,3,:)-Pixel_coordinates(2,2,:))/2]);

%% Obtain a measurement
%Set the covariance matrix for rotation and position of measurement Z
% PR = 1e-5*diag([4910 6050 5980]);    Po = 1e-5*diag([11.196 92.553 4.1512]);
PR = 1e-5*diag([1000 1000 1000]);    Po = 1e-5*diag([90 120 90]);

%Run through the frames and add the noise to the ground truth
for i = 1:length(GT)
    MR_B = GT{i}(1:3,1:3);
    Mo_B = GT{i}(1:3,4);
    xiR = zeros(3,1) + sqrtm(PR)*randn(3,1);
    xio = zeros(3,1) + sqrtm(Po)*randn(3,1);
    Z{1,i}  = MR_B*expm(hat(xiR));
    Z{2,i}  = Mo_B+xio;
end
end