%% Camera calibration
% This script performs the extrinsic camera calibration to find the
% relative transformation matrix between the camera sensor frame A and the
% camera casing frame G. It uses the intrinsics obtained via an intrinsic
% calibration and uses the photo's and reconstructed poses in the
% "extrinsics" folder (obtained via VISP library). This script reads out
% the poses and computes the relative transformation. It is mainly based on
% the following two works:
% 
% [1] F. C. Park and B. J. Martin, "Robot sensor calibration: solving AX=XB
% on the Euclidean group," in IEEE Transactions on Robotics and Automation,
% vol. 10, no. 5, pp. 717-721, Oct. 1994, doi: 10.1109/70.326576.
%
% [2] R. Y. Tsai and R. K. Lenz, "A new technique for fully autonomous and 
% efficient 3D robotics hand/eye calibration," in IEEE Transactions on 
% Robotics and Automation, vol. 5, no. 3, pp. 345-358, June 1989,
% doi: 10.1109/70.34770.

%% Load the data
%Load mocap data
files = dir('calibratecam/extrinsics/*.yaml');      
nfiles = length(files);    % Number of files found
cnt_c = 1; cnt_m = 1;      % Initialize counters
for ii=1:nfiles
   currentfilename = files(ii).name;
   data = readyaml(currentfilename);
   if contains(currentfilename,'cPo')       
       AH_B(:,:,cnt_c) = [expm(hat(data.data(4:6,1))) data.data(1:3,1); zeros(1,3) 1];
       cnt_c = cnt_c+1;
   elseif contains(currentfilename,'fPe')
       MH_G(:,:,cnt_m) = [expm(hat(data.data(4:6,1))) data.data(1:3,1); zeros(1,3) 1];
       cnt_m = cnt_m+1;
   else
       continue;
   end
end

%% The following is based on [1]. See that paper for details
%First create the matrices A and B
tel = 1;
for ii=1:size(MH_G,3)
    for jj=ii+1:size(MH_G,3)
        B(:,:,tel)=inv(inv(MH_G(:,:,jj))*MH_G(:,:,ii));
        A(:,:,tel)=AH_B(:,:,ii)*inv(AH_B(:,:,jj));
        tel = tel+1;
    end
end

% Now find the rotation matrix
M=zeros(3,3);
for ii = 1:length(B)
    M = M + logm(B(1:3,1:3,ii))*logm(A(1:3,1:3,ii))';
end
AR_G = (M'*M)^(-0.5)*M';

% Using the found rotation matrix, we find the position using LS
C = eye(3)-A(1:3,1:3,1);
d = A(1:3,4,1)-AR_G*B(1:3,4,1);
for ii = 2:length(B)
    C = [C; eye(3)-A(1:3,1:3,ii)];
    d = [d; A(1:3,4,ii)-AR_G*B(1:3,4,ii);];
end
Ao_G = (C'*C)\C'*d;

% Which combined gives us the final transformation matrix
GH_A = inv([AR_G Ao_G; zeros(1,3),1]);

%% Optimization to find the relative transformation
% Alternatively, one could uncomment the code below, which computes the
% relative transformation using MATLAB's lsqnonlin function

% R_init = [1 0 0; 0 1 0; 0 0 1];
% Theta0 = zeros(6,1);%[0 0 -0.01 rotm2eul(R_init,'XYZ')]';
% fun = @(x)MyCalibCostFun(x, MH_G, AH_B);
% Theta = lsqnonlin(fun,Theta0);
% GH_A = [eul2rotm(Theta(4:6)','XYZ'),Theta(1:3); 0 0 0 1] ;

% function F = MyCalibCostFun(vars, MH_G, AH_B)
% F = [];
% 
% GH_A = [eul2rotm([vars(4),vars(5),vars(6)], 'XYZ') , vars(1:3); 0 0 0 1];
% for ii=1:size(MH_G,3)
%     for jj=ii+1:size(MH_G,3)
%         H = inv(MH_G(:,:,jj))*MH_G(:,:,ii)*GH_A*AH_B(:,:,ii)*inv(AH_B(:,:,jj))*inv(GH_A);
%         F = [F ; H(1:3,4) ; -rotationMatrixToVector(H(1:3,1:3))' ];
%     end
% end
% 
% end

%% 
fid = fopen('video.mp4','w');
fwrite(fid,double(data.Rec_20231128T120603Z.SENSOR_MEASUREMENT.Reference_camera.datalog.ds));
fclose(fid);

%%

videoObject = VideoReader('video.mp4');
% Determine how many frames there are.
numberOfFrames = videoObject.NumFrames;

% Read and write each frame.
for frame = 1 : numberOfFrames
  thisFrame = read(videoObject, frame); 
  imwrite(thisFrame,sprintf('Images/Rec_20231128T120603Z/%03d.jpg',frame))
end

