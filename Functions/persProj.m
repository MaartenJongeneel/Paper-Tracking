function uv=persProj(xyz,K,AH_M)
% This function computes the 2D points in the image plane from the 3D world
% coordinates xyz. (Perspective Projection)
%
% INPUTS:    xyz       : [x y z] or [x; y; z] coordinates in the 3D world
%            K         : Intrinsic camera parameter matrix
%
% OUTPUTS:   uv        : [u;v] coordinates of the points in the image plane
%% Compute the points
xyz_aug = [xyz; ones(1,length(xyz(1,:)))];

xyz_A = AH_M*xyz_aug;
xyz_A = xyz_A(1:3,:);

uv=K*(xyz_A./xyz_A(3,:));  % uv_point=K*(xyz_point/z);
