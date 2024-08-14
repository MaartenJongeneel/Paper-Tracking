function [L,S1,S2,S3,Hin,Hf,Hout,see_face,pin,pface,pout]=likelihood(state,Href,hsi,K,box,AH_M)
% Computes the likelihood 
% INPUTS:    state        : cell(R;o;v;w): state of the particle 
%            Href         : Cell aray with reference histogram of each face
%            hsi          : HSI image of the current frame
%            K            : Intrinsic camera matrix (3x3)
%            box          : Struct containing geometric properties of the
%                           box
%
% OUTPUTS:   L            : Likelihood 
%            S1           : Similarity 1
%            S2           : Similarity 2
%            S3           : Similarity 3
%% Compute the likelihood
%Compute the points on the surface of the cuboid and outside the cuboid
[pin,pface,pout,ed1,see_face] = compute_points(box,state,K,AH_M);
N = length(see_face);

nr_bins = size(Href);
%Compute histograms for both sets of points
for ii = 1:N
Hin{see_face(ii)}  = Hpoints(pin{see_face(ii)},hsi,nr_bins); 
Hf{see_face(ii)} = Hpoints(pface{see_face(ii)},hsi,nr_bins);
end

%Weighting parameters for the likelihood function
k1=1; k2=2; k3=7; b=0.1;

[Hout] = Hpoints(pout,hsi,nr_bins); 

%Compute the likelihood based on Bhattacharyya distance between histograms.
S1(1:3)=0;
S2(1:3)=0;
S3(1:3)=0;
for ii = 1:N
    for c1=1:nr_bins(1)
        for c2=1:nr_bins(2)
            for c3=1:nr_bins(3)
                S1(ii)=S1(ii)+ sqrt(Hin{see_face(ii)}(c1,c2,c3)*Href(c1,c2,c3)); %Similarity contour inside points and reference
                S2(ii)=S2(ii)+ sqrt(Hf{see_face(ii)}(c1,c2,c3)*Href(c1,c2,c3));  %Similarity face points and reference
                S3(ii)=S3(ii)+ sqrt(Hin{see_face(ii)}(c1,c2,c3)*Hout(c1,c2,c3)); %Similarity contour inside and contour outside points               
            end
        end
    end
end
S1 = sum(S1(:))/N;
S2 = sum(S2(:))/N;
S3 = sum(S3(:))/N;
%Specify the distance function
D = (k1*(1-S1)+k2*(1-S2)+k3*S3)/(k1+k2+k3); 
%Likelihood function
L = exp(-(abs(D))/b);
