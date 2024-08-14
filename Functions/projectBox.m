function [bb_x bb_y] = projectBox(AH_B,box,K,color,doplot)

    for j = 1:length(box.vertices)
        vertices_aug(:,j) = [AH_B(1:3,4)+AH_B(1:3,1:3)*box.vertices(:,j); 1];
        pixel_coordinates(:,j) = K*(vertices_aug(1:3,j)./vertices_aug(3,j));
    end

    if doplot
        for k = 1:length(box.edges)
            two_vertices=[pixel_coordinates(1:2,box.edges(1,k)),pixel_coordinates(1:2,box.edges(2,k))];
            line(two_vertices(1,:),two_vertices(2,:),'color',color,'LineWidth',1.5);
        end
    end
    
    %Bounding box of the projection
    bb_x = max(pixel_coordinates(1,:))-min(pixel_coordinates(1,:));
    bb_y = max(pixel_coordinates(2,:))-min(pixel_coordinates(2,:));
end