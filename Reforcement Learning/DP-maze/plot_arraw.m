function newimg = plot_arraw(img,i,j,dire,num)

x1 = (i-1)*num + num / 2;
y1 = (j-1)*num + num / 2;
newimg = img;
if strcmp(dire,'left');
    x2 = x1;
    y2 = y1 - num/2;
    coor_y = y2:y1;
    n = length(coor_y);
    coor_x = ones(1,n)*x1;
    newimg(x1-1,y2+1) = 255;
    newimg(x1+1,y2+1) = 255;
elseif strcmp(dire,'up');
    x2 = x1 - num/2;
    y2 = y1;
    coor_x = x2:x1;
    n = length(coor_x);
    coor_y = ones(1,n)*y1;
    newimg(x2+1,y2-1) = 255;
    newimg(x2+1,y2+1) = 255;
elseif strcmp(dire,'right');
    x2 = x1;
    y2 = y1 + num/2;
    coor_y = y1:y2;
    n = length(coor_y);
    coor_x = ones(1,n)*x1;
    newimg(x2-1,y2-1) = 255;
    newimg(x2+1,y2-1) = 255;
elseif strcmp(dire,'down');
    x2 = x1 + num/2;
    y2 = y1;
    coor_x = x1:x2;
    n = length(coor_x);
    coor_y = ones(1,n)*y1;
    newimg(x2-1,y2-1) = 255;
    newimg(x2-1,y2+1) = 255;
end
newimg(coor_x,coor_y) = 255;
end