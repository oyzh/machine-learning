N = 4;
M = 4;
Width = 20;
Maz = zeros([N,M]);
endPoints = [1,1;4,4];
policy = maze(Maz, endPoints);


img = zeros([N*Width,M*Width]);
for i=1:N
    for j=1:M
        
        if policy(i,j,1)
           img = plot_arraw( img,i,j,'left',Width);
        end
        if policy(i,j,2)
           img = plot_arraw( img ,i,j,'up',Width);
        end
        if policy(i,j,3)
           img = plot_arraw(img,i,j,'right',Width);
        end
        if policy(i,j,4)
           img = plot_arraw(img, i,j,'down',Width); 
        end
    end
end
for i=1:(N-1)
    for j=1:(M-1)
        x = i*Width;
        y = j*Width;
        k1 = 1:(Width*M);
        k2 = 1:(Width*N);
        
        k1 = k1(find(mod(k1,2) == 1));
        k2 = k2(find(mod(k2,2) == 1));
        img(x,k1) = 255;
        img(k2,y) = 255;
    end
end

img = imresize(img, 5);
imshow(img);
