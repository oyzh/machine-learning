function policy = maze(Maz, endPoints)
% get the action of each point
% input
% Maz: N*M matrix, 0 means path, 1 means wall.
% outPoints: N*2 matrix, each row means a end point
% output
% policy: for each path point im Maz, it has four values, each meas the possibal of the action of left,up,right,and down.

[N,M] = size(Maz);
newMaz = zeros(N+2, M+2); % we add a wall to round the maz
newMaz(:) = 1;
newMaz(2:(N+1), 2:(M+1)) = Maz;
v = zeros(N+2, M+2);
policy = zeros(N+2,M+2,4);
policy(:) = 0.25;
%policy(:,:,1) = 1; % init random
endPoints(:) = endPoints(:) + 1;
[x,y] = find(ones(N,M));
index = [x,y];
index(:) = index(:) + 1;
index = setdiff(index, endPoints, 'rows');
index_n = size(index,1);

while 1
   % caculate the value of the policy
   v = zeros(N+2, M+2);
   while 1
   new_v = v;
   for k=1:index_n
      i = index(k,1);
      j = index(k,2);
      new_v(i,j) = -1 + v(i,j-1)*policy(i,j,1) + v(i-1,j)*policy(i,j,2) + v(i,j+1)*policy(i,j,3)+v(i+1,j)*policy(i,j,4);
   end
   if abs(sum(sum(new_v-v))) < 0.00000001
        break;
   end
   v = new_v;
   v(:,1) = v(:,2);
   v(:,M+2) = v(:,M+1);
   v(1,:) = v(2,:);
   v(N+2,:) = v(N+1,:);
   end
   
   % get new policy
   new_policy = policy;
   for k=1:index_n
       i = index(k,1);
       j = index(k,2);
       new_policy(i,j,:) = get_possibal([v(i,j-1), v(i-1,j), v(i,j+1), v(i+1,j)]);
   end
   % if new policy equal to old policy, then return;
   if new_policy == policy
       n = size(endPoints,1);
       for l=1:n
           policy(endPoints(l,1),endPoints(l,2),:) = 0;
       end
 %      policy(endPoints(:,1),endPoints(:,2)) = 0;
       policy = policy(2:N+1,2:M+1,:);
              
       return;
   end
   policy = new_policy;
end



function possibal = get_possibal(four_v)
    max_v = max(four_v);
    max_index = find(abs(four_v-max_v) < 0.00000001);
    max_n = length(max_index);
    possibal = zeros(1,4);
    possibal(max_index) = 1 / max_n;
end

end
