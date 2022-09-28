function [y,d]=map_to_arcl(edges,vertices,x)

% map all datapoints to latent variable which is obtained by mapping point
% to closest point on path 
% path is indexed continuously in [0,l(P)] where l(P) is the length of the path


[n,d]=size(x);
segments=zeros(d,2,size(edges,1)-1);

e=edges; segment=1; lengths=zeros(size(segments,3)+1,1);
i=find(sum(e)==2);i=i(1); %get an endpoint of path
j=find(e(i,:)>0);         % get neighbor

while segment <= size(segments,3)
  e(i,j)=0;e(j,i)=0;        % remove used edges
  segments(:,:,segment) = [vertices(:,i) vertices(:,j)];
  lengths(segment+1) = lengths(segment)+norm(vertices(:,i)-vertices(:,j));
  segment=segment+1;
  i=j;                      % find next segment
  j=find(e(i,:)>0);         % get neighbor
end % while

y=zeros(n,d+1); % labels in arc length (1-dim) + projected points (d-dim)
msqd=0;

dists = zeros(n,size(segments,3));
rest   = zeros(n,d+1,size(segments,3));
for i=1:size(segments,3)
  [d t p]=seg_dist1(segments(:,1,i),segments(:,2,i),x'); 
  dists(:,i)=d;
  rest(:,:,i)=[t p];
end
[d,vr]=min(dists,[],2);
for i=1:n
y(i,:) = rest(i,:,vr(i));
y(i,1)=y(i,1)+lengths(vr(i));
end





