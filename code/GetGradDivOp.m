function [GradMat,DivMat,grad_lbo,grad_lbo_RQt] = GetGradDivOp(shape)

nv = numel(shape.X);
nf = size(shape.TRIV,1);

%%
vert = [shape.X shape.Y shape.Z];
TriDiffFun  = @(a,b)vert(shape.TRIV(:,a),:) - vert(shape.TRIV(:,b),:);

Diff12 = TriDiffFun(1,2);
Diff23 = TriDiffFun(2,3);
Diff31 = TriDiffFun(3,1);

tri_normals = cross(Diff12,-Diff31,2);
tri_area    = .5*sqrt(sum(tri_normals.^2,2));
tri_normals = bsxfun(@rdivide,tri_normals, 2*tri_area);% to unit-length

% %% plot normals
% figure(356);clf;
% tmp = plotmesh(shape,0);hold on
% set(tmp,'EdgeColor','k')
% quiver3(...
%     mean(shape.X(shape.TRIV),2),...
%     mean(shape.Y(shape.TRIV),2),...
%     mean(shape.Z(shape.TRIV),2),...
%     tri_normals(:,1),tri_normals(:,2),tri_normals(:,3))
%% calc grad
GradValFun = @(d)bsxfun(@rdivide, cross(tri_normals,d),2*tri_area);
uVec = [
    GradValFun(Diff23);...  % will be multiplied with values of vertex #1
    GradValFun(Diff31);...  % will be multiplied with values of vertex #2
    GradValFun(Diff12)];    % will be multiplied with values of vertex #3
% % % sanity - sum over clsed loop (=triangle) should be zero
% norm(GradValFun(Diff23)+GradValFun(Diff31)+GradValFun(Diff12))

tmp_idx1 = repmat([(1:nf) ; (1:nf)+nf ; (1:nf)+2*nf]',3,1);
tmp_idx2 = repmat(shape.TRIV(:),1,3);

GradMat = sparse(tmp_idx1(:),tmp_idx2(:),uVec(:),3*nf,nv);

% %% sanity - make sure each triangle has 3 vertices
% tmp = reshape(sum(full(GradMat')~=0)',nf,3)
% tmp0=[];[tmp0(:,1),tmp0(:,2)] = find(tmp~=3);tmp0

% %% sanity2, multiplication by constant should be zero
% tmp = norm(GradMat * ones(nv,1))


%%


% calculate edge lengths
v12 = sqrt(sum(Diff12.^2,2));
v23 = sqrt(sum(Diff23.^2,2));
v31 = sqrt(sum(Diff31.^2,2));

CosFun = @(v1,v2,v0)...
    (v1.^2. + v2.^2 - v0.^2) ./ ...
    (2 * v1 .* v2);
cos1 = CosFun(v12,v31,v23);
cos2 = CosFun(v23,v12,v31);
cos3 = CosFun(v31,v23,v12);

cot1 = cos1 ./ sqrt(1-cos1.^2);
cot2 = cos2 ./ sqrt(1-cos2.^2);
cot3 = cos3 ./ sqrt(1-cos3.^2);

DivValFun = @(d,c)d(:).* repmat(c,3,1);
uVec = 0.5 * [
    DivValFun(-Diff12,cot3) + DivValFun( Diff31,cot2),...
    DivValFun( Diff12,cot3) + DivValFun(-Diff23,cot1),...
    DivValFun( Diff23,cot1) + DivValFun(-Diff31,cot2),...
    ];


tmp_idx1 = repmat((1:3*nf)',1,3);
tmp_idx2 = repmat(shape.TRIV,3,1);

DivMat = sparse(tmp_idx2(:),tmp_idx1(:),uVec(:),nv,3*nf);

% %% sanity - make sure each triangle has 3 vertices
% tmp = reshape(sum(full(DivMat)~=0)',nf,3)
% tmp0=[];[tmp0(:,1),tmp0(:,2)] = find(tmp~=3);tmp0

%% optional - calculate the the lbo gradient and it's pseudo inverse
if nargout < 3
    return
end
grad_lbo = GradMat * shape.phi(:,2:end);

if nargout < 4
    return
end
[Q,R] = qr(grad_lbo,0);
tol = abs(R(1)) * nf * eps(class(R));
xrank = sum(abs(diag(R)) > tol);
assert(xrank==size(grad_lbo,2));
RQt = R \ Q';
grad_lbo_RQt = RQt;