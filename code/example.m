function example

pthCode = fileparts(mfilename('fullpath'));

pthData = fullfile(fileparts(pthCode),'data');

%% load shape and run eigen decomposition
shape = load(fullfile(pthData,'cat1.mat'));
shape = shape.surface;

shape.nv = numel(shape.X);
shape.nf = size(shape.TRIV,1);

shape.VERT = [shape.X shape.Y shape.Z];
[shape.M,shape.S] = laplacian(shape.VERT,shape.TRIV);
shape.M = sparse(1:shape.nv,1:shape.nv,sum(shape.M));


K = 250;
tLBO = tic;
[shape.phi,shape.lambda] = eigs(-shape.S,shape.M,K,1e-5,struct('v0',-ones(shape.nv,1)/shape.nv/1e3));
tLBO = toc(tLBO);
shape.lambda = diag(shape.lambda);


tOps = tic;
[shape.GradOp,shape.DivOp,shape.grad_lbo,shape.grad_lbo_RQt] = GetGradDivOp(shape);
tOps = toc(tOps);


%% "randomize" 5 source points
rng(0)
nSrc = 5;
x0 = randsample(shape.nv,nSrc);

u0 = sparse(x0,1:nSrc,ones(nSrc,1),shape.nv,nSrc);

%% run heat method (note I use non-deault time value)

warning off MATLAB:nearlySingularMatrix
tHeat = tic;
shape.edge_len = EdgeLengthStats(shape);
d1 = HeatMethod(shape,u0,10);
tHeat = toc(tHeat);
warning on MATLAB:nearlySingularMatrix


%% run SpectroMeter (note I use non-deault time value)
shape.tri_area = calc_tri_areas(shape);
tSpectral = tic;
d2 = SpectroMeter(shape,u0,20.3e-3);
tSpectral = toc(tSpectral);

%% run SpectroMeter "SUBLINEAR"
% run FPS
sampFactor  = 1.1;
nSampVert   = ceil(K * sampFactor);
tFps        = tic;
tmp_perm    = FPS(shape, nSampVert, 1);
tFps        = toc(tFps);
sample_tris = any(ismember(shape.TRIV,tmp_perm(1:nSampVert)),2);

% re-calculate QR pseudo inverse
shape_tmp = shape;
[shape_tmp.Q_,shape_tmp.R_]=qr(shape.grad_lbo(repmat(sample_tris,3,1),:),0);
shape_tmp.grad_lbo_RQt = (shape_tmp.R_\shape_tmp.Q_');

tSpectral2 = tic;
d3 = SpectroMeter(shape_tmp,u0,20.3e-3,sample_tris);
tSpectral2 = toc(tSpectral2);

%% print times

fprintf('---------------------------------------------\n')
fprintf('LBO \t\t\t\t\t took %.1e seconds \n',tLBO)
fprintf('ops \t\t\t\t\t took %.1e seconds \n',tOps)
fprintf('Heat method\t\t\t\t took %.1e seconds \n',tHeat)
fprintf('SpectroMeter \t\t\t took %.1e seconds \n',tSpectral)
fprintf('FPS \t\t\t\t\t took %.1e seconds \n',tFps)
fprintf('SpectroMeter sublinear\t took %.1e seconds \n',tSpectral2)
fprintf('---------------------------------------------\n')

%% plot distmap and "iso-curves"
DistFun = {@(x)sin(.3*x),@(x)x};

for p = 1:2
    figure(666*p);clf
    
    pFun = DistFun{p};
    
    for ii = 1:nSrc
        subplot(3,nSrc,ii+nSrc*0);hold on
        PlotDistMap(shape,pFun(d1(:,ii)),x0(ii));
        subplot(3,nSrc,ii+nSrc*1);hold on
        PlotDistMap(shape,pFun(d2(:,ii)),x0(ii));
            subplot(3,nSrc,ii+nSrc*2);hold on
            PlotDistMap(shape,pFun(d3(:,ii)),x0(ii));
    end
end




end

function el = EdgeLengthStats(shape)
%%
edges = [
    shape.TRIV(:,[1 2])
    shape.TRIV(:,[1 3])
    shape.TRIV(:,[2 3])    ];
edges = unique([edges ; fliplr(edges)],'rows');

vert = [shape.X shape.Y shape.Z];
edge_len = sqrt(sum((...
    vert(edges(:,1),:) - ...
    vert(edges(:,2),:)).^2,2));

el.avg = mean(  edge_len);
el.min = min(   edge_len);
el.max = max(   edge_len);
el.med = median(edge_len);
end


function S_tri = calc_tri_areas(M)
%%
getDiff  = @(a,b)M.VERT(M.TRIV(:,a),:) - M.VERT(M.TRIV(:,b),:);
getTriArea  = @(X,Y).5*sqrt(sum(cross(X,Y).^2,2));
S_tri = getTriArea(getDiff(1,2),getDiff(1,3));


end

function h = PlotDistMap(shape,fun,x0)
%%
scatter3(shape.X(x0),shape.Y(x0),shape.Z(x0),20,'m','filled')

h = trisurf(shape.TRIV,shape.X,shape.Y,shape.Z,fun);
axis equal off
shading interp

view(3)
camlight head
lighting gouraud



end

function [perm] = FPS(shape, nSamp, i0)

d        = Inf(shape.nv, 1);
perm     = zeros(nSamp,1);
idx_next = i0;
warning off MATLAB:nearlySingularMatrix

for ii = 1:nSamp

    % Compute distance map from sample on the shape.
    u0 = zeros(shape.nv, 1);
    u0(idx_next) = 1;    
    perm(ii) = idx_next;
    
    cur_dist = HeatMethod(shape,u0,10);

    d = min(d, cur_dist);
    [~, idx_next] = max(d);
    assert(~ismember(idx_next,perm(1:ii)) ,'FPS failed')
end
warning on MATLAB:nearlySingularMatrix

end
