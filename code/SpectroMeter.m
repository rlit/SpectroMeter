function [d,coeffs] = SpectroMeter(shape,u0,c,sampl_tris,sampl_targets)
% sampl_tris - (optional) indecis of trianges on which deat diffution will
%               be calculated, can accelerate calcualtion. 
% NOTE - if this input is used the QR decompisition need to be done again
%       (i.e. "grad_lbo_RQt")
%%
[nv,n_src ]= size(u0);
assert(size(shape.X,1) == nv)
nf = size(shape.TRIV,1);

do_sub_sample = 0~=exist('sampl_tris','var');
if do_sub_sample
    nSampTri = nnz(sampl_tris);
    if islogical(sampl_tris)
        sampl_tris = find(sampl_tris);sampl_tris=sampl_tris(:);
    end

    % get vertices participating in selected triangles
    sampl_verts = shape.TRIV(sampl_tris,:);
    sampl_verts = unique(sampl_verts(:));
    assert(size(shape.phi,2) * 1.05 <= numel(sampl_verts),'not enough samples');
else
    sampl_tris = (1:nf)';
    nSampTri = nf;
    sampl_verts = (1:nv)';
end
sampl_tris3 = [sampl_tris ; sampl_tris + nf  ; sampl_tris + 2*nf ];


shape_lbo = shape.phi(sampl_verts,:);
GradOp = shape.GradOp(sampl_tris3,sampl_verts);


%%
if ~exist('c','var')
    c = 8.3e-3;
end
shape_area = sum(shape.tri_area);
timeVal    = c * shape_area;
expVal     = exp(shape.lambda(1:end) * timeVal);

hks_samp = -shape_lbo * bsxfun(@times, (u0' * shape.phi)' , expVal) * shape_area;

%% apply grad on HKS values
grad_hks = GradOp * hks_samp;


%% normalize to unit length
grad_hks = reshape(grad_hks,nSampTri,3,n_src);
hks_grad_magnitude = sqrt(sum(grad_hks.^2,2));
assert(all(hks_grad_magnitude(:)>0),'some zero magnitude gradients')
grad_hks_nrm = bsxfun(@rdivide,grad_hks,hks_grad_magnitude+eps());
grad_hks_nrm = reshape(grad_hks_nrm,nSampTri*3,n_src);

%%
% gradient of LBO
if isfield(shape,'grad_lbo')
    grad_lbo = shape.grad_lbo(sampl_tris3,:);
else
    grad_lbo = GradOp * shape_lbo(:,2:end);
end

% use QR for pseudo inderse
if isfield(shape,'grad_lbo_RQt') % use precomputed QR decomposition
    RQt = shape.grad_lbo_RQt;
    assert(isequal(size(RQt,2),3*nSampTri),'precomputed QR size is invalid')
else
    [Q,R]=qr(grad_lbo,0);
    RQt = (R\Q');
end
coeffs = RQt*grad_hks_nrm;


new_min = min(shape.phi(any(u0,2),2:end) * coeffs,[],1);
coeffs = [-new_min/shape.phi(1) ; coeffs]';
if nargin > 4
d = shape.phi(sampl_targets,:) * coeffs';
else
d = shape.phi * coeffs';
end
