function d = HeatMethod(shape,u0,c)

if ~exist('c','var')
    c=1;
end
nf = size(shape.TRIV,1);
[nv,n_src ]= size(u0);
assert(size(shape.X,1) == nv)
%% solve heat
t = c*shape.edge_len.avg^2;
ut = (shape.M + t*shape.S) \ full(u0);


%% gradient
grad = shape.GradOp * ut;

%% normalize
grad3 = reshape(grad,nf,3,n_src);
grad3_magnitude = sqrt(sum(grad3.^2,2));
grad3 = bsxfun(@rdivide,grad3,grad3_magnitude);
grad3 = reshape(grad3,nf*3,n_src);

%% Divergence
div_res = - shape.DivOp * grad3;

%% Poisson
d = (shape.S)\full(div_res);


%% restore minimum
d = bsxfun(@minus, d,min(d));

