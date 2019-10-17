function test_outputs = PL_AGGD(train_data,train_p_target,test_data,k,ker,par,Maxiter,lambda,mu,gama)
%PL_AGGD[1] is a partial label learning algorithm 
%    Syntax
%
%       test_outputs = PL_AGGD(train_data,train_p_target,test_data,k,ker,par,Maxiter,lambda,mu,gama);
%
%    Description
%      
%      parameters,
%           train_data     - An m * d array, the ith instance of training instance is stored in train_data(i,:)
%           train_p_target - An m * q array, if the jth class label is one of the partial labels for the ith training instance, then train_p_target(i,j) equals +1, otherwise train_p_target(i,j) equals 0
%			test_data      - An m * d array, the ith instance of test instance is stored in test_data(i,:) 
%           k              - Number of neighbors,here we set k=10
%           ker            - Type of kernel function,here we use rbf kernel
%           par            - Parameters of kernel function
%      and returns,
%           test_outputs   - An m * q array ,classification results on test data
%   [1]D.-B. Wang, L. Li, M.-L. Zhang. Adaptive graph guided disambiguation for partial label learning. In: Proceedings of the 25th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD'19), Anchorage, AK, 2019.
if nargin < 10
	gama = 0.05;
end
if nargin < 9
	mu = 1;
end
if nargin < 8
	lambda = 1;
end
if nargin < 7
	Maxiter = 10;
end
if nargin < 6
	par = 1*mean(pdist(train_data));
end
if nargin < 5
	ker = 'rbf';
end
if nargin < 4
	k = 10;
end
if nargin < 3
	error('Not enough input parameters!');
end
y=build_label_manifold(train_data,train_p_target,k);
fprintf('Update parameters...\n')
[train_outputs, test_outputs] = MulRegression(train_data, y, test_data, gama, par, ker);
for i = 1:Maxiter
	fprintf('The %d-th iteration\n',i);
	W = obtain_W(train_data,y,k,lambda,mu);
	fprintf('Generate the labeling confidence...\n');
	y = UpdateY(W,train_p_target,train_outputs,mu);
	fprintf('Update parameters...\n')
	[train_outputs, test_outputs] = MulRegression(train_data, y, test_data, gama, par, ker);
end

end