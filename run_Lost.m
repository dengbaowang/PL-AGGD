load('Lost.mat');
%[n,~] = size(data);
%data = zscore(data);
% parameter for PL-AGGD
par = 1*mean(pdist(train_data)); %Parameters of kernel function
%training
test_outputs = PL_AGGD(train_data,train_p_target,test_data,k,ker,par,Maxiter,lambda,mu,gama);
accuracy = CalAccuracy(test_outputs, test_target);
fprintf('The accuracy of PL-AGGD is: %f \n',accuracy);