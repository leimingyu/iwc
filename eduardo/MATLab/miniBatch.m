 function [params, J_history] = miniBatch(f,params,X,y,epsilon,alpha,batch_size,epochs)
 %  This is a mini-batch learnibg algorithm with adaptative learning rates
 %  and momentum

% Setup some useful variables
m=size(X,1);
numbatches=floor(m/batch_size);
J_history=zeros(epochs,1);

Theta_velocity=zeros(size(params));
local_gain=ones(size(params));
previous_grad=ones(size(params));

for e=1:epochs
    perm=randperm(m);
    X=X(perm,:);
    y=y(perm,:);
    for b=1:numbatches
        Xbatch=X((b-1)*batch_size+1:b*batch_size,:);
        ybatch=y((b-1)*batch_size+1:b*batch_size,:);
        [J_history(e), Theta_grad]=f(params,Xbatch,ybatch);
        
        previous_grad=Theta_grad.*previous_grad;
        local_gain(previous_grad>0)=local_gain(previous_grad>0)+0.05;
        local_gain(previous_grad<=0)=local_gain(previous_grad<=0).*0.95;
        previous_grad=Theta_grad;
        
        Theta_velocity=alpha.*Theta_velocity-(epsilon*local_gain).*Theta_grad;
        params=params+Theta_velocity;
    end
    fprintf('Iteration %4i | Cost: %4.6e\r', e, J_history(e));
end

end
