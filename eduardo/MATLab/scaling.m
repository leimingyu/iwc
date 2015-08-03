function [X] = scaling(X)

m=size(X,1);
miu=repmat(mean(X,1),m,1);
sig=repmat(std(X,0,1),m,1);
X(sig~=0)=(X(sig~=0)-miu(sig~=0))./sig(sig~=0);

end
