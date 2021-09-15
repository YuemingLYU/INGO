function [xmin,hist,flag]=FastINGO(N,strfitnessfct)
% Fast INGO (diagnal covaraince matrix), 
% It is better for separable problem, but not rotation-invariant
% N is the input dimension
% test function: 'fcigar', 'ftablet', 'felli', 'felliL1', 'frastrigin10','flevy'

  flag = 0;
  tic
  hist=[];

  mu = rand(N,1);    % objective variables initial point
  
  stopfitness =  1* 1e-10;  % stop if fitness < stopfitness (minimization)
  lambda = 4+floor(3*log(N));
  lambda = 2-mod(lambda,2)+lambda;

  disp(strcat(['lambda: ', num2str(lambda)]));

  ad = log10(N);

  stopeval = 1*10^(3+ad)*lambda;
 
  bestVa =Inf;
  SIGMA = ones(N,1) * 0.25;
  INVsigma = 1./SIGMA;

  beta =  1/(N^0.5)   ;
  counteval = 0;  
  while counteval < stopeval

      
      Z = randn(N,lambda);
      X = bsxfun(@times,Z,SIGMA.^(1/2));
      arx = bsxfun(@plus,X,mu);

      arfitness = feval(strfitnessfct, arx);
      counteval = counteval+lambda;
     

      fstd = std(arfitness);
      
      Nscore = (arfitness'-mean(arfitness))/fstd/length(arfitness);
      
      
          
      mu = mu - beta*X*Nscore;
      
     
      Zw2 =  (Z.^2) * Nscore;
     
      INVsigma = INVsigma + beta* Zw2.*INVsigma;
    
      SIGMA = 1./INVsigma;
      if isnan(SIGMA)
          break;
      end
      if min(arfitness) <= stopfitness   
          flag=1;break;
      end
      if min(arfitness)<bestVa
          [bestVa,idx]=min(arfitness);
         
          xmin = arx(:,idx);
      end

      if mod(counteval/lambda,1000)==0
          disp([num2str(counteval/lambda) ': ' num2str(min(arfitness)) '  EvaN: ' num2str(counteval)]);
          hist =[hist;[counteval,min(arfitness)]]; 
      end

  end

disp([num2str(counteval/lambda) ': ' num2str(min(arfitness)) '  EvaN:' num2str(counteval)]);
hist =[hist;[counteval,min(arfitness)]];
toc



% ---------------------------------------------------------------


function f=fcigar(x)
 f = x(1,:).^2 + 1e6*sum(x(2:end,:).^2,1);


function f=ftablet(x)
 f = 1e6*x(1,:).^2 + sum(x(2:end,:).^2,1);

function f=felli(x)
  N = size(x,1); if N < 2 error('dimension must be greater one'); end
  f=1e6.^((0:N-1)/(N-1)) * x.^2;

function f=felliL1(x)
  N = size(x,1); if N < 2 error('dimension must be greater one'); end
  f=1e6.^((0:N-1)/(N-1)) * abs(x);





function f=felliL12(x)
  N = size(x,1); if N < 2 error('dimension must be greater one'); end
  f=1e6.^((0:N-1)/(N-1)) * abs(x).^(1/2);




function f=fdiffpow(x)
  N = size(x,1); if N < 2 error('dimension must be greater one'); end
  f=sum(bsxfun(@power,abs(x),(2+10*(0:N-1)'/(N-1))),1);



function f=frastrigin10(x)
  N = size(x,1); if N < 2 error('dimension must be greater one'); end
  Ma =10;
  scale=Ma.^((0:N-1)'/(N-1));
 f = Ma*size(x,1) + sum((bsxfun(@times,x,scale)).^2 - Ma*cos(2*pi*(bsxfun(@times,x,scale))),1);



function f=flevy(x)
d=size(x,1);
w= 1+ (x-1)/4;
f = sin(w(1,:)*pi).^2 + sum((w(1:(d-1),:)-1).^2.*(1+10*sin(pi*w(1:(d-1),:)+1).^2),1) + (w(d,:)-1).^2 .* (1+10*sin(2*pi*w(d,:)).^2);

 
