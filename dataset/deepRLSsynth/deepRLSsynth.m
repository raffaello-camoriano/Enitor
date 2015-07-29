
classdef deepRLSsynth < dataset
   
   properties
        
%         d
        omega
        shift
        amplitude
        noisesd
   end
   
   methods
       

        function y = gauss(obj,x,m,s)
             y = exp(-(x-m).^2./(2*s^2));
        end

        function y = fsin(obj,x,m,s)
             y = sin(2*pi/s*(x-m));
        end            

        function y = createFunct(obj,x)
            y = obj.gauss(x,-1,0.2).*obj.fsin(x,-0.7,0.21) + ...
                obj.gauss(x,-0.2,0.7).*obj.fsin(x,-0.231,2) + ...
                obj.gauss(x,1,0.2).*obj.fsin(x,0.67,0.21);
        end       
        %     d : number of input attributes in each case, for example `32'.
        function obj = deepRLSsynth(nTr , nTe)
            
            
            sigmanoise = 0.1;
            
            obj.nTr = nTr;
            obj.nTe = nTe;
%             obj.d = d;
            obj.d = 1;
            

            Xtr = 4*rand(nTr, 1) - 2;
            Xts = linspace(-2,2,obj.nTe)';
            ytr = obj.createFunct(Xtr) + sigmanoise * randn(size(Xtr,1),1);
            yts = obj.createFunct(Xts)+ sigmanoise * randn(size(Xts,1),1);
            

            obj.X = [Xtr ; Xts];
            obj.Y = [ytr ; yts];
            
%             obj.X = obj.scale(obj.X);   
            
            obj.n = size(obj.X , 1);
            obj.nTrTot = obj.nTr;
            obj.nTeTot = obj.nTe ;
            
            % Select consecutive samples
            obj.trainIdx = 1:nTr;          
            obj.testIdx = obj.nTrTot + 1 : obj.nTrTot + nTe;      
            
%             obj.shuffleAllIdx();
%             obj.shuffleTrainIdx();
%             obj.shuffleTestIdx();
            
            % Set problem type
%             if obj.hasRealValues(obj.Y)
                obj.problemType = 'regression';
%             else
%                 obj.problemType = 'classification';
%             end
        end
        
        
        
        % Checks if matrix Y contains real values. Useful for
        % discriminating between classification and regression, or between
        % predicted scores and classes
        function res = hasRealValues(obj , M)
        
            res = false;
            for i = 1:size(M,1)
                for j = 1:size(M,2)
                    if mod(M(i,j),1) ~= 0
                        res = true;
                    end
                end
            end
        end
        
        % Compute predictions matrix from real-valued scores matrix
%         function Ypred = scoresToClasses(obj , Yscores)    
%             
%             if strcmp(obj.outputFormat, 'zeroOne')
%                 Ypred = zeros(size(Yscores));
%             elseif strcmp(obj.outputFormat, 'plusMinusOne')
%                 Ypred = -1 * ones(size(Yscores));
%             elseif strcmp(obj.outputFormat, 'plusOneMinusBalanced')
%                 Ypred = -1/(obj.t - 1) * ones(size(Yscores));
%             end
%             
%             for i = 1:size(Ypred,1)
%                 [~,maxIdx] = max(Yscores(i,:));
%                 Ypred(i,maxIdx) = 1;
%             end
%         end
            
        % Compute performance measure on the given outputs
        function perf = performanceMeasure(obj , Y , Ypred , varargin)
            % RMSE
            perf = sqrt(sum((Y - Ypred).^2)/size(Y,1));
        end
        
        % Scales matrix M between -1 and 1
        function Ms = scale(obj , M)
            
%             minVal = double([-95 0 -37 15 -50 -50 -50 -50 -200 -200 -200 -200]);
%             maxVal = double([10 161 80 106 50 50 50 50 200 200 200 200]);
            minVal = min(M,[],1);
            maxVal = max(M,[],1);
            
            Ms = zeros(size(M));
            for i = 1:size(M,2)
             
                Ms(:,i) = ((abs(minVal(i)) + M(:,i)  ) / (maxVal(i) - minVal(i))) * 2 - 1;
                
            end
        end        
   end % methods
end % classdef