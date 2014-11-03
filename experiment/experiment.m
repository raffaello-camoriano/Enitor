classdef experiment < handle
    %UNTITLED Summary of this class goes here
    %   Detailed explanation goes here
    
    properties

        algo
        ds
        performanceMeasure
        numRep          % Number of repetitions
        measureTime     % Flag for computational time storage
        time            % Struct containing the stored computational time
        saveResult      % Flag for result structure saving
        resdir          % Results saving directory
        
        %name
        customStr
        result

    end
    
    methods
        
        function obj = experiment( algo , ds , numRep , measureTime , saveResult , customStr , resdir)
            %obj.name = name_;
            
            assert(isa(algo,'algorithm') , '1st argument is not of class algorithm');
            assert(isa(ds,'dataset') , '2nd argument is not of class dataset');

            obj.algo = algo;
            obj.ds = ds;
            
            if nargin > 2
               obj.numRep = numRep; 
            end
            
            if nargin > 3
               obj.measureTime = measureTime;
               obj.time = struct;
            end
            
            if nargin > 4
               obj.saveResult = saveResult; 
            end      
            
            if nargin > 5
               obj.customStr = customStr; 
            else
               obj.customStr = '';
            end      
            
            if nargin > 6
               obj.resdir = resdir; 
            else
               obj.resdir = '';
            end
            
            % get handle to the performanceMeasure function specific of the
            % dataset, to be passed to the algo.train method
            obj.performanceMeasure = @ds.performanceMeasure;
            
            % Set experiment metadata
            
            obj.result.algorithm = class(obj.algo);
            obj.result.dataset = class(obj.ds);
            obj.result.numRep = obj.numRep;
        end
        
        function obj = run (obj )

            % Get training set (Xtr, Ytr) from dataset
            Xtr = obj.ds.X(obj.ds.trainIdx,:);
            Ytr = obj.ds.Y(obj.ds.trainIdx,:);
            Xte = obj.ds.X(obj.ds.testIdx,:);
            Yte = obj.ds.Y(obj.ds.testIdx,:);
                      
            if obj.measureTime
                tic;
            end
            
                obj.algo.train( Xtr , Ytr, obj.performanceMeasure , true , 0.2);
                
            if obj.measureTime
                obj.time.train = toc;
            end
            
            % Fill result structure
            
            if obj.measureTime
                tic;
            end
            
            obj.result.Ypred = obj.algo.test(Xte);   
            
            if obj.measureTime
                obj.time.test = toc;
                obj.result.time = obj.time;
            end
            
            
            obj.result.perf = obj.ds.performanceMeasure( Yte , obj.result.Ypred );
            
            if isprop(obj.algo,'kerParStar')
                obj.result.kerParStar = obj.algo.kerParStar;
            end
            
            if isprop(obj.algo,'filterParStar')
                obj.result.filterParStar = obj.algo.filterParStar;
            end
            
            % Save result structure
            if obj.saveResult
               dt = datetime('now','TimeZone','local','Format','d-MMM-y HH:mm:ss Z');
               fName = ['./' , obj.resdir , '/Exp_' , obj.result.algorithm , '_' , obj.result.dataset , '_' , obj.customStr , '_' , datestr(dt)];
               res = obj.result;
               save( fName , 'res' );
            end
        end
    end
end

