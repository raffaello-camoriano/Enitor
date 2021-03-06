classdef experiment < handle
    %UNTITLED Summary of this class goes here
    %   Detailed explanation goes here
    
    properties

        algo
        filter
        ds
        performanceMeasure
        numRep          % Number of repetitions
        measureTime     % Flag for computational time storage
        time            % Struct containing the stored computational time
        saveResult      % Flag for result structure saving
        resdir          % Results saving directory
        memoryProfiler  % Flag for memory profiling
        verbose         % Verbosity flag
        recompute
        validationPart
        
        %name
        customStr
        result

    end
    
    methods
        
        function obj = experiment( algo , ds , numRep , measureTime , saveResult , customStr , resdir , verbose , recompute, validationPart)
            
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
               obj.memoryProfiler = 1-measureTime;
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
            
            % Set verbosity
            obj.verbose = 0;
            if nargin > 7
                if verbose == 1
                   obj.verbose = verbose;
                end
            end
            
            if nargin > 8
                obj.recompute = recompute;
            else
                obj.recompute = 0;
            end
            
            
            if nargin > 9
                obj.validationPart = validationPart;
            else
                obj.validationPart = 0.2;
            end            

            if obj.memoryProfiler && obj.measureTime
                warning('Both time and memory profiling have been activated. Time measurements may not be reliable!')
            end
                
            % get handle to the performanceMeasure function specific of the
            % dataset, to be passed to the algo.train method
            obj.performanceMeasure = @ds.performanceMeasure;
            
            % Set experiment metadata
            
            obj.result.algorithm = class(obj.algo);
            obj.result.dataset = class(obj.ds);
            obj.result.numRep = obj.numRep;
            obj.result.numMapParGuesses = algo.numMapParGuesses;
            obj.result.numFilterParGuesses = algo.numFilterParGuesses;
            obj.result.nTr = ds.nTr;
            obj.result.nTe = ds.nTe;
            obj.result.d = ds.d;
            obj.result.t = ds.t;
            
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
            
            if obj.memoryProfiler
                profile -memory on
            end

            obj.algo.train( Xtr , Ytr, obj.performanceMeasure , obj.recompute , obj.validationPart, 'Xte' , Xte, 'Yte' , Yte);
                
            if obj.measureTime
                obj.time.train = toc;
            end
            
            % Fill result structure
            if obj.measureTime
                tic;
            end
            
            obj.result.Ypred = obj.algo.test(Xte);   
            obj.result.Y = Yte;
            
            if obj.memoryProfiler
                profile off;
            end
            
            if obj.measureTime
                obj.time.test = toc;
                obj.result.time = obj.time;
            end
            
            if obj.memoryProfiler
                p = profile('info');
                                
                obj.result.memoryProfile = struct;

                for i = 1:size(p.FunctionTable,1)
                    obj.result.memoryProfile(i).FunctionName = p.FunctionTable(i).FunctionName;
                    obj.result.memoryProfile(i).TotalMemAllocated = p.FunctionTable(i).TotalMemAllocated;
                    obj.result.memoryProfile(i).TotalMemFreed = p.FunctionTable(i).TotalMemFreed;
                    obj.result.memoryProfile(i).PeakMem = p.FunctionTable(i).PeakMem;
                end

            end
            
            obj.result.perf = abs(obj.ds.performanceMeasure( Yte , obj.result.Ypred , obj.ds.testIdx));
            
            % Save hyperparameter ranges
%             if isprop(obj.algo,'filterParGuesses')
%                 obj.result.filterParGuesses = obj.algo.filterParGuesses;
%             end
%             
%             if isprop(obj.algo,'kerParGuesses')
%                 obj.result.kerParGuesses = obj.algo.kerParGuesses;
%             end
%             
%             if isprop(obj.algo,'mapParGuesses')
%                 obj.result.mapParGuesses = obj.algo.mapParGuesses;
%             end

            obj.result.algo = obj.algo;
            obj.result.filter = obj.filter;
                        
            
            % Save best hyperparameters
            if isprop(obj.algo,'kerParStar')
                obj.result.kerParStar = obj.algo.kerParStar;
            end
            
            if isprop(obj.algo,'mapParStar')
                obj.result.mapParStar = obj.algo.mapParStar;
            end
            
            if isprop(obj.algo,'filterParStar')
                obj.result.filterParStar = obj.algo.filterParStar;
            end
            
            
            % Save result structure
            if obj.saveResult
            
               % Compatible version
               dt = clock;
               dt = fix(dt);
               fName = ['./' , obj.resdir , '/Exp_' , obj.result.algorithm , '_' , obj.result.dataset , '_' , obj.customStr , '_' , mat2str(dt)];
               
               res = obj.result;
               save( fName , 'res' );
            end
        end
    end
end

