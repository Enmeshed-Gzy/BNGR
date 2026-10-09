function result=case2(mode,workers,outDir,seed)
if nargin<1 || isempty(mode), mode='auto'; end
if nargin<2 || isempty(workers), workers=8; end
if nargin<3, outDir=''; end
if nargin<4 || isempty(seed), seed=2026; end
mode=validatestring(mode,{'auto','cpu','gpu','mex','split'});
validateattributes(workers,{'numeric'},{'scalar','integer','nonnegative'});
validateattributes(seed,{'numeric'},{'scalar','real','finite','integer','>=',0,'<=',2^32-1});
seed=double(seed);
root=fileparts(mfilename('fullpath'));
featureNames=arrayfun(@(j)sprintf('X%d',j),1:8,'UniformOutput',false);
dataNames=[{'SourceRow'},featureNames,{'Response'}];
trainFile=fullfile(root,'train.csv'); testFile=fullfile(root,'test.csv');
trainTable=readtable(trainFile,'VariableNamingRule','preserve');
testTable=readtable(testFile,'VariableNamingRule','preserve');
assert(isequal(trainTable.Properties.VariableNames,dataNames) ...
    &&isequal(testTable.Properties.VariableNames,dataNames), ...
    'BNGR:CSVSchema','Expected SourceRow, input columns, and Response only.');
data=sortrows([trainTable;testTable],'SourceRow');
sourceRows=data.SourceRow; A=data{:,dataNames(2:end)}; n=height(data);
assert(all_finite(A)&&n>=10&&size(A,2)==8+1,'BNGR:Data','Invalid data dimensions or values.');
assert(all_finite(sourceRows)&&numel(unique(sourceRows))==n,'BNGR:SourceRows','SourceRow must be unique.');
assert(all(A(:,end)~=0),'BNGR:MAPE','MAPE requires nonzero observed responses.');
[~,~,groups]=unique(A,'rows'); groupCount=max(groups);
assert(groupCount>=5,'BNGR:Groups','At least five distinct records are required.');
previousRng=rng;
rngGuard=onCleanup(@() rng(previousRng));
rng(seed,'twister');
partition=cvpartition(groupCount,'KFold',5);
groupFold=zeros(groupCount,1);
for f=1:5, groupFold(test(partition,f))=f; end
clear rngGuard
labels=groupFold(groups);
orderStream=RandStream('mt19937ar','Seed',mod(seed+1,2^32));
globalOrder=randperm(orderStream,n)';
orders=cell(1,5); trainingOrder=table;
for f=1:5
    ids=globalOrder(labels(globalOrder)~=f);
    held=find(labels==f);
    assert(numel(ids)>=5&&~isempty(held)&&isempty(intersect(ids,held)), ...
        'BNGR:Partition','Invalid training/test fold.');
    assert(std(A(ids,end),0)>0&&sum((A(held,end)-mean(A(held,end))).^2)>0, ...
        'BNGR:Response','A fold has constant responses; do not select a new seed based on its score.');
    orders{f}=ids;
    trainingOrder=[trainingOrder;table(repmat(f,numel(ids),1),(1:numel(ids))',sourceRows(ids), ...
        'VariableNames',{'Fold','Rank','SourceRow'})];
end
assert(all(accumarray(groups,labels,[],@min)==accumarray(groups,labels,[],@max)), ...
    'BNGR:Duplicates','Identical records must remain in one fold.');
foldAssignment=table(sourceRows,groups,labels,'VariableNames',{'SourceRow','RecordGroup','Fold'});
P=struct('Case',2,'Rows',n,'Features',8,'Folds',5,'Seed',seed, ...
    'TrainingOrderSeed',mod(seed+1,2^32),'DistinctRecords',groupCount, ...
    'PartitionUnit','Identical full records grouped together', ...
    'ValidationScope','Random five-fold cross-validation on the pooled CSV data', ...
    'IndependentHoldout',false,'ResponseStratification',false, ...
    'ScoreSelectedPartition',false,'ScoreSelectedTrainingOrder',false, ...
    'TrainingOrderRule','One independently seeded random order filtered by training membership', ...
    'TrainSHA256',file_hash(trainFile),'TestSHA256',file_hash(testFile), ...
    'MethodSHA256',file_hash([mfilename('fullpath'),'.m']), ...
    'DriverSHA256',file_hash([mfilename('fullpath'),'.m']), ...
    'RequestedMode',mode,'RequestedWorkers',workers,'InputTransform','none', ...
    'EvidenceRule','Unchanged scalar-Hessian Laplace with the existing checked quadrature fallback');
if 2==2, P.InputTransform='Natural logarithm of age (X8)'; end
if isempty(outDir)
    outDir=fullfile(fileparts(root),['case2_random5_',char(datetime('now','Format','yyyyMMdd_HHmmss_SSS'))]);
end
assert(exist(outDir,'file')==0&&exist(outDir,'dir')==0, ...
    'BNGR:OutputExists','Choose a new output directory.');
[ok,msg]=mkdir(outDir); assert(ok,'BNGR:Output','%s',msg);
[~,info]=fileattrib(outDir); outDir=info.Name;
write_precision_table(foldAssignment,fullfile(outDir,'fold_assignment.csv'));
write_precision_table(trainingOrder,fullfile(outDir,'training_order.csv'));
write_json(fullfile(outDir,'protocol.json'),P);
if strcmp(mode,'split')
    result=struct('Completed',true,'SplitOnly',true,'Protocol',P,'FoldAssignment',foldAssignment, ...
        'TrainingOrder',trainingOrder,'OutputDirectory',outDir);
    disp(groupsummary(foldAssignment,'Fold'));
    return
end
B=A;
if 2==2
    assert(all(A(:,8)>0),'BNGR:Age','Age must be positive.');
    B(:,8)=log(A(:,8));
end
C=bngr_method('options');
[pathCleanup,mexBuild]=setup_optional_mex();
if exist('bngr_grid_mex','file')==3
    assert(mexBuild.Available&&isfield(mexBuild,'BinaryPath') ...
        &&strcmp(canonical_path(which('bngr_grid_mex')),canonical_path(mexBuild.BinaryPath)), ...
        'BNGR:MEXSource','An external MEX binary shadows the verified kernel.');
end
if license('test','Distrib_Computing_Toolbox')&&exist('gcp','file')~=0
    pool=gcp('nocreate');
    if ~isempty(pool)&&~contains(class(pool),'Thread')
        addAttachedFiles(pool,{[mfilename('fullpath'),'.m']});
    end
end
opts=struct('mode',mode,'workers',workers,'useParallel',true); ids=orders{1};
hardware=prepare_hardware(B(ids,1:end-1),B(ids,end),opts);
write_json(fullfile(outDir,'hardware.json'),hardware);
models=cell(1,5); fitSeconds=zeros(5,1);
fprintf('Case %d | seed %u | %d samples | random five-fold CV\n',2,seed,n);
for f=1:5
    ids=orders{f}; clock=tic;
    model=bngr_method('fit',B(ids,1:end-1),B(ids,end),C,hardware,@bngr_grid);
    validate_model(model,numel(ids),8,C);
    fitSeconds(f)=toc(clock); models{f}=model;
    fprintf('Fold %d/5: selected %s from %d subsets, %.2f s\n', ...
        f,mat2str(model.selectedFeatures),model.candidateCount,fitSeconds(f));
end
metrics=table; selected=table; candidates=table; summary=table; oof=nan(n,1);
for f=1:5
    model=models{f};
    chosen=table(2,f,model.selected.mask,string(mat2str(model.selectedFeatures)), ...
        model.selected.varphi1,model.selected.varphi2,model.selected.logEvidence, ...
        model.candidateCount,fitSeconds(f),'VariableNames', ...
        {'Case','Fold','Mask','Features','Phi1','Phi2','LogEvidence','CandidateCount','FitSeconds'});
    selected=[selected;chosen];
    allModels=model.allModels; allModels.Case=repmat(2,height(allModels),1);
    allModels.Fold=repmat(f,height(allModels),1); candidates=[candidates;allModels];
    for part=1:2
        if part==1, ids=orders{f}; label="Training"; else, ids=find(labels==f); label="TestFold"; end
        prediction=bngr_method('predict',model,B(ids,1:end-1)); actual=A(ids,end);
        row=prediction_metrics(actual,prediction);
        row.Case=2; row.Fold=f; row.Partition=label; row.SampleCount=numel(ids);
        row.SSE=sum((actual-prediction).^2); row.SST=sum((actual-mean(actual)).^2);
        assert(abs(row.R2-(1-row.SSE/row.SST))<1e-12,'BNGR:Metric','R2 disagreement.');
        metrics=[metrics;row];
        if part==2, oof(ids)=prediction; end
    end
    fprintf('\nFold %d training/test metrics:\n',f);
    disp(metrics(metrics.Fold==f,{'Partition','SampleCount','R2','RMSE','MAE','MAPE','RAE'}));
end
assert(all(isfinite(oof)),'BNGR:OOF','Every sample requires one held-out prediction.');
metricNames={'R2','RMSE','MAE','MAPE','RAE'};
for label=["Training","TestFold"]
    rows=metrics(metrics.Partition==label,:);
    item=table(2,label,5,'VariableNames',{'Case','Partition','FoldCount'});
    for j=1:numel(metricNames)
        item.([metricNames{j},'Mean'])=mean(rows.(metricNames{j}));
        item.([metricNames{j},'SD'])=std(rows.(metricNames{j}),0);
    end
    summary=[summary;item];
end
pooled=prediction_metrics(A(:,end),oof); pooled.Case=2; pooled.SampleCount=n;
pooled.SSE=sum((A(:,end)-oof).^2); pooled.SST=sum((A(:,end)-mean(A(:,end))).^2);
oofTable=table(sourceRows,labels,A(:,end),oof,'VariableNames',{'SourceRow','Fold','Actual','Prediction'});
write_precision_table(metrics,fullfile(outDir,'fold_metrics.csv'));
write_precision_table(summary,fullfile(outDir,'summary_mean_sd.csv'));
write_precision_table(pooled,fullfile(outDir,'pooled_oof_metrics.csv'));
write_precision_table(selected,fullfile(outDir,'selected_models.csv'));
write_precision_table(candidates,fullfile(outDir,'candidate_models.csv'));
write_precision_table(oofTable,fullfile(outDir,'oof_predictions.csv'));
result=struct('Completed',true,'SplitOnly',false,'Case',2,'OutputDirectory',outDir, ...
    'Protocol',P,'FoldAssignment',foldAssignment,'TrainingOrder',trainingOrder, ...
    'FoldMetrics',metrics,'Summary',summary,'PooledMetrics',pooled,'SelectedModels',selected, ...
    'Hardware',hardware,'MEXBuild',mexBuild,'AllModels',{cellfun(@(m)m.allModels,models,'UniformOutput',false)}, ...
    'OOFPrediction',oof,'TrainingPredictionIncludesSelf',true);
write_json(fullfile(outDir,'complete.json'),struct('Completed',true,'Case',2,'Folds',5,'Seed',seed));
disp(summary); fprintf('Saved: %s\n',outDir);
end

function write_json(file,value)
fid=fopen(file,'w','n','UTF-8'); assert(fid>0); guard=onCleanup(@() fclose(fid));
fprintf(fid,'%s\n',jsonencode(value,'PrettyPrint',true));
end

function yes=all_finite(values)
yes=all(isfinite(values(:)));
end

function validate_model(model,n,m,C)
T=model.allModels; count=2^m-1;
assert(model.N==n && model.m==m && model.candidateCount==count && height(T)==count ...
    && isequal(sort(T.mask),(1:count)') && all(isfinite(T.logEvidence)), ...
    'BNGR:Subsets','Every nonempty subset must have finite evidence.');
[~,winner]=max(T.logEvidence);
assert(model.selected.mask==T.mask(winner) && model.selected.logEvidence==T.logEvidence(winner) ...
    && model.selected.varphi1==T.varphi1(winner) && model.selected.varphi2==T.varphi2(winner) ...
    && isequal(model.selectedFeatures(:)',find(bitget(model.selected.mask,1:m))), ...
    'BNGR:Winner','The selected model must maximize evidence.');
names=fieldnames(C);
for j=1:numel(names)
    if strcmp(names{j},'useParallel'), continue; end
    assert(isequal(model.config.(names{j}),C.(names{j})),'BNGR:Config','Numerical configuration changed.');
end
end

function [pathCleanup,build]=setup_optional_mex()
oldPath=path; pathCleanup=onCleanup(@() path(oldPath));
build=struct('Attempted',false,'Available',false,'Message','');
try
    [source,sourceHash]=embedded_mex_source();
    cache=fullfile(tempdir,'BNGR_ThreeFiles_mex',sourceHash,version('-release'),computer('arch'));
    binary=fullfile(cache,['bngr_grid_mex.',mexext]);
    sourcePath=fullfile(cache,'bngr_grid_mex.cpp'); receipt=fullfile(cache,'binary.sha256');
    cacheValid=exist(binary,'file')==3 && exist(sourcePath,'file')==2 && exist(receipt,'file')==2;
    if cacheValid
        cacheValid=strcmp(file_hash(sourcePath),sourceHash) && strcmp(file_hash(binary),strtrim(fileread(receipt)));
    end
    if ~cacheValid
        compiler=mex.getCompilerConfigurations('C++','Selected');
        if isempty(compiler)
            build.Message='No configured C++ compiler; CPU/GPU remain available.';
            return
        end
        build.Attempted=true; if ~exist(cache,'dir'), mkdir(cache); end
        fid=fopen(sourcePath,'wb'); assert(fid>0); fwrite(fid,uint8(source),'uint8'); fclose(fid);
        assert(strcmp(file_hash(sourcePath),sourceHash),'BNGR:MEXSource','Embedded C++ hash mismatch.');
        args={'-R2018a','-O','-outdir',cache,sourcePath};
        if ispc && contains(lower(compiler.Manufacturer),'microsoft')
            args=[args,{'COMPFLAGS=$COMPFLAGS /O2 /fp:precise'}];
        elseif ispc || (isunix && ~ismac)
            args=[args,{'CXXFLAGS=$CXXFLAGS -O3 -std=c++11 -ffp-contract=off'}];
        end
        mex(args{:});
        fid=fopen(receipt,'wt'); assert(fid>0); fprintf(fid,'%s\n',file_hash(binary)); fclose(fid);
    end
    addpath(cache,'-begin'); clear bngr_grid_mex; rehash;
    assert(exist('bngr_grid_mex','file')==3 && strcmp(canonical_path(which('bngr_grid_mex')),canonical_path(binary)), ...
        'BNGR:MEXSource','The embedded-source MEX cache is shadowed by another binary.');
    build.BinaryPath=binary; build.BinarySHA256=file_hash(binary); build.Available=true;
    build.Message='Optional kernel must pass the original runtime equivalence check before use.';
catch issue
    build.Message=issue.message;
    warning('BNGR:OptionalMEX','Optional MEX unavailable: %s',issue.message);
end
end

function value=canonical_path(file)
[ok,info]=fileattrib(file); assert(ok,'BNGR:Path','Missing path: %s',file); value=info.Name;
end

function value=file_hash(file)
assert(usejava('jvm'),'BNGR:JVM','MATLAB JVM is required for SHA-256 checks.');
fid=fopen(file,'rb'); assert(fid>0,'BNGR:Hash','Cannot read %s',file);
cleanup=onCleanup(@() fclose(fid));
digest=java.security.MessageDigest.getInstance('SHA-256');
while true
    bytes=fread(fid,1024*1024,'*uint8'); if isempty(bytes), break; end
    digest.update(typecast(bytes,'int8'));
end
value=lower(reshape(dec2hex(typecast(digest.digest(),'uint8'),2)',1,[]));
end

function write_precision_table(T,file)
fid=fopen(file,'w','n','UTF-8'); assert(fid>0); cleanup=onCleanup(@() fclose(fid));
headers=cellfun(@csv_quote,T.Properties.VariableNames,'UniformOutput',false);
fprintf(fid,'%s\n',strjoin(headers,','));
for i=1:height(T)
    fields=cell(1,width(T));
    for j=1:width(T)
        v=T{i,j}; if iscell(v), v=v{1}; end
        if isnumeric(v) || islogical(v), assert(isscalar(v)); fields{j}=sprintf('%.17g',double(v));
        else, fields{j}=csv_quote(char(string(v))); end
    end
    fprintf(fid,'%s\n',strjoin(fields,','));
end
end

function value=csv_quote(value)
value=['"',strrep(char(value),'"','""'),'"'];
end

function [source,sourceHash]=embedded_mex_source()

lines={
    '#include <cmath>'
    '#include <cstddef>'
    '#include <limits>'
    '#ifdef _OPENMP'
    '#include <omp.h>'
    '#endif'
    ''
    'extern "C" void bngr_q_grid(const double* h, const double* lam,'
    '    const double* previous, const double* actual, double shift,'
    '    std::size_t n, const double* t, std::size_t count, int threads, double* q) {'
    '#ifdef _OPENMP'
    '    #pragma omp parallel for num_threads(threads) if(threads>1 && count>1)'
    '#endif'
    '    for (std::ptrdiff_t a=0; a<static_cast<std::ptrdiff_t>(count); ++a) {'
    '        const double inv=std::exp(-t[a]);'
    '        double ss=0; std::size_t off=0;'
    '        for (std::size_t row=1; row<=n; ++row) {'
    '            double den=0, num=0;'
    '            for (std::size_t j=0; j<row; ++j) {'
    '                const double z=h[off+j]*inv;'
    '                if(z>750) continue;'
    '                const double w=z==0 ? 1.0 : std::exp(-z);'
    '                den+=w; num+=w*previous[off+j];'
    '            }'
    '            const double error=actual[row-1]-num/den;'
    '            ss+=lam[row-1]*error*error;'
    '            off+=row;'
    '        }'
    '        q[a]=shift+std::log(ss);'
    '    }'
    '}'
    ''
    '#ifndef BNGR_STANDALONE'
    '#include "mex.h"'
    'static void numeric(const mxArray* a,const char* name) {'
    '    if(!mxIsDouble(a)||mxIsComplex(a)||mxIsSparse(a))'
    '        mexErrMsgIdAndTxt("BNGR:MEXType","%s must be a full real double array.",name);'
    '}'
    'void mexFunction(int nlhs,mxArray* plhs[],int nrhs,const mxArray* prhs[]) {'
    '    if((nrhs!=7&&nrhs!=8)||nlhs!=1)'
    '        mexErrMsgIdAndTxt("BNGR:MEXUsage","Use q=bngr_grid_mex(h,lam,previous,actual,shift,n,t[,threads]).");'
    '    const char* names[]={"h","lam","previous","actual","shift","n","t","threads"};'
    '    for(int j=0;j<nrhs;++j)numeric(prhs[j],names[j]);'
    '    const double nd=mxGetScalar(prhs[5]);'
    '    if(mxGetNumberOfElements(prhs[4])!=1||mxGetNumberOfElements(prhs[5])!=1||'
    '       !std::isfinite(nd)||nd<1||nd!=std::floor(nd)||nd>1000000)'
    '        mexErrMsgIdAndTxt("BNGR:MEXSize","n must be S.n (training count minus one), a positive integer.");'
    '    const std::size_t n=static_cast<std::size_t>(nd), pairs=n*(n+1)/2;'
    '    if(mxGetNumberOfElements(prhs[0])!=pairs||mxGetNumberOfElements(prhs[2])!=pairs||'
    '       mxGetNumberOfElements(prhs[1])!=n||mxGetNumberOfElements(prhs[3])!=n)'
    '        mexErrMsgIdAndTxt("BNGR:MEXSize","Pair or conditional-row array lengths are inconsistent.");'
    '    int threads=1;'
    '    if(nrhs==8){'
    '        const double v=mxGetScalar(prhs[7]);'
    '        if(mxGetNumberOfElements(prhs[7])!=1||!std::isfinite(v)||v<1||v!=std::floor(v)||v>1024)'
    '            mexErrMsgIdAndTxt("BNGR:MEXThreads","threads must be a positive integer no larger than 1024.");'
    '        threads=static_cast<int>(v);'
    '    }'
    '#ifndef _OPENMP'
    '    if(threads>1)mexErrMsgIdAndTxt("BNGR:MEXOpenMP","Rebuild with OpenMP to request more than one inner thread.");'
    '#endif'
    '    const double shift=mxGetScalar(prhs[4]);'
    '    if(!std::isfinite(shift))mexErrMsgIdAndTxt("BNGR:MEXInput","lambdaShift must be finite.");'
    '    const std::size_t count=mxGetNumberOfElements(prhs[6]);'
    '    const double* ts=mxGetDoubles(prhs[6]);'
    '    for(std::size_t i=0;i<count;++i)'
    '        if(!std::isfinite(ts[i])||ts[i]<-700||ts[i]>700)'
    '            mexErrMsgIdAndTxt("BNGR:MEXInput","Log bandwidths must be finite and lie in [-700,700].");'
    '    plhs[0]=mxCreateNumericArray(mxGetNumberOfDimensions(prhs[6]),mxGetDimensions(prhs[6]),mxDOUBLE_CLASS,mxREAL);'
    '    bngr_q_grid(mxGetDoubles(prhs[0]),mxGetDoubles(prhs[1]),mxGetDoubles(prhs[2]),'
    '        mxGetDoubles(prhs[3]),shift,n,ts,count,threads,mxGetDoubles(plhs[0]));'
    '    const double* output=mxGetDoubles(plhs[0]);'
    '    for(std::size_t i=0;i<count;++i)'
    '        if(!std::isfinite(output[i]))mexErrMsgIdAndTxt("BNGR:MEXResidual","Degenerate residual sum: evidence undefined.");'
    '}'
    '#endif'
};
source=[strjoin(lines,char(10)),char(10)];
sourceHash='1a7911f455ad8cc2ccda4802e439fc9aa3f7b27645cc82b009c561bdb7151bd2';
end

function hardware=prepare_hardware(X,y,opts)

sigma=std(X,0,1); sigma(sigma==0)=1; D=zeros(size(X,1));
for j=1:size(X,2), D=D+((X(:,j)-X(:,j)')./sigma(j)).^2; end
S=bngr_method('pairs',D,(y-mean(y))/std(y));
C=bngr_method('options');
hardware=bngr_hardware(S,linspace(C.logLower,log(C.prior1),C.gridSize),opts);
end

function T=prediction_metrics(actual,prediction)
actual=actual(:); prediction=prediction(:); e=actual-prediction;
assert(numel(actual)==numel(prediction) && all(isfinite(prediction)), ...
    'Invalid predictions.');
R2=1-sum(e.^2)/sum((actual-mean(actual)).^2);
RMSE=sqrt(mean(e.^2)); MAE=mean(abs(e));
if any(actual==0), MAPE=NaN; else, MAPE=100*mean(abs(e./actual)); end
RAE=sum(abs(e))/sum(abs(actual-mean(actual)));
T=table(R2,RMSE,MAE,MAPE,RAE);
end

function [q,info] = bngr_grid(S,ts,backend,batchSize)

if nargin<3 || isempty(backend), backend='cpu'; end
if nargin<4 || isempty(batchSize), batchSize=32; end
backend=validatestring(backend,{'cpu','gpu','mex'});
validateattributes(ts,{'double'},{'real','finite','nonempty','vector'});
validateattributes(batchSize,{'numeric'},{'scalar','positive','integer'});
required={'h','G','previousEta','eta','lambdaScaled','lambdaShift'};
assert(all(isfield(S,required)),'BNGR:PairFields','Missing prepare_pairs fields.');
assert(isa(S.h,'double') && isa(S.G,'double') && isa(S.previousEta,'double') ...
    && isa(S.eta,'double') && isa(S.lambdaScaled,'double'), ...
    'BNGR:DoubleOnly','All numerical pair fields must be double precision.');
assert(~isa(S.h,'gpuArray'),'BNGR:HostPairs','Supply host prepare_pairs arrays.');
n=numel(S.eta); p=numel(S.h); shape=size(ts); tt=ts(:)';
assert(issparse(S.G) && isequal(size(S.G),[n p]) ...
    && numel(S.previousEta)==p && numel(S.lambdaScaled)==n, ...
    'BNGR:PairDimensions','Invalid prepare_pairs dimensions.');
assert(all(isfinite(S.h(:))) && all(S.h(:)>=0),'BNGR:PairDistance','Invalid h.');
info=struct('Backend',backend,'Precision','double','RequestedBatchSize',batchSize, ...
    'ActualBatchSize',0,'MemoryRetries',0,'PairCount',p,'GridCount',numel(tt));
h=S.h(:); G=S.G; previous=S.previousEta(:); actual=S.eta(:);
lambda=S.lambdaScaled(:); shift=S.lambdaShift;
if strcmp(backend,'mex')
    assert(exist('bngr_grid_mex','file')==3,'BNGR:MissingMEX', ...
        'Compile the optional kernel with build_bngr_grid_mex first, or use CPU.');

    assert(isfield(S,'n') && S.n==n && p==n*(n+1)/2, ...
        'BNGR:MEXPacking','MEX requires the unchanged prepare_pairs packing.');

    q=bngr_grid_mex(h,lambda,previous,actual,shift,S.n,ts,1);
    assert(isequal(size(q),shape) && all(isfinite(q(:))), ...
        'BNGR:MEXResult','MEX returned a nonfinite result or incorrect shape.');
    info.ActualBatchSize=1; info.MEXThreads=1;
    return
end

bytesPerColumn=8*(5*p+8*n);
if strcmp(backend,'gpu')
    assert(exist('gpuDevice','file')~=0,'BNGR:NoGPUAPI','GPU API is unavailable.');
    if exist('getCurrentTask','file')~=0
        assert(isempty(getCurrentTask),'BNGR:GPUWorker', ...
            'Run GPU grids on the MATLAB client, outside parallel workers.');
    end
    device=gpuDevice;

    residentBytes=8*(3*p+2*n)+16*nnz(G)+8*(p+1);
    budget=max(0,0.25*double(device.AvailableMemory)-residentBytes);
    assert(budget>=bytesPerColumn,'BNGR:GPUMemory', ...
        'Insufficient GPU memory even for one grid column. Use CPU.');
    batchSize=min([batchSize,numel(tt),max(1,floor(budget/bytesPerColumn))]);
    h=gpuArray(h); G=gpuArray(G); previous=gpuArray(previous);
    actual=gpuArray(actual); lambda=gpuArray(lambda);
else

    budget=128*1024^2;
    batchSize=min([batchSize,numel(tt),max(1,floor(budget/bytesPerColumn))]);
end
q=zeros(size(tt)); first=1; info.ActualBatchSize=batchSize;
while first<=numel(tt)
    ids=first:min(first+batchSize-1,numel(tt));
    try
        inverse=exp(-tt(ids));
        if strcmp(backend,'gpu'), inverse=gpuArray(inverse); end
        w=exp(-h.*inverse);
        denominator=G*w;
        prediction=(G*(w.*previous))./denominator;
        residual=actual-prediction;
        ss=sum(lambda.*residual.^2,1);
        if strcmp(backend,'gpu'), ss=gather(ss); end
        assert(all(isfinite(ss) & ss>0),'BNGR:DegenerateResidual', ...
            'The unchanged evidence objective has a nonpositive/nonfinite SSE.');
        q(ids)=shift+log(ss);
        first=ids(end)+1;
    catch err

        label=lower([err.identifier,' ',err.message]);
        outOfMemory=contains(label,'out of memory') || contains(label,'outofmemory') ...
            || contains(label,'insufficient memory') || contains(label,'alloc_failed');
        if ~strcmp(backend,'gpu') || ~outOfMemory || batchSize==1, rethrow(err); end
        clear w denominator prediction residual ss inverse
        batchSize=max(1,floor(batchSize/2));
        info.ActualBatchSize=batchSize;
        info.MemoryRetries=info.MemoryRetries+1;
    end
end
q=reshape(q,shape);
end

function H = bngr_hardware(S,ts,opts)

if nargin<3 || isempty(opts), opts=struct; end
defaults=struct('mode','auto','workers',0,'poolType','processes', ...
    'useParallel',true,'trials',2,'benchmarkColumns',385,'benchmarkJobs',0, ...
    'relativeTolerance',1e-10,'absoluteTolerance',1e-10);
names=fieldnames(defaults);
for k=1:numel(names)
    if ~isfield(opts,names{k}), opts.(names{k})=defaults.(names{k}); end
end
opts.mode=validatestring(opts.mode,{'auto','cpu','gpu','mex'});
opts.poolType=validatestring(opts.poolType,{'processes','threads'});
validateattributes(opts.workers,{'numeric'},{'scalar','integer','nonnegative'});
validateattributes(opts.trials,{'numeric'},{'scalar','integer','positive'});
validateattributes(opts.benchmarkColumns,{'numeric'},{'scalar','integer','positive'});
validateattributes(opts.benchmarkJobs,{'numeric'},{'scalar','integer','nonnegative'});
H=struct('backend','cpu','parallelSubsets',false,'parallelRefinement',false, ...
    'workers',0,'batchSize',16, ...
    'matlabVersion',version,'matlabRelease',version('-release'),'computer',computer, ...
    'toolboxes',ver,'options',opts,'poolReused',false,'poolCreated',false, ...
    'poolClass','','gpuAvailable',false,'gpuName','','gpuAvailableBytes',0, ...
    'gpuComputeCapability','','mexAvailable',exist('bngr_grid_mex','file')==3, ...
    'mexPath',which('bngr_grid_mex'),'mexThreads',1,'mexValidated',false, ...
    'mexParallelValidated',false,'useMEXForRefinement',false, ...
    'benchmark',struct([]),'notes',{{}}, ...
    'selectionScope','Measured sequential-residual grid throughput, not total fitting time', ...
    'benchmarkScope','Grid kernel only. GPU grids may be followed by a separate CPU parallel refinement stage.');
H.createdAt=char(datetime('now','Format','yyyy-MM-dd HH:mm:ss'));
columns=unique(round(linspace(1,numel(ts),min(numel(ts),opts.benchmarkColumns))));
testTs=reshape(ts(columns),1,[]);
reference=bngr_grid(S,testTs,'cpu',1);
candidate=struct('mode',{},'batchSize',{},'secondsPerJob',{},'maxAbsoluteError',{},'passed',{});

for batch=[1 4 16 32]
    [q,info]=bngr_grid(S,testTs,'cpu',batch);
    [passed,err]=equivalent(q,reference,opts);
    elapsed=zeros(1,opts.trials);
    for k=1:opts.trials
        timer=tic; bngr_grid(S,testTs,'cpu',batch); elapsed(k)=toc(timer);
    end
    candidate(end+1)=entry('cpu-serial',info.ActualBatchSize,median(elapsed),err,passed);
end
cpuCandidates=candidate([candidate.passed]);
assert(~isempty(cpuCandidates),'BNGR:CPUConsistency','CPU consistency test failed.');
[~,best]=min([cpuCandidates.secondsPerJob]); cpuBatch=cpuCandidates(best).batchSize;

if H.mexAvailable
    try
        q=bngr_grid(S,testTs,'mex',1);
        [passed,err]=equivalent(q,reference,opts);
        elapsed=zeros(1,opts.trials);
        for k=1:opts.trials
            timer=tic; bngr_grid(S,testTs,'mex',1); elapsed(k)=toc(timer);
        end
        candidate(end+1)=entry('mex-serial',1,median(elapsed),err,passed);
        H.mexValidated=passed;
        if ~passed, H.notes{end+1}='Optional MEX failed double-precision equivalence and was excluded.'; end
    catch issue
        H.notes{end+1}=['Optional MEX unavailable or failed: ',issue.message];
    end
end
pool=[]; hasPCT=license('test','Distrib_Computing_Toolbox') && exist('gcp','file')~=0;
if hasPCT && opts.useParallel
    try
        pool=gcp('nocreate'); H.poolReused=~isempty(pool);
        if isempty(pool)
            if strcmp(opts.poolType,'threads')
                if opts.workers>0, pool=parpool('Threads',opts.workers);
                else, pool=parpool('Threads'); end
            else

                if opts.workers>0, pool=parpool('local',opts.workers);
                else, pool=parpool('local'); end
            end
            H.poolCreated=true;
        end
        H.workers=pool.NumWorkers; H.poolClass=class(pool);
        jobs=opts.benchmarkJobs;
        if jobs==0, jobs=max(2,2*H.workers); end
        values=cell(1,jobs);

        parfor k=1:jobs, values{k}=bngr_grid(S,testTs,'cpu',cpuBatch); end
        elapsed=zeros(1,opts.trials);
        for trial=1:opts.trials
            timer=tic;
            parfor k=1:jobs, values{k}=bngr_grid(S,testTs,'cpu',cpuBatch); end
            elapsed(trial)=toc(timer)/jobs;
        end
        err=0; passed=true;
        for k=1:jobs
            [ok,difference]=equivalent(values{k},reference,opts);
            passed=passed && ok; err=max(err,difference);
        end
        candidate(end+1)=entry('cpu-parallel',cpuBatch,median(elapsed),err,passed);
        H.parallelRefinement=passed && H.workers>1;
        if ~passed
            H.notes{end+1}='CPU parallel equivalence failed; CPU refinement will run serially.';
        end
        if H.mexValidated
            try

                parfor k=1:jobs, values{k}=bngr_grid(S,testTs,'mex',1); end
                elapsed=zeros(1,opts.trials);
                for trial=1:opts.trials
                    timer=tic;
                    parfor k=1:jobs, values{k}=bngr_grid(S,testTs,'mex',1); end
                    elapsed(trial)=toc(timer)/jobs;
                end
                err=0; passed=true;
                for k=1:jobs
                    [ok,difference]=equivalent(values{k},reference,opts);
                    passed=passed && ok; err=max(err,difference);
                end
                candidate(end+1)=entry('mex-parallel',1,median(elapsed),err,passed);
                H.mexParallelValidated=passed;
            catch issue
                H.notes{end+1}=['Parallel MEX benchmark failed; preserving pool: ',issue.message];
            end
        end
    catch issue
        H.notes{end+1}=['CPU pool unavailable or benchmark failed: ',issue.message];
    end
end
if any(strcmp(opts.mode,{'auto','gpu'})) && hasPCT && exist('gpuDevice','file')~=0
    try
        device=gpuDevice;
        H.gpuAvailable=true; H.gpuName=device.Name;
        H.gpuAvailableBytes=double(device.AvailableMemory);
        H.gpuComputeCapability=device.ComputeCapability;
        for batch=[4 16 32 64]
            [q,info]=bngr_grid(S,testTs,'gpu',batch);
            [passed,err]=equivalent(q,reference,opts);
            elapsed=zeros(1,opts.trials);
            for k=1:opts.trials
                wait(device); timer=tic;
                bngr_grid(S,testTs,'gpu',batch);
                wait(device); elapsed(k)=toc(timer);
            end
            candidate(end+1)=entry('gpu-client',info.ActualBatchSize,median(elapsed),err,passed);
        end
    catch issue
        H.notes{end+1}=['GPU unavailable or benchmark failed: ',issue.message];
    end
end
H.benchmark=candidate;
eligible=[candidate.passed];
if strcmp(opts.mode,'gpu')
    eligible=eligible & strcmp({candidate.mode},'gpu-client');
    if ~any(eligible)
        H.notes{end+1}='Requested GPU failed availability/equivalence checks; falling back to CPU.';
        eligible=[candidate.passed] & ~strcmp({candidate.mode},'gpu-client');
    end
elseif strcmp(opts.mode,'cpu')
    eligible=eligible & ~strcmp({candidate.mode},'gpu-client');
elseif strcmp(opts.mode,'mex')
    eligible=eligible & startsWith({candidate.mode},'mex-');
    if ~any(eligible)
        H.notes{end+1}='Requested MEX failed availability/equivalence checks; falling back to MATLAB CPU.';
        eligible=[candidate.passed] & startsWith({candidate.mode},'cpu-');
    end
end
indices=find(eligible); [~,best]=min([candidate(indices).secondsPerJob]);
chosen=candidate(indices(best)); H.selectedMode=chosen.mode; H.batchSize=chosen.batchSize;
H.backend='cpu'; H.parallelSubsets=endsWith(chosen.mode,'-parallel');
if strcmp(chosen.mode,'gpu-client'), H.backend='gpu'; end
if startsWith(chosen.mode,'mex-'), H.backend='mex'; end
H.measuredSecondsPerJob=chosen.secondsPerJob;
H.useMEXForRefinement=H.mexValidated && (~H.parallelRefinement || H.mexParallelValidated);
H.notes{end+1}='Existing parallel pools are neither resized nor deleted; no GPU reset is performed.';
H.notes{end+1}='Pool creation and GPU initialization are excluded from steady-state timings.';
H.notes{end+1}='Timing includes host/device transfers. Grid timings cannot guarantee fastest whole-fit runtime.';
H.notes{end+1}='Optional MEX is enabled only after a local equivalence check; every MEX call uses one native thread.';
H.notes{end+1}='GPU client grids and CPU parallel refinement may run as two sequential stages; workers never invoke GPU kernels.';
fprintf('BNGR selected %s, double, batch %d, %.4g s per representative grid.\n', ...
    H.selectedMode,H.batchSize,H.measuredSecondsPerJob);
end

function C=entry(mode,batch,seconds,err,passed)
C=struct('mode',mode,'batchSize',batch,'secondsPerJob',seconds, ...
    'maxAbsoluteError',err,'passed',passed);
end

function [passed,err]=equivalent(actual,reference,opts)
delta=abs(actual(:)-reference(:)); err=max(delta);
passed=all(isfinite(actual(:))) && all(delta<= ...
    opts.absoluteTolerance+opts.relativeTolerance*abs(reference(:)));
end

function varargout=bngr_method(action,varargin)

switch action
    case 'options', varargout{1}=model_options();
    case 'fit', varargout{1}=fit_models(varargin{:});
    case 'predict', [varargout{1:nargout}]=predict_response(varargin{:});
    case 'pairs', varargout{1}=prepare_pairs(varargin{:});
    otherwise, error('BNGR:Action','Unknown method action.');
end
end

function C = model_options()

C.useParallel = false;
C.prior1 = 50;
C.prior2 = 50;
C.logLower = -46.051701859880914;
C.gridSize = 385;
C.optTol = 1e-9;
C.quadRelTol = 2e-8;
C.quadAbsTol = 1e-14;
C.errorLimit = 2e-7;
C.z95 = 1.959963984540054;
C.hessianRelativeSteps = [1e-3,5e-4,2.5e-4];
C.hessianCheckTolerance = 5e-3;
C.hessianStationarityTolerance = 1e-4;
C.variancePriorTailTolerance = 1e-8;
C.laplaceLogTolerance = log(1.01);
end

function model = fit_models(X,H,C,hardware,gridEvaluator)
assert(nargin>=5&&isa(gridEvaluator,'function_handle'));
if isempty(C), C=model_options(); end
useGPU=strcmp(hardware.backend,'gpu');
C.useParallel=hardware.parallelRefinement;
C.useMEX=hardware.useMEXForRefinement;
precomputeGrid=useGPU || ~hardware.parallelSubsets;
H=H(:);
assert(size(X,1)==numel(H) && all(isfinite(X(:))) && all(isfinite(H)), ...
    'Invalid data.');
model.xMean=mean(X,1);
model.xStd=std(X,0,1);
model.xStd(model.xStd==0)=1;
model.etaMean=mean(H);
model.etaStd=std(H);
assert(model.etaStd>0,'Response variance must be positive.');
Xi=((X-model.xMean)./model.xStd)';
eta=(H-model.etaMean)/model.etaStd;
N=size(X,1); m=size(X,2); count=2^m-1;
DD=cell(1,m);
for j=1:m, DD{j}=((X(:,j)-X(:,j)')./model.xStd(j)).^2; end
tt=linspace(C.logLower,log(C.prior1),C.gridSize);
grid=zeros(count,0);
if precomputeGrid

    grid=zeros(count,C.gridSize);
    for mask=1:count
        D=zeros(N);
        for j=find(bitget(mask,1:m)), D=D+DD{j}; end
        S=prepare_pairs(D,eta);
        grid(mask,:)=gridEvaluator(S,tt,hardware.backend,hardware.batchSize);
    end
end
cellRecords=cell(count,1);
if C.useParallel
    parfor mask=1:count
        selected=find(bitget(mask,1:m));
        D=zeros(N);
        for j=selected, D=D+DD{j}; end
        if precomputeGrid
            initial=grid(mask,:);
        else
            S=prepare_pairs(D,eta);
            initial=gridEvaluator(S,tt,hardware.backend,hardware.batchSize);
        end
        cellRecords{mask}=fit_subset(D,eta,mask,C,initial,gridEvaluator);
    end
else
    for mask=1:count
        selected=find(bitget(mask,1:m));
        D=zeros(N);
        for j=selected, D=D+DD{j}; end
        if precomputeGrid
            initial=grid(mask,:);
        else
            S=prepare_pairs(D,eta);
            initial=gridEvaluator(S,tt,hardware.backend,hardware.batchSize);
        end
        cellRecords{mask}=fit_subset(D,eta,mask,C,initial,gridEvaluator);
    end
end
records=[cellRecords{:}];
T=struct2table(records);
T.Features=cell(count,1);
for k=1:count, T.Features{k}=mat2str(find(bitget(T.mask(k),1:m))); end
z=T.logEvidence; z=z-max(z); T.posterior=exp(z)/sum(exp(z));
[~,i]=max(T.logEvidence);
model.selected=records(i);
model.selectedFeatures=find(bitget(model.selected.mask,1:m));
model.Xi=Xi; model.eta=eta; model.X=X; model.H=H;
model.config=C; model.candidateCount=count; model.N=N; model.m=m;
model.allModels=T;
model.acceleration=hardware;
assert(abs(sum(T.posterior)-1)<1e-12 && height(T)==count,'Incomplete posterior.');
end

function r=fit_subset(D,eta,mask,C,initialGrid,gridEvaluator)

if nargin<5, initialGrid=[]; end
S=prepare_pairs(D,eta);
tt=linspace(C.logLower,log(C.prior1),C.gridSize);
cachedT=[]; cachedQ=[];
if isempty(initialGrid)
    qq=objective(tt);
else
    assert(isequal(size(initialGrid),size(tt))&&all(isfinite(initialGrid)));
    qq=initialGrid; cachedT=tt; cachedQ=qq;
end
[~,index]=min(qq); tstar=tt(index); best=objective(tstar);
local=find(qq(2:end-1)<=qq(1:end-2)&qq(2:end-1)<=qq(3:end) ...
    &(qq(2:end-1)<qq(1:end-2)|qq(2:end-1)<qq(3:end)))+1;
options=optimset('TolX',C.optTol,'MaxFunEvals',2000,'MaxIter',1000,'Display','off');
brackets=[tt(1),tt(2);tt(end-1),tt(end)];
for k=local, brackets(end+1,:)=[tt(k-1),tt(k+1)]; end
for k=1:size(brackets,1)
    [t,q,flag]=fminbnd(@objective,brackets(k,1),brackets(k,2),options);
    assert(flag>0,'Bandwidth refinement did not converge (mask %d).',mask);
    if q<best, best=q; tstar=t; end
end
[best,dt,dtt]=sequential_residuals(S,tstar);
varphi1=exp(tstar); varphi2=min(C.prior2,exp(best)/S.n);
a=S.n/2-1;
curvatures=NaN(size(C.hessianRelativeSteps)); analytic=a*(dtt-dt);
for k=1:numel(C.hessianRelativeSteps)
    h=C.hessianRelativeSteps(k);
    if varphi1*(1+2*h)>=C.prior1, continue; end
    v=a*(objective(tstar+log1p(h*[-2,-1,1,2]))-best);
    curvatures(k)=(-v(4)+16*v(3)+16*v(2)-v(1))/(12*h^2);
end
Hessian=curvatures(end)/varphi1^2;
valid=all(isfinite(curvatures))&&all(curvatures>0)&&analytic>0 ...
    &&max(abs(curvatures-analytic))/max(abs(analytic),1e-12)<C.hessianCheckTolerance ...
    &&abs(curvatures(end)-curvatures(end-1))/max(abs(analytic),1e-12)<C.hessianCheckTolerance ...
    &&abs(a*dt)<=C.hessianStationarityTolerance*max(1,sqrt(max(0,analytic))) ...
    &&tt(2)<=tstar&&tstar<=tt(end-1);
laplace=NaN;
if valid
    laplace=laplace_evidence(best,S.logLambda,S.n,Hessian,C.prior1,C.prior2);
end
bound=prior_tail_log_bound(S,eta,tstar,best,C);
skip=false;
quadrature=NaN; errorEstimate=NaN; tail=NaN;
if ~skip
    points=unique([tt(1:4:end),tstar,tt(local)]);
    points=points(points>C.logLower&points<log(C.prior1));
    logfun=@(z) -a*(objective(z)-best)+z ...
        +log_gamma_upper(a,exp(objective(z))/(2*C.prior2));
    logs=logfun(tt);
    shift=max([logs,logfun(tstar)]);
    maxima=find(logs(2:end-1)>=logs(1:end-2)&logs(2:end-1)>=logs(3:end))+1;
    for k=maxima
        [~,value,flag]=fminbnd(@(z)-logfun(z),tt(k-1),tt(k+1),options);
        assert(flag>0,'Evidence scaling did not converge (mask %d).',mask);
        shift=max(shift,-value);
    end
    assert(isfinite(shift),'Nonfinite evidence scaling (mask %d).',mask);
    [value,err]=quadgk(@(z)exp(logfun(z)-shift),C.logLower,log(C.prior1), ...
        'Waypoints',points,'AbsTol',C.quadAbsTol,'RelTol',C.quadRelTol, ...
        'MaxIntervalCount',2000);
    assert(value>0&&isfinite(value)&&err/value<=C.errorLimit, ...
        'Evidence integration tolerance failed (mask %d).',mask);
    logvalue=shift+log(value); errorEstimate=err/value;
    constant=-S.n/2*log(2*pi)+sum(S.logLambda)/2-log(C.prior1)-log(C.prior2);
    quadrature=constant+gammaln(a)-a*(best-log(2))+logvalue;
    if valid
        [wide,wideError]=quadgk(@(z)exp(-a*(objective(z)-best)+z), ...
            C.logLower,log(C.prior1),'Waypoints',points, ...
            'AbsTol',C.quadAbsTol,'RelTol',C.quadRelTol,'MaxIntervalCount',2000);
        assert(wide>0&&isfinite(wide)&&wideError/wide<=C.errorLimit, ...
            'Variance-prior integration tolerance failed (mask %d).',mask);
        tail=max(0,-expm1(min(0,logvalue-log(wide))));
        valid=tail<=C.variancePriorTailTolerance;
    end
end
priorHessianValid=valid;
if valid, originalEvidence=laplace; else, originalEvidence=quadrature; end
accuracyError=abs(laplace-quadrature);
accuracyPassed=isfinite(laplace)&&isfinite(quadrature) ...
    &&accuracyError<=C.laplaceLogTolerance;
valid=priorHessianValid&&accuracyPassed;
if valid
    evidence=laplace; evidenceMethod="hessian_laplace"; fallbackReason="";
else
    evidence=quadrature; evidenceMethod="quadrature_fallback";
    if priorHessianValid, fallbackReason="laplace_accuracy_error";
    else, fallbackReason="original_rule_fallback"; end
end
assert(isfinite(evidence),'Nonfinite model evidence (mask %d).',mask);
r=struct('mask',mask,'varphi1',varphi1,'varphi2',varphi2,'logQ',best, ...
    'logEvidence',evidence,'logEvidenceLaplace',laplace, ...
    'logEvidenceQuadrature',quadrature,'Hessian',Hessian,'hessianUsed',valid, ...
    'integrationError',errorEstimate,'LaplaceMinusQuadrature',laplace-quadrature, ...
    'VariancePriorLogBound',bound,'PriorWidthBoundUsed',skip, ...
    'VariancePriorTailEstimate',tail,'HessianPriorValid',priorHessianValid, ...
    'LaplaceAccuracyPassed',accuracyPassed, ...
    'LaplaceAccuracyFallback',priorHessianValid&&~accuracyPassed, ...
    'LaplaceAccuracyAbsLogError',accuracyError, ...
    'LaplaceAccuracyLogTolerance',C.laplaceLogTolerance, ...
    'logEvidenceOriginalRule',originalEvidence,'evidenceMethod',evidenceMethod, ...
    'hessianFallbackReason',fallbackReason);
    function q=objective(t)
        shape=size(t); values=t(:)';
        [known,where]=ismember(values,cachedT);
        q=zeros(size(values)); q(known)=cachedQ(where(known));
        if any(~known)
            fresh=values(~known);
            if isfield(C,'useMEX') && C.useMEX
                result=gridEvaluator(S,fresh,'mex',32);
            elseif isscalar(fresh), result=sequential_residuals(S,fresh);
            else, result=bandwidth_residuals(S,fresh); end
            q(~known)=result;
            cachedT=[cachedT,fresh]; cachedQ=[cachedQ,result];
        end
        q=reshape(q,shape);
    end
end

function bound=prior_tail_log_bound(S,eta,t,logQ,C)

a=S.n/2-1; lower=cummin(eta); upper=cummax(eta);
lower=lower(1:end-1); upper=upper(1:end-1);
peak=max(abs(eta(2:end)-lower),abs(eta(2:end)-upper));
L=2/exp(1)*sum(S.lambdaScaled.*peak.*(upper-lower).*(0:S.n-1)');
qscaled=exp(logQ-S.lambdaShift); bound=Inf;
if ~(a>0&&qscaled>0&&C.logLower<t&&t<log(C.prior1)), return; end
delta=min([1e-3,(t-C.logLower)/2,(log(C.prior1)-t)/2]);
if L>0, delta=min(delta,qscaled/(2*a*L)); end
if ~(delta>0&&isfinite(delta)), return; end
logLower=log(2)+t+log(sinh(delta))-a*log1p(L*delta/qscaled);
bound=log(C.prior1)+a*(logQ-log(2*C.prior2))-gammaln(a+1)-logLower;
end

function S = prepare_pairs(D, eta)

eta = eta(:); N = numel(eta);
assert(N >= 5 && isequal(size(D),[N N]), 'Invalid training arrays.');
assert(all(isfinite(D(:))) && all(D(:)>=0), 'Distances must be finite and nonnegative.');
[j,i] = find(triu(true(N),1));
g = i-1; dist = D(sub2ind([N N],i,j));
S.n = N-1;
S.G = sparse(g,(1:numel(g))',1,S.n,numel(g));
meanDist = (S.G*dist)./(1:S.n)';
den = 2*meanDist(g);
rowMin = accumarray(g,dist,[S.n 1],@min);
centeredDist = dist-rowMin(g);
S.h = zeros(size(dist)); positive = den>0;
S.h(positive) = centeredDist(positive)./den(positive);
z = -2*dist;
rowMax = accumarray(g,z,[S.n 1],@max);
S.logLambda = rowMax+log(S.G*exp(z-rowMax(g)));
S.lambdaShift = max(S.logLambda);
S.lambdaScaled = exp(S.logLambda-S.lambdaShift);
S.previousEta = eta(j);
S.eta = eta(2:end);
end

function [logQ, dlogQ, ddlogQ, etaHat] = sequential_residuals(S, logVarphi1)

k = S.h*exp(-logVarphi1);
w = exp(-k);
den = S.G*w;
etaHat = (S.G*(w.*S.previousEta))./den;
e = S.eta-etaHat;
ss = sum(S.lambdaScaled.*e.^2);
assert(isfinite(ss) && ss>0, 'Degenerate residual sum: evidence undefined.');
logQ = S.lambdaShift+log(ss);
if nargout>1
    wk = w.*k;
    u = (S.G*wk)./den;
    dp = (S.G*(wk.*S.previousEta))./den-etaHat.*u;
    wk2 = wk.*k;
    ddp = (S.G*(wk2.*S.previousEta))./den ...
        -etaHat.*((S.G*wk2)./den)-(2*u+1).*dp;
    qt = -2*sum(S.lambdaScaled.*e.*dp);
    qtt = 2*sum(S.lambdaScaled.*(dp.^2-e.*ddp));
    dlogQ = qt/ss;
    ddlogQ = qtt/ss-(qt/ss)^2;
end
end

function q=bandwidth_residuals(S,t)

shape=size(t); t=t(:)'; q=zeros(size(t));
for first=1:64:numel(t)
    ids=first:min(first+63,numel(t));
    w=exp(-S.h.*exp(-t(ids)));
    prediction=(S.G*(w.*S.previousEta))./(S.G*w);
    residual=S.eta-prediction;
    ss=sum(S.lambdaScaled.*residual.^2,1);
    assert(all(isfinite(ss)&ss>0),'Degenerate conditional residual sum.');
    q(ids)=S.lambdaShift+log(ss);
end
q=reshape(q,shape);
end

function result=log_gamma_upper(a,x)

assert(isscalar(a)&&isfinite(a)&&a>0,'Gamma shape must be positive.');
result=zeros(size(x));
for index=1:numel(x)
    z=x(index);
    if z<=0, result(index)=0; continue; end
    value=gammainc(z,a,'upper');
    if value>1e-250, result(index)=log(value); continue; end
    if isinf(z), result(index)=-Inf; continue; end
    assert(isfinite(z)&&z>a,'Invalid upper-Gamma continuation.');
    tiny=1e-300; b=z+1-a; c=1/tiny; d=1/b; h=d;
    converged=false;
    for k=1:10000
        an=-k*(k-a); b=b+2; d=an*d+b; c=b+an/c;
        if abs(d)<tiny
            if d<0, d=-tiny; else, d=tiny; end
        end
        if abs(c)<tiny
            if c<0, c=-tiny; else, c=tiny; end
        end
        d=1/d; change=d*c; h=h*change;
        if abs(change-1)<2e-15, converged=true; break; end
    end
    assert(converged&&h>0,'Upper-Gamma continued fraction did not converge.');
    result(index)=-z+a*log(z)-gammaln(a)+log(h);
end
end

function z=laplace_evidence(logQ,logLambda,n,Hessian,sigma1,sigma2)

assert(n>2&&numel(logLambda)==n);
assert(isfinite(Hessian)&&Hessian>0,'Laplace requires positive curvature.');
a=n/2-1;
z=-log(2)+gammaln(a)-log(sigma1)-log(sigma2)-n/2*log(pi) ...
    +0.5*(log(2*pi)+sum(logLambda)-log(Hessian))-a*logQ;
end

function [prediction,sd,interval95]=predict_response(model,Xquery)
selected=model.selectedFeatures;
prediction=zeros(size(Xquery,1),1);
sd=zeros(size(prediction));
for i=1:size(Xquery,1)
    D=zeros(size(model.X,1),1);
    for j=selected
        D=D+((model.X(:,j)-Xquery(i,j))./model.xStd(j)).^2;
    end
    omegaSq=model.selected.varphi1*mean(D);
    scale=2*omegaSq;
    if scale==0, scale=1; end
    w=exp(-(D-min(D))/scale); w=w/sum(w);
    prediction(i)=w'*model.H;
    a=-2*D; maximum=max(a);
    logLambda=maximum+log(sum(exp(a-maximum)));
    sd(i)=exp(0.5*(log(model.selected.varphi2)-logLambda))*model.etaStd;
end
assert(all(isfinite(sd)),'Prediction uncertainty exceeds double precision range.');
interval95=[prediction-model.config.z95*sd,prediction+model.config.z95*sd];
end

