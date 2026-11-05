function run_assignment(runClassification)
% Complete CA2 MATLAB implementation. No Statistics Toolbox required.
% Usage from repo root: addpath('matlab'); run_assignment
% Feature/plot-only run: run_assignment(false)
% Command line (low load): matlab -singleCompThread -batch "addpath('matlab'); run_assignment"
% Shared Python-generated plan fixes the same subsets and fold memberships.
% If the plan is absent, MATLAB generates seeded stratified folds itself.
if nargin < 1, runClassification = true; end
root = fileparts(fileparts(mfilename('fullpath')));
cfg = jsondecode(fileread(fullfile(root,'config.json')));
rng(cfg.seed,'twister');
out = fullfile(root,'artifacts','matlab');
if ~exist(out,'dir'), mkdir(out); end
diary(fullfile(out,'run.log'));
cleanup = onCleanup(@() diary('off')); %#ok<NASGU>
subjects = dir(fullfile(root,'Data','sub*'));
subjects = subjects([subjects.isdir]);
[~,order] = sort({subjects.name}); subjects=subjects(order);
assert(numel(subjects)==4,'Expected four top-level subject folders.');
fprintf('Top-level subject folders: %d\n',numel(subjects));
fprintf('Normal/Aggressive folders: %d\n',numel(subjects)*2);
fprintf('Nested Data/sub3/sub3 is excluded.\n');
categories = {'Normal','Aggressive'};
classNames = cell(20,1); classCategories = cell(20,1);
for category=1:2
    files=dir(fullfile(root,'Data',subjects(1).name,categories{category},'txt','*.txt'));
    [~,order]=sort({files.name}); files=files(order);
    assert(numel(files)==10,'Expected 10 actions per category.');
    for a=1:10
        label=(category-1)*10+a;
        [~,classNames{label}]=fileparts(files(a).name);
        classCategories{label}=categories{category};
    end
end
Nf=8*cfg.n_segments*2;
X=zeros(1200,Nf); y=zeros(1200,1); subject_id=y; trial_id=y;
featureNames=cell(1,Nf);
for c=1:8
    for s=1:cfg.n_segments
        j=((c-1)*cfg.n_segments+s-1)*2+1;
        featureNames{j}=sprintf('ch%d_seg%d_mean',c,s);
        featureNames{j+1}=sprintf('ch%d_seg%d_variance',c,s);
    end
end
manifest=cell(80,5); record=0; row=0;
for subject=1:4
    for category=1:2
        folder=fullfile(root,'Data',subjects(subject).name,categories{category},'txt');
        files=dir(fullfile(folder,'*.txt'));
        [~,order]=sort({files.name}); files=files(order);
        assert(numel(files)==10,'Expected 10 files.');
        for a=1:10
            label=(category-1)*10+a;
            [~,name]=fileparts(files(a).name);
            assert(strcmp(name,classNames{label}),'Class ordering mismatch.');
            signal=readmatrix(fullfile(folder,files(a).name),'FileType','text');
            fprintf('%s/%s/%s: %d x %d\n',subjects(subject).name,categories{category},files(a).name,size(signal));
            features=extract_trial_features(signal,cfg.n_trials,cfg.n_segments);
            rows=row+(1:cfg.n_trials);
            X(rows,:)=features; y(rows)=label; subject_id(rows)=subject;
            trial_id(rows)=(1:cfg.n_trials)'; row=row+cfg.n_trials;
            record=record+1;
            manifest(record,:)={subjects(subject).name,categories{category},files(a).name,size(signal,1),size(signal,2)};
            if subject==1 && any(cfg.binary_labels==label)
                fig=figure('Visible','off','Position',[50 50 950 700]);
                for c=1:8
                    subplot(4,2,c); plot(0:size(signal,1)-1,signal(:,c));
                    title(sprintf('%s: channel %d',name,c));
                    xlabel('Sample index'); ylabel('Stored units');
                end
                saveas(fig,fullfile(out,sprintf('raw_%s.png',name))); close(fig);
            end
        end
    end
end
assert(isequal(size(X),[1200 Nf]) && all(accumarray(y,1)==60));
data=[X y]; %#ok<NASGU>
save(fullfile(out,'emg_features.mat'),'X','y','subject_id','trial_id','data','classNames','featureNames');
writetable(cell2table(manifest,'VariableNames',{'Subject','Category','File','Rows','Channels'}),fullfile(out,'manifest.csv'));
writetable(table((1:20)',classNames,classCategories,'VariableNames',{'Label','Action','Category'}),fullfile(out,'class_labels.csv'));
fprintf('Combined input/target matrix: %d x %d\n',size(data));
fprintf('Two example feature vectors (subject 1, first interval):\n');
fig=figure('Visible','off');
for i=1:2
    idx=find(y==cfg.binary_labels(i)&subject_id==1,1);
    disp(classNames{cfg.binary_labels(i)}); disp(X(idx,:));
    for statistic=1:2
        subplot(1,2,statistic); hold on;
        plot(X(idx,statistic:2:end),'DisplayName',classNames{cfg.binary_labels(i)});
        xlabel('Channel/segment pair'); legend('show');
        if statistic==1, ylabel('Mean'); else, ylabel('Sample variance'); end
    end
end
saveas(fig,fullfile(out,'example_features.png')); close(fig);
referencePath=fullfile(root,'artifacts','emg_features.mat');
if exist(referencePath,'file')
    reference=load(referencePath);
    assert(isequal(size(X),size(reference.X)) && isequal(y,double(reference.y)),'Reference shape/labels mismatch.');
    relativeError=max(abs(X(:)-reference.X(:))./max(1,abs(reference.X(:))));
    assert(relativeError<1e-9,'MATLAB/Python feature mismatch.');
    fprintf('MATLAB/Python maximum relative feature difference: %.3g\n',relativeError);
end
if ~runClassification, return; end
planPath=fullfile(root,'artifacts','experiment_plan.mat');
if exist(planPath,'file')
    plan=load(planPath);
    assert(size(plan.fold_ids,3)==cfg.n_mc && size(plan.selected_labels,2)==cfg.n_combinations,'Plan/config mismatch.');
else
    plan=makePlan(y,cfg);
end
accuracy=zeros(18,cfg.n_combinations,cfg.n_mc);
for mi=1:18
    M=mi+1;
    for ci=1:cfg.n_combinations
        labels=double(squeeze(plan.selected_labels(mi,ci,1:M)));
        mask=ismember(y,labels);
        for mc=1:cfg.n_mc
            folds=double(squeeze(plan.fold_ids(mi,ci,mc,mask)));
            accuracy(mi,ci,mc)=cvKnn(X(mask,:),y(mask),folds,cfg.n_neighbors);
            pause(cfg.pause_seconds);
        end
        pause(cfg.batch_pause_seconds);
    end
    fprintf('M=%d: mean accuracy %.2f%%\n',M,100*mean(reshape(accuracy(mi,:,:),[],1)));
    save(fullfile(out,'checkpoint.mat'),'accuracy','mi');
end
binaryMask=ismember(y,cfg.binary_labels);
binaryScores=zeros(cfg.n_mc,1); fullScores=binaryScores;
binaryConfusion=zeros(2); fullConfusion=zeros(20);
for mc=1:cfg.n_mc
    [binaryScores(mc),pred]=cvKnn(X(binaryMask,:),y(binaryMask),double(plan.binary_fold_ids(mc,binaryMask))',cfg.n_neighbors);
    for i=1:2
        for j=1:2
            binaryConfusion(i,j)=binaryConfusion(i,j)+sum(y(binaryMask)==cfg.binary_labels(i)&pred==cfg.binary_labels(j));
        end
    end
    [fullScores(mc),pred]=cvKnn(X,y,double(plan.full_fold_ids(mc,:))',cfg.n_neighbors);
    fullConfusion=fullConfusion+accumarray([y pred],1,[20 20]);
    pause(cfg.batch_pause_seconds);
end
losoScores=zeros(4,1);
for subject=1:4
    test=subject_id==subject;
    pred=knnPredict(X(~test,:),y(~test),X(test,:),cfg.n_neighbors);
    losoScores(subject)=mean(pred==y(test));
end
save(fullfile(out,'classification_results.mat'),'accuracy','binaryScores','fullScores','binaryConfusion','fullConfusion','losoScores','plan');
meanAccuracy=mean(mean(accuracy,3),2);
stdSubsets=std(mean(accuracy,3),0,2);
writetable(table((2:19)',meanAccuracy,stdSubsets,'VariableNames',{'M','MeanAccuracy','SDSubsetMeans'}),fullfile(out,'accuracy_curve.csv'));
fig=figure('Visible','off');
imagesc(2:19,1:cfg.n_mc,100*squeeze(mean(accuracy,2))'); axis xy; colorbar;
xlabel('Number of classes M'); ylabel('Monte Carlo repeat'); title('Accuracy (%) averaged over 10 subsets');
saveas(fig,fullfile(out,'mc_heatmap.png')); close(fig);
fig=figure('Visible','off');
errorbar(2:19,100*meanAccuracy,100*stdSubsets,'o-'); hold on;
plot(2:19,100./(2:19),'--'); xlabel('Number of classes M'); ylabel('Accuracy (%)');
legend('Mean +/- SD across subset means','Balanced chance'); grid on;
saveas(fig,fullfile(out,'accuracy_vs_classes.png')); close(fig);
fig=figure('Visible','off'); imagesc(binaryConfusion); colorbar;
set(gca,'XTick',1:2,'YTick',1:2,'XTickLabel',classNames(cfg.binary_labels),'YTickLabel',classNames(cfg.binary_labels));
xlabel('Predicted'); ylabel('Actual'); title('Binary confusion: all repeats');
saveas(fig,fullfile(out,'binary_confusion.png')); close(fig);
drops=-diff(meanAccuracy); [largestDrop,idx]=max(drops);
firstRapid=find(drops>=0.05,1);
fprintf('Binary: %.2f%%; 20-class: %.2f%%; held-out-subject: %.2f%%\n',100*mean(binaryScores),100*mean(fullScores),100*mean(losoScores));
fprintf('Largest adjacent drop: M=%d to M=%d, %.2f percentage points.\n',idx+1,idx+2,100*largestDrop);
if isempty(firstRapid)
    fprintf('No adjacent drop reaches the stated 5-percentage-point rapid-drop criterion.\n');
else
    fprintf('First >=5-point drop ends at M=%d (descriptive, not a validated change point).\n',firstRapid+2);
end
end

function folds=stratifiedFolds(y,K)
folds=zeros(numel(y),1);
for label=unique(y)'
    indices=find(y==label); indices=indices(randperm(numel(indices)));
    folds(indices)=mod((0:numel(indices)-1)',K)+1;
end
end

function plan=makePlan(y,cfg)
plan.selected_labels=zeros(18,cfg.n_combinations,20);
plan.fold_ids=zeros(18,cfg.n_combinations,cfg.n_mc,numel(y),'int8');
for mi=1:18
    M=mi+1;
    combos=nchoosek(1:20,M); % At most nchoosek(20,10)=184756 rows.
    ranks=randi(size(combos,1),cfg.n_combinations,1); % With replacement, per handout.
    for ci=1:cfg.n_combinations
        labels=combos(ranks(ci),:); mask=ismember(y,labels);
        plan.selected_labels(mi,ci,1:M)=labels;
        for mc=1:cfg.n_mc
            plan.fold_ids(mi,ci,mc,mask)=reshape(stratifiedFolds(y(mask),cfg.n_folds),1,1,1,[]);
        end
    end
end
plan.binary_fold_ids=zeros(cfg.n_mc,numel(y),'int8');
plan.full_fold_ids=zeros(cfg.n_mc,numel(y),'int8');
mask=ismember(y,cfg.binary_labels);
for mc=1:cfg.n_mc
    plan.binary_fold_ids(mc,mask)=stratifiedFolds(y(mask),cfg.n_folds)';
    plan.full_fold_ids(mc,:)=stratifiedFolds(y,cfg.n_folds)';
end
end

function [accuracy,pred]=cvKnn(X,y,folds,k)
folds=folds(:); pred=zeros(size(y));
for fold=unique(folds)'
    test=folds==fold;
    pred(test)=knnPredict(X(~test,:),y(~test),X(test,:),k);
end
accuracy=mean(pred==y);
end

function pred=knnPredict(trainX,trainY,testX,k)
% Euclidean kNN, uniform votes; mode breaks voting ties at smallest label.
mu=mean(trainX,1); sigma=std(trainX,0,1); sigma(sigma==0)=1;
trainX=bsxfun(@rdivide,bsxfun(@minus,trainX,mu),sigma);
testX=bsxfun(@rdivide,bsxfun(@minus,testX,mu),sigma);
pred=zeros(size(testX,1),1);
% Per-query squared distances avoid a large all-pairs matrix and cancellation.
for q=1:size(testX,1)
    delta=bsxfun(@minus,trainX,testX(q,:));
    [~,indices]=sort(sum(delta.^2,2),'ascend');
    pred(q)=mode(trainY(indices(1:k)));
end
end
