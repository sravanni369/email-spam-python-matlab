% Original MATLAB implementation; same fixed source rows as Python.
% Source concepts: Abhishek Thakur, validation pp.23-25, metrics pp.37-39.
clear; clc;
diary matlab_run.log
unzip('data/spambase.zip','matlab_data');
A = readmatrix('matlab_data/spambase.data','FileType','text');
S = jsondecode(fileread('email_artifacts/split_source_indices.json'));
tr = S.train+1; va = S.valid+1; te = S.test+1;
assert(isempty(intersect(tr,te)) && isempty(intersect(tr,va)));
assert(isempty(intersect(va,te)) && size(A,2)==58);
assert(all(isfinite(A),'all') && all(ismember(A(:,58),[0 1])));
assert(isempty(intersect(A(tr,1:57),A(te,1:57),'rows')));
assert(isempty(intersect(A(tr,1:57),A(va,1:57),'rows')));
assert(isempty(intersect(A(va,1:57),A(te,1:57),'rows')));
mu = mean(A(tr,1:57)); sigma = std(A(tr,1:57),1);
sigma(sigma==0)=1;
X = (A(:,1:57)-mu)./sigma; y = A(:,58);
rng(42);
[model,info] = fitclinear(X(tr,:),y(tr),'Learner','logistic', ...
    'Regularization','ridge','Lambda',1/numel(tr),'Solver','lbfgs');
assert(all(info.TerminationCode>0),'Inspect optimizer termination before accepting results');
[~,v] = predict(model,X(va,:));
col = find(model.ClassNames==1); thresholds = .1:.01:.9;
f = zeros(size(thresholds));
for k=1:numel(thresholds)
    p=v(:,col)>=thresholds(k); t=y(va)==1;
    tp=sum(p&t); fp=sum(p&~t); fn=sum(~p&t);
    f(k)=1.25*tp/max(1.25*tp+.25*fn+fp,eps);
end
[~,best]=max(f); threshold=thresholds(best);
[~,scores] = predict(model,X(te,:)); p=scores(:,col)>=threshold;
t=y(te)==1; tp=sum(p&t); fp=sum(p&~t); fn=sum(~p&t); tn=sum(~p&~t);
precision=tp/max(tp+fp,1); recall=tp/max(tp+fn,1);
f1=2*tp/max(2*tp+fp+fn,1); accuracy=mean(p==t);
assert(tp+fp+fn+tn==numel(te));
R=struct('precision',precision,'recall',recall,'f1',f1, ...
    'accuracy',accuracy,'threshold',threshold,'counts',[tn fp fn tp]);
disp(R); disp(info.TerminationStatus);
fid=fopen('matlab_metrics.json','w'); fprintf(fid,'%s',jsonencode(R)); fclose(fid);
save('matlab_model.mat','model','mu','sigma','threshold');
writematrix([te-1,y(te),double(p),scores(:,col)],'matlab_predictions.csv');
figure('Color','w'); bar([precision recall f1 accuracy]*100);
xticklabels({'Precision','Recall','F1','Accuracy'}); ylim([0 100]);
ylabel('Percent'); title('MATLAB: held-out historical email benchmark');
exportgraphics(gcf,'matlab_results.png','Resolution',180);
fprintf('PASS: schema, labels, disjoint rows, feature overlap, convergence and counts.\n');
diary off
