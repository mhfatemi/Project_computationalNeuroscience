
clear all
close all

% Load ECoG and Motion Data

subID = "20090525S1_K1";

Path = 'G:\simulation\data\20090525S1_Food-Tracking_K1_Zenas+Chao_mat_ECoG64-Motion8\data\20090525S1_FTT_K1_ZenasChao_mat_ECoG64-Motion8\';

% Load ECoG data
for ch = 1:64
    ch_file_name = [Path,'ECoG_ch', num2str(ch),'.mat'];
    loaded_data=load(ch_file_name);
    data_name = ['ECoGData_ch', num2str(ch)];
    electrode_data = loaded_data.(data_name);
    ECoGData(ch,:) = electrode_data;
end
ECoGData = double(ECoGData);
load([Path,'ECoG_time.mat'])

indx = find(ECoGTime>=0 & ECoGTime<=900);
sM = indx(1,1);
eM   = indx(1,end);
ECoGData = ECoGData(:,sM:eM);
%%  Load Motion Data
load([Path,'Motion.mat'])

LWRI = MotionData{3, 1};
RWRI = MotionData{6, 1};

LELB = MotionData{2, 1};
RELB = MotionData{5, 1};

LSHO = MotionData{1, 1};
RSHO = MotionData{4, 1};

body_contered = (RSHO + LSHO)/2; 

Wrist = RWRI - body_contered; 

Xnew = RSHO - LSHO; 

THETA = zeros(1,size(Xnew,2));

for i = 1:size(Xnew,1)
    THETA(1,i) = atan(Xnew(i,2)/Xnew(i,1));
end

Rw = zeros(size(Wrist,1), 3);

for i=1:size(Wrist,1)

    Rw(i,1) = Wrist(i,1)*cos(THETA(1,i))+Wrist(i,2)*sin(THETA(1,i));
    Rw(i,2) = -Wrist(i,1)*sin(THETA(1,i))+Wrist(i,2)*cos(THETA(1,i));

end

Rw(:,3) = Wrist(:,3);





indx_m = find(MotionTime>=0 & MotionTime<=900);

MotionTime_new = MotionTime(indx_m);

sM = indx_m(1,1);
eM = indx_m(1,end);

RWRI_ref = Rw(sM:eM,:);

% up sample position signal
time_new = linspace(MotionTime_new(1),MotionTime_new(end),size(ECoGData,2));
for i = 1:3

    nLWRInew(:,i) = interp1(MotionTime_new,RWRI_ref(:,i),time_new);

end

% Normalize Motion data (z-Score)
for i = 1:size(nLWRInew,2)
    nRWRI(:,i) = (nLWRInew(:,i) - mean(nLWRInew(:,i)))/std(nLWRInew(:,i));
end


%% Preprocessing ECoG Data ---------------------------

% Band Pass filter  

Fs = 1000;
fl = 0.1;
fh = 499;
order = 4;
wn= [fl fh]/(Fs/2);
type= 'bandpass';
[b,a]= butter(order,wn,type);

for ch = 1:size(ECoGData,1)
    filtered_ECoG(ch,:) = filtfilt(b,a,ECoGData(ch, :));
end



% CAR Filter

for i = 1:size(filtered_ECoG,2)
    mu = mean(filtered_ECoG(:, i));
    filteredcar_ECoG(:,i) = filtered_ECoG(:,i) - mu;
end

%% Feature Extraction
down_sample_rate = 4;
d_filteredcar_ECoG = filteredcar_ECoG(:,1:down_sample_rate:end);


Timestamp = [1:275];
WAVE = 7;
FMIN = 10/250;
FMAX = 120/250;
N = 10;
TRACE = 0;
wRange = 275; %10;
wStep = 25;

feature_vec = {};
feature_temp =[];
window = 0;

for rng = 1:wStep:size(d_filteredcar_ECoG,2)-wRange
    window = window+1

    sW = rng ;
    eW =(rng+wRange-1);
    feature_temp = [];

    sig = d_filteredcar_ECoG(:,sW:eW);
    for ch = 1: size(d_filteredcar_ECoG,1)

        [TFR,T,F,WT] = tfrscalo(sig(ch,:)',Timestamp,WAVE,FMIN,FMAX,N,TRACE);
        Scalogram = TFR(:,26:25:end);

        for i = 1:size(Scalogram,2)
            mu_fr = mean(Scalogram(:,i));
            sd_fr = std(Scalogram(:,i));
            nScal(:, i) = (Scalogram(:, i) - mu_fr)/sd_fr;
        end

        Scalog(:,:,ch) = nScal;


    end
    Wavelet_feature(:,:,:,1) = Scalog;
    feature_vec{window,1} = Wavelet_feature;

end


clear temp
clear filteredcar_ECoG
clear ECoGData
%% Divide Feature Data to Test and train set
x = feature_vec;
y_c = [nRWRI(1:100:end,:)];
y_c(1:11,:) = [];

feature_vec = x ;
y = num2cell(y_c);
ndLWRI = y;

div = 0.75;
numel = floor(div*size(x,1));

Train_Feature = feature_vec(1:numel,1);
Test_Feature = feature_vec(numel+1:end,1);

Train_Motion_X = ndLWRI(1:numel,1);
Train_Motion_Y = ndLWRI(1:numel,2);
Train_Motion_Z = ndLWRI(1:numel,3);

Test_Motion_X = ndLWRI(numel+1:end,1);
Test_Motion_Y = ndLWRI(numel+1:end,2);
Test_Motion_Z = ndLWRI(numel+1:end,3);

%% Structure of LSTM
numHiddenUnits = 125;
numResponses = 1;
inputSize = [10 10 64 1];


layers = [ ...
    sequenceInputLayer(inputSize,'Name','input')
    flattenLayer('Name','flatten')
    bilstmLayer(numHiddenUnits,'OutputMode','sequence')
    bilstmLayer(numHiddenUnits,'OutputMode','sequence')
    %lstmLayer(numHiddenUnits,'OutputMode','sequence')
    %lstmLayer(numHiddenUnits,'OutputMode','sequence')
    fullyConnectedLayer(numResponses)
    regressionLayer]

% layers = [
%     image3dInputLayer([10 10 64 1])
%     convolution3dLayer(2,20)
%     batchNormalizationLayer('Name','bn')
%     reluLayer('Name','relu')
% 
%     sequenceUnfoldingLayer('Name','unfold')
% 
%     flattenLayer('Name','flatten')
% 
%     lstmLayer(numHiddenUnits,'OutputMode','sequence')
%     fullyConnectedLayer(numResponses)
%     regressionLayer
%     ];


% ncom = 100;
% % Estimate X
% [XL,YL,XS,YS,BETA_X,PCTVAR,MSE_X,stats] = plsregress(Train_Feture,Train_Motion_X,ncom,'cv',10);
% X_predictedPLS = [ones(n,1) Train_Feture]*BETA_X;
% 
% [MAE, MSE,RMSE_X, NRMSE, R_X, R_squar] = Evaluations(Train_Motion_X,X_predictedPLS);
% 
% % Estimate Y
% [XL,YL,XS,YS,BETA_Y,PCTVAR,MSE_Y,stats] = plsregress(Train_Feture,Train_Motion_Y,ncom,'cv',10);
% Y_predictedPLS = [ones(n,1) Train_Feture]*BETA_Y;
% 
% [MAE, MSE,RMSE_Y, NRMSE, R_Y, R_squar] = Evaluations(Train_Motion_Y,Y_predictedPLS);
% 
% % Estimate Z
% [XL,YL,XS,YS,BETA_Z,PCTVAR,MSE_Z,stats] = plsregress(Train_Feture,Train_Motion_Z,ncom,'cv',10);
% Z_predictedPLS = [ones(n,1) Train_Feture]*BETA_Z;
% 
% [MAE, MSE,RMSE_Z, NRMSE, R_Z, R_squar] = Evaluations(Train_Motion_Z,Z_predictedPLS);

%
%% %% Structure of 3dCNN


%%
options = trainingOptions('rmsprop',...
    'InitialLearnRate',0.001, ...
    'MaxEpochs',10, ...
    'ExecutionEnvironment','cpu',...
    'Plots','training-progress');

netX = trainNetwork(Train_Feature,Train_Motion_X,layers,options);

netY = trainNetwork(Train_Feature,Train_Motion_Y,layers,options);
netZ = trainNetwork(Train_Feature,Train_Motion_Z,layers,options);

%% --------------------------------------------------------

X_predictedPLS = predict(netX,Test_Feature);
Y_predictedPLS = predict(netY,Test_Feature);
Z_predictedPLS = predict(netZ,Test_Feature);



%
% miniBatchSize  = 128;
% validationFrequency = floor(numel(YTrain)/miniBatchSize);
% options = trainingOptions('sgdm', ...
%     'MiniBatchSize',miniBatchSize, ...
%     'MaxEpochs',30, ...
%     'InitialLearnRate',1e-3, ...
%     'LearnRateSchedule','piecewise', ...
%     'LearnRateDropFactor',0.1, ...
%     'LearnRateDropPeriod',20, ...
%     'Shuffle','every-epoch', ...
%     'ValidationData',{XValidation,YValidation}, ...
%     'ValidationFrequency',validationFrequency, ...
%     'Plots','training-progress', ...
%     'Verbose',false);
%
X_predictedPLS = cell2mat(X_predictedPLS);
Y_predictedPLS = cell2mat(Y_predictedPLS);
Z_predictedPLS = cell2mat(Z_predictedPLS);


Test_Motion_X = cell2mat(Test_Motion_X);
Test_Motion_Y = cell2mat(Test_Motion_Y);
Test_Motion_Z = cell2mat(Test_Motion_Z);

[MAE_X, MSE_X,RMSE_X, NRMSE_X, R_X, R_squar_X] = Evaluations(Test_Motion_X,X_predictedPLS);
[MAE_Y, MSE_Y,RMSE_Y, NRMSE_Y, R_Y, R_squar_Y] = Evaluations(Test_Motion_Y,Y_predictedPLS);
[MAE_Z, MSE_Z,RMSE_Z, NRMSE_Z, R_Z, R_squar_Z] = Evaluations(Test_Motion_Z,Z_predictedPLS);

R_X
R_Y
R_Z

%%
T = (1:length(Test_Motion_X))/20;
figure
subplot(3,1,1)
plot(T,Test_Motion_X)
text(5,max(Test_Motion_X),['r = ', num2str(R_X)],'FontSize',10)
hold on
plot(T,X_predictedPLS,'r')


legend('Observed Trajectory', 'Predicted Trajectory')
ylabel({'X−position','(normalized)'});

subplot(3,1,2)
plot(T,Test_Motion_Y)
text(5,max(Test_Motion_Y),['r = ', num2str(R_Y)],'FontSize',10)
hold on
plot(T,Y_predictedPLS,'r')
%legend('Observed Trajectory', 'Predicted Trajectory')
ylabel({'Y−position','(normalized)'});

subplot(3,1,3)
plot(T,Test_Motion_Z)
text(5,max(Test_Motion_Z),['r = ', num2str(R_Z)],'FontSize',10)
hold on
plot(T,Z_predictedPLS,'r')

%legend('Observed Trajectory', 'Predicted Trajectory')

xlabel({'Time(Second)'});
ylabel({'Z−position','(normalized)'});



