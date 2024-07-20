close all, 
clear all, 
clc

%% Parametri
% set date
ratioVal = 0.15;  % procent imagini validare
ratioTest = 0.15; % procent imagini testare

% antrenare 
NEP = 100; % numar epoci
MBS = 20; % numar exemple in minilot (o iteratie)

% arhitectura
NH = 50; % numar neuroni ascunsi, mai mare

%% Fisiere rezultate
filenameRez = ['RezMLP_', num2str(NH), '_', num2str(NEP), '.mat'];
filenameDiary = ['RezMLP_', num2str(NH), '_', num2str(NEP), '.txt'];

diary(filenameDiary) % comanda suspecta!

%% Set date 
syntheticDir = '../dataset';

% creare obiect datastore
imds = imageDatastore(syntheticDir, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');

% verificare distribuție etichete
countEachLabel(imds)

% impartire set date pe antrenare, validare, testare
[imdsTrain, imdsVal, imdsTest] = splitEachLabel(imds, 1 - ratioVal - ratioTest, ratioVal, ratioTest, 'randomized');

% verificare distribuție după împărțire
countEachLabel(imdsTrain)
countEachLabel(imdsVal)
countEachLabel(imdsTest)

% preprocesare imagini - format nivele de gri, redimensionare - vector
img = readimage(imds, 1);
imgSize = size(img);
targetSize = [imgSize(1), imgSize(2), 1]; % păstrează dimensiunea originală dar convertește în nivele de gri

imdsTrain.ReadFcn = @(loc) preprocessImage(loc, targetSize);
imdsVal.ReadFcn = @(loc) preprocessImage(loc, targetSize);
imdsTest.ReadFcn = @(loc) preprocessImage(loc, targetSize);

%% MLP - arhitectura
labels = unique(imds.Labels);
numClasses = length(labels);

layers = [
    imageInputLayer(targetSize)
    fullyConnectedLayer(NH)
    reluLayer % înlocuit tanh cu relu pentru performanță mai bună
    fullyConnectedLayer(NH) % adăugat un strat ascuns suplimentar
    reluLayer % înlocuit tanh cu relu
    fullyConnectedLayer(numClasses) % ponderi initializate cu valori aleatoare
    softmaxLayer 
    classificationLayer];

%% MLP - antrenare
options = trainingOptions('sgdm', ...
    'MiniBatchSize', MBS, ...            
    'MaxEpochs', NEP, ...      
    'InitialLearnRate', 1e-3, ... % ajustată rata de învățare
    'ValidationData', imdsVal, ...
    'Verbose', false, ...
    'Plots', 'training-progress');

net = trainNetwork(imdsTrain, layers, options);

%% Rezultate - antrenare
YTrain_net = classify(net, imdsTrain);
AccTrain = mean(YTrain_net == imdsTrain.Labels);
disp(['Acuratețe antrenare: ', num2str(AccTrain)])

%% Rezultate - testare
YTest_net = classify(net, imdsTest);
AccTest = mean(YTest_net == imdsTest.Labels);
disp(['Acuratețe testare: ', num2str(AccTest)])

%% Rezultate - validare
YVal_net = classify(net, imdsVal);
AccVal = mean(YVal_net == imdsVal.Labels);
disp(['Acuratețe validare: ', num2str(AccVal)])

%% Rezultate - salvare
save(filenameRez, 'net');
diary off


