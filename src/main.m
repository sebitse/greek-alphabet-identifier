close all, 
clear all, 
clc

%% Parametri
% setare date
ratioVal = 0.15;  % procent imagini validare
ratioTest = 0.15; % procent imagini testare

% antrenare
NEP = 100; % număr epoci
MBS = 10;  % număr exemple în mini-lot (o iterație)

% arhitectură
NH = 50;   % număr neuroni ascunși

%% Fișiere rezultate
filenameRez = ['RezMLP_', num2str(NH), '_', num2str(NEP), '.mat'];
filenameDiary = ['RezMLP_', num2str(NH), '_', num2str(NEP), '.txt'];

diary(filenameDiary)

%% Set date

syntheticDir = '../dataset';

% crearea obiectului datastore
imds = imageDatastore(syntheticDir, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');

% împărțirea setului de date în antrenare, validare și testare
[imdsTest, imdsVal, imdsTrain] = splitEachLabel(imds, ratioTest, ratioVal, 1 - ratioTest - ratioVal, 'randomized');

% preprocesare imagini - conversie la nivele de gri, redimensionare - vector
imgSize = size(read(imds));
targetSize = [1 imgSize(1) * imgSize(2)]; % 1 exemplu = vector linie


imdsValResized = transform(imdsVal, @(x) imresize(im2gray(x), targetSize));
imdsTrainResized = transform(imdsTrain, @(x) imresize(im2gray(x), targetSize));
imdsTestResized = transform(imdsTest, @(x) imresize(im2gray(x), targetSize));

imdsTestL = arrayDatastore(imdsTest.Labels);
imdsTrainL = arrayDatastore(imdsTrain.Labels);
imdsValL = arrayDatastore(imdsVal.Labels);

imdsValResizedFinal = combine(imdsValResized, imdsValL);
imdsTrainResizedFinal = combine(imdsTrainResized, imdsTrainL);
imdsTestResizedFinal = combine(imdsTestResized, imdsTestL);

imgAfterTransform = read(imdsValResized);
% figure
% imshow(imgAfterTransform, [])

%% MLP - arhitectură
labels = unique(imds.Labels);
numClasses = length(labels);

if numClasses ~= 24
    error('Numar de clase gresit :(');
end

layers = [
    imageInputLayer(targetSize)
    fullyConnectedLayer(NH)
    tanhLayer
    fullyConnectedLayer(NH)
    tanhLayer    
    fullyConnectedLayer(numClasses)
    softmaxLayer 
    classificationLayer
];

%% MLP - antrenare
options = trainingOptions('sgdm', ...
    'MiniBatchSize', MBS, ...
    'MaxEpochs', NEP, ...
    'InitialLearnRate', 1e-3, ...
    'ValidationData', imdsValResizedFinal, ...
    'Verbose', false, ...
    'Plots', 'training-progress');

net = trainNetwork(imdsTrainResizedFinal, layers, options);

%% Rezultate - antrenare
YTrain_net = classify(net, imdsTrainResized);
AccTrain = mean(YTrain_net == imdsTrainResized.UnderlyingDatastores{1, 1}.Labels)

%% Rezultate - testare
YTest_net = classify(net, imdsTestResized);
AccTest = mean(YTest_net == imdsTest.Labels)

%% Rezultate - validare
YVal_net = classify(net, imdsValResized);
AccVal = mean(YVal_net == imdsVal.Labels)

%% Rezultate - salvare
feval(@save, filenameRez, 'net');
diary off
