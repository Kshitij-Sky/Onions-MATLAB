clc;
clear;
close all;
%% Initialize variables

Files_depth = dir('*depth.png');
Files_truth = dir('*truth.png');
Files_rgb = dir('*rgb.png');
numFiles = length(Files_depth);
circularity_onions = cell(1,numFiles);
eccentricity_onions = cell(1,numFiles);
solidity_onions = cell(1,numFiles);
extent_onions = cell(1,numFiles);
circularity_weeds = cell(1,numFiles);
eccentricity_weeds = cell(1,numFiles);
solidity_weeds = cell(1,numFiles);
extent_weeds = cell(1,numFiles);
contrast_onions_r = zeros(1,numFiles);
contrast_onions_g = zeros(1,numFiles);
contrast_onions_b = zeros(1,numFiles);
contrast_onions_d = zeros(1,numFiles);
correlation_onions_r = zeros(1,numFiles);
correlation_onions_g = zeros(1,numFiles);
correlation_onions_b = zeros(1,numFiles);
correlation_onions_d = zeros(1,numFiles);
energy_onions_r = zeros(1,numFiles);
energy_onions_g = zeros(1,numFiles);
energy_onions_b = zeros(1,numFiles);
energy_onions_d = zeros(1,numFiles);
homogeneity_onions_r = zeros(1,numFiles);
homogeneity_onions_g = zeros(1,numFiles);
homogeneity_onions_b = zeros(1,numFiles);
homogeneity_onions_d = zeros(1,numFiles);
contrast_weeds_r = zeros(1,numFiles);
contrast_weeds_g = zeros(1,numFiles);
contrast_weeds_b = zeros(1,numFiles);
contrast_weeds_d = zeros(1,numFiles);
correlation_weeds_r = zeros(1,numFiles);
correlation_weeds_g = zeros(1,numFiles);
correlation_weeds_b = zeros(1,numFiles);
correlation_weeds_d = zeros(1,numFiles);
energy_weeds_r = zeros(1,numFiles);
energy_weeds_g = zeros(1,numFiles);
energy_weeds_b = zeros(1,numFiles);
energy_weeds_d = zeros(1,numFiles);
homogeneity_weeds_r = zeros(1,numFiles);
homogeneity_weeds_g = zeros(1,numFiles);
homogeneity_weeds_b = zeros(1,numFiles);
homogeneity_weeds_d = zeros(1,numFiles);
RGB = cell(1,numFiles);
DEPTH = cell(1,numFiles);
TRUTH = cell(1,numFiles);

%% Read Images

for k = 1:numFiles
    disp('Processing image:');disp(k);

    RGB{k} = imread(Files_rgb(k).name);
    DEPTH{k} = imread(Files_depth(k).name);
    TRUTH{k} = imread(Files_truth(k).name);

    %% Compute shape and texture features
    
    onions_truth = logical(TRUTH{k}(:,:,1));
    weeds_truth = logical(TRUTH{k}(:,:,3));
    mask_onions_rgb = imoverlay(RGB{k},~onions_truth,'black');
    mask_weeds_rgb = imoverlay(RGB{k},~weeds_truth,'black');
    mask_onions_r = im2gray(mask_onions_rgb(:,:,1));
    mask_onions_g = im2gray(mask_onions_rgb(:,:,2));
    mask_onions_b = im2gray(mask_onions_rgb(:,:,3));
    mask_weeds_r = im2gray(mask_weeds_rgb(:,:,1));
    mask_weeds_g = im2gray(mask_weeds_rgb(:,:,2));
    mask_weeds_b = im2gray(mask_weeds_rgb(:,:,3));
    mask_onions_depth = im2gray(imoverlay(DEPTH{k},~onions_truth,'black'));
    mask_weeds_depth = im2gray(imoverlay(DEPTH{k},~weeds_truth,'black'));
    
    % Shape features
    stats_onions = regionprops('table', onions_truth,'Circularity','Solidity','Eccentricity','Extent');
    circularity_onions{k} = stats_onions.Circularity;
    eccentricity_onions{k} = stats_onions.Eccentricity;
    solidity_onions{k} = stats_onions.Solidity;
    extent_onions{k} = stats_onions.Extent;
    
    avg_circularity_onions(k) = mean(circularity_onions{1,k});
    avg_eccentricity_onions(k) = mean(eccentricity_onions{1,k});
    avg_solidity_onions(k) = mean(solidity_onions{1,k});
    avg_extent_onions(k) = mean(extent_onions{1,k});

    stats_weeds = regionprops('table', weeds_truth,'Circularity','Solidity','Eccentricity','Extent');
    circularity_weeds{k} = stats_weeds.Circularity;
    eccentricity_weeds{k} = stats_weeds.Eccentricity;
    solidity_weeds{k} = stats_weeds.Solidity;
    extent_weeds{k} = stats_weeds.Extent;

    avg_circularity_weeds(k) = mean(circularity_weeds{1,k});
    avg_eccentricity_weeds(k) = mean(eccentricity_weeds{1,k});
    avg_solidity_weeds(k) = mean(solidity_weeds{1,k});
    avg_extent_weeds(k) = mean(extent_weeds{1,k});

    % Texture features
    offsets = [0 1; -1 1;-1 0;-1 -1];
    glcms_onions_r = graycomatrix(mask_onions_r,'Offset',offsets);
    glcms_onions_r(1,1,:) = 0; % Set value of black to black as zero as we dont want to observe textures in the masked background
    glcms_onions_g = graycomatrix(mask_onions_g,'Offset',offsets);
    glcms_onions_g(1,1,:) = 0; 
    glcms_onions_b = graycomatrix(mask_onions_b,'Offset',offsets);
    glcms_onions_r(1,1,:) = 0; 
    glcms_onions_d = graycomatrix(mask_onions_depth,'Offset',offsets);
    glcms_onions_d(1,1,:) = 0; 

    glcms_weeds_r = graycomatrix(mask_weeds_r,'Offset',offsets);
    glcms_weeds_r(1,1,:) = 0; % Set value of black to black as zero as we dont want to observe textures in the masked background
    glcms_weeds_g = graycomatrix(mask_weeds_g,'Offset',offsets);
    glcms_weeds_g(1,1,:) = 0;
    glcms_weeds_b = graycomatrix(mask_weeds_b,'Offset',offsets);
    glcms_weeds_b(1,1,:) = 0;
    glcms_weeds_d = graycomatrix(mask_weeds_depth,'Offset',offsets);
    glcms_weeds_d(1,1,:) = 0;


    texts_onions_r_0 = graycoprops(glcms_onions_r(:,:,1));
    texts_onions_r_45 = graycoprops(glcms_onions_r(:,:,2));
    texts_onions_r_90 = graycoprops(glcms_onions_r(:,:,3));
    texts_onions_r_135 = graycoprops(glcms_onions_r(:,:,4));

    texts_onions_g_0 = graycoprops(glcms_onions_g(:,:,1));
    texts_onions_g_45 = graycoprops(glcms_onions_g(:,:,2));
    texts_onions_g_90 = graycoprops(glcms_onions_g(:,:,3));
    texts_onions_g_135 = graycoprops(glcms_onions_g(:,:,4));

    texts_onions_b_0 = graycoprops(glcms_onions_b(:,:,1));
    texts_onions_b_45 = graycoprops(glcms_onions_b(:,:,2));
    texts_onions_b_90 = graycoprops(glcms_onions_b(:,:,3));
    texts_onions_b_135 = graycoprops(glcms_onions_b(:,:,4));

    texts_onions_d_0 = graycoprops(glcms_onions_d(:,:,1));
    texts_onions_d_45 = graycoprops(glcms_onions_d(:,:,2));
    texts_onions_d_90 = graycoprops(glcms_onions_d(:,:,3));
    texts_onions_d_135 = graycoprops(glcms_onions_d(:,:,4));

    texts_weeds_r_0 = graycoprops(glcms_weeds_r(:,:,1));
    texts_weeds_r_45 = graycoprops(glcms_weeds_r(:,:,2));
    texts_weeds_r_90 = graycoprops(glcms_weeds_r(:,:,3));
    texts_weeds_r_135 = graycoprops(glcms_weeds_r(:,:,4));

    texts_weeds_g_0 = graycoprops(glcms_weeds_g(:,:,1));
    texts_weeds_g_45 = graycoprops(glcms_weeds_g(:,:,2));
    texts_weeds_g_90 = graycoprops(glcms_weeds_g(:,:,3));
    texts_weeds_g_135 = graycoprops(glcms_weeds_g(:,:,4));

    texts_weeds_b_0 = graycoprops(glcms_weeds_b(:,:,1));
    texts_weeds_b_45 = graycoprops(glcms_weeds_b(:,:,2));
    texts_weeds_b_90 = graycoprops(glcms_weeds_b(:,:,3));
    texts_weeds_b_135 = graycoprops(glcms_weeds_b(:,:,4));

    texts_weeds_d_0 = graycoprops(glcms_weeds_d(:,:,1));
    texts_weeds_d_45 = graycoprops(glcms_weeds_d(:,:,2));
    texts_weeds_d_90 = graycoprops(glcms_weeds_d(:,:,3));
    texts_weeds_d_135 = graycoprops(glcms_weeds_d(:,:,4));

    contrast_onions_r(k) = (texts_onions_r_0.Contrast + texts_onions_r_45.Contrast + texts_onions_r_90.Contrast + texts_onions_r_135.Contrast) / 4;
    contrast_onions_g(k) = (texts_onions_g_0.Contrast + texts_onions_g_45.Contrast + texts_onions_g_90.Contrast + texts_onions_g_135.Contrast) / 4;
    contrast_onions_b(k) = (texts_onions_b_0.Contrast + texts_onions_b_45.Contrast + texts_onions_b_90.Contrast + texts_onions_b_135.Contrast) / 4;
    contrast_onions_d(k) = (texts_onions_d_0.Contrast + texts_onions_d_45.Contrast + texts_onions_d_90.Contrast + texts_onions_d_135.Contrast) / 4;

    correlation_onions_r(k) = (texts_onions_r_0.Correlation + texts_onions_r_45.Correlation + texts_onions_r_90.Correlation + texts_onions_r_135.Correlation) / 4;
    correlation_onions_g(k) = (texts_onions_g_0.Correlation + texts_onions_g_45.Correlation + texts_onions_g_90.Correlation + texts_onions_g_135.Correlation) / 4;
    correlation_onions_b(k) = (texts_onions_b_0.Correlation + texts_onions_b_45.Correlation + texts_onions_b_90.Correlation + texts_onions_b_135.Correlation) / 4;
    correlation_onions_d(k) = (texts_onions_d_0.Correlation + texts_onions_d_45.Correlation + texts_onions_d_90.Correlation + texts_onions_d_135.Correlation) / 4;

    energy_onions_r(k) = (texts_onions_r_0.Energy + texts_onions_r_45.Energy + texts_onions_r_90.Energy + texts_onions_r_135.Energy) / 4;
    energy_onions_g(k) = (texts_onions_g_0.Energy + texts_onions_g_45.Energy + texts_onions_g_90.Energy + texts_onions_g_135.Energy) / 4;
    energy_onions_b(k) = (texts_onions_b_0.Energy + texts_onions_b_45.Energy + texts_onions_b_90.Energy + texts_onions_b_135.Energy) / 4;
    energy_onions_d(k) = (texts_onions_d_0.Energy + texts_onions_d_45.Energy + texts_onions_d_90.Energy + texts_onions_d_135.Energy) / 4;

    homogeneity_onions_r(k) = (texts_onions_r_0.Homogeneity + texts_onions_r_45.Homogeneity + texts_onions_r_90.Homogeneity + texts_onions_r_135.Homogeneity) / 4;
    homogeneity_onions_g(k) = (texts_onions_g_0.Homogeneity + texts_onions_g_45.Homogeneity + texts_onions_g_90.Homogeneity + texts_onions_g_135.Homogeneity) / 4;
    homogeneity_onions_b(k) = (texts_onions_b_0.Homogeneity + texts_onions_b_45.Homogeneity + texts_onions_b_90.Homogeneity + texts_onions_b_135.Homogeneity) / 4;
    homogeneity_onions_d(k) = (texts_onions_d_0.Homogeneity + texts_onions_d_45.Homogeneity + texts_onions_d_90.Homogeneity + texts_onions_d_135.Homogeneity) / 4;

    contrast_weeds_r(k) = (texts_weeds_r_0.Contrast + texts_weeds_r_45.Contrast + texts_weeds_r_90.Contrast + texts_weeds_r_135.Contrast) / 4;
    contrast_weeds_g(k) = (texts_weeds_g_0.Contrast + texts_weeds_g_45.Contrast + texts_weeds_g_90.Contrast + texts_weeds_g_135.Contrast) / 4;
    contrast_weeds_b(k) = (texts_weeds_b_0.Contrast + texts_weeds_b_45.Contrast + texts_weeds_b_90.Contrast + texts_weeds_b_135.Contrast) / 4;
    contrast_weeds_d(k) = (texts_weeds_d_0.Contrast + texts_weeds_d_45.Contrast + texts_weeds_d_90.Contrast + texts_weeds_d_135.Contrast) / 4;

    correlation_weeds_r(k) = (texts_weeds_r_0.Correlation + texts_weeds_r_45.Correlation + texts_weeds_r_90.Correlation + texts_weeds_r_135.Correlation) / 4;
    correlation_weeds_g(k) = (texts_weeds_g_0.Correlation + texts_weeds_g_45.Correlation + texts_weeds_g_90.Correlation + texts_weeds_g_135.Correlation) / 4;
    correlation_weeds_b(k) = (texts_weeds_b_0.Correlation + texts_weeds_b_45.Correlation + texts_weeds_b_90.Correlation + texts_weeds_b_135.Correlation) / 4;
    correlation_weeds_d(k) = (texts_weeds_d_0.Correlation + texts_weeds_d_45.Correlation + texts_weeds_d_90.Correlation + texts_weeds_d_135.Correlation) / 4;

    energy_weeds_r(k) = (texts_weeds_r_0.Energy + texts_weeds_r_45.Energy + texts_weeds_r_90.Energy + texts_weeds_r_135.Energy) / 4;
    energy_weeds_g(k) = (texts_weeds_g_0.Energy + texts_weeds_g_45.Energy + texts_weeds_g_90.Energy + texts_weeds_g_135.Energy) / 4;
    energy_weeds_b(k) = (texts_weeds_b_0.Energy + texts_weeds_b_45.Energy + texts_weeds_b_90.Energy + texts_weeds_b_135.Energy) / 4;
    energy_weeds_d(k) = (texts_weeds_d_0.Energy + texts_weeds_d_45.Energy + texts_weeds_d_90.Energy + texts_weeds_d_135.Energy) / 4;

    homogeneity_weeds_r(k) = (texts_weeds_r_0.Homogeneity + texts_weeds_r_45.Homogeneity + texts_weeds_r_90.Homogeneity + texts_weeds_r_135.Homogeneity) / 4;
    homogeneity_weeds_g(k) = (texts_weeds_g_0.Homogeneity + texts_weeds_g_45.Homogeneity + texts_weeds_g_90.Homogeneity + texts_weeds_g_135.Homogeneity) / 4;
    homogeneity_weeds_b(k) = (texts_weeds_b_0.Homogeneity + texts_weeds_b_45.Homogeneity + texts_weeds_b_90.Homogeneity + texts_weeds_b_135.Homogeneity) / 4;
    homogeneity_weeds_d(k) = (texts_weeds_d_0.Homogeneity + texts_weeds_d_45.Homogeneity + texts_weeds_d_90.Homogeneity + texts_weeds_d_135.Homogeneity) / 4;

end

%% Plot figures
% Plot shape features
figure; histogram(cell2mat(cellfun(@(x)x(:),circularity_onions(:),'un',0)), 'FaceColor', "red");
hold on; histogram(cell2mat(cellfun(@(x)x(:),circularity_weeds(:),'un',0)), 'FaceColor', "blue"); hold off;
title("Shape feature - Circularity"); legend("Onions","Weeds");

figure; histogram(cell2mat(cellfun(@(x)x(:),eccentricity_onions(:),'un',0)), 'FaceColor', "red");
hold on; histogram(cell2mat(cellfun(@(x)x(:),eccentricity_weeds(:),'un',0)), 'FaceColor', "blue"); hold off;
title("Shape feature - Eccentricity"); legend("Onions","Weeds");

figure; histogram(cell2mat(cellfun(@(x)x(:),solidity_onions(:),'un',0)), 'FaceColor', "red");
hold on; histogram(cell2mat(cellfun(@(x)x(:),solidity_weeds(:),'un',0)), 'FaceColor', "blue"); hold off;
title("Shape feature - Solidity"); legend("Onions","Weeds");

figure; histogram(cell2mat(cellfun(@(x)x(:),extent_onions(:),'un',0)), 'FaceColor', "red");
hold on; histogram(cell2mat(cellfun(@(x)x(:),extent_weeds(:),'un',0)), 'FaceColor', "blue"); hold off;
title("Shape feature - Extent"); legend("Onions","Weeds");

% Plot contrast
figure; histogram(contrast_onions_r, 'FaceColor', "red");
hold on; histogram(contrast_weeds_r, 'FaceColor', "blue"); hold off;
title("Texture feature - Contrast (Red colour channel)"); legend("Onions","Weeds");

figure; histogram(contrast_onions_g, 'FaceColor', "red");
hold on; histogram(contrast_weeds_g, 'FaceColor', "blue"); hold off;
title("Texture feature - Contrast (Green colour channel)"); legend("Onions","Weeds");

figure; histogram(contrast_onions_b, 'FaceColor', "red");
hold on; histogram(contrast_weeds_b, 'FaceColor', "blue"); hold off;
title("Texture feature - Contrast (Blue colour channel)"); legend("Onions","Weeds");

figure; histogram(contrast_onions_d, 'FaceColor', "red");
hold on; histogram(contrast_weeds_d, 'FaceColor', "blue"); hold off;
title("Texture feature - Contrast (Depth image)"); legend("Onions","Weeds");

% Plot correlation
figure; histogram(correlation_onions_r, 'FaceColor', "red");
hold on; histogram(correlation_weeds_r, 'FaceColor', "blue"); hold off;
title("Texture feature - Correlation (Red colour channel)"); legend("Onions","Weeds");

figure; histogram(correlation_onions_g, 'FaceColor', "red");
hold on; histogram(correlation_weeds_g, 'FaceColor', "blue"); hold off;
title("Texture feature - Correlation (Green colour channel)"); legend("Onions","Weeds");

figure; histogram(correlation_onions_b, 'FaceColor', "red");
hold on; histogram(correlation_weeds_b, 'FaceColor', "blue"); hold off;
title("Texture feature - Correlation (Blue colour channel)"); legend("Onions","Weeds");

figure; histogram(correlation_onions_d, 'FaceColor', "red");
hold on; histogram(correlation_weeds_d, 'FaceColor', "blue"); hold off;
title("Texture feature - Correlation (Depth image)"); legend("Onions","Weeds");

% Plot energy
figure; histogram(energy_onions_r, 'FaceColor', "red");
hold on; histogram(energy_weeds_r, 'FaceColor', "blue"); hold off;
title("Texture feature - Energy (Red colour channel)"); legend("Onions","Weeds");

figure; histogram(energy_onions_g, 'FaceColor', "red");
hold on; histogram(energy_weeds_g, 'FaceColor', "blue"); hold off;
title("Texture feature - Energy (Green colour channel)"); legend("Onions","Weeds");

figure; histogram(energy_onions_b, 'FaceColor', "red");
hold on; histogram(energy_weeds_b, 'FaceColor', "blue"); hold off;
title("Texture feature - Energy (Blue colour channel)"); legend("Onions","Weeds");

figure; histogram(energy_onions_d, 'FaceColor', "red");
hold on; histogram(energy_weeds_d, 'FaceColor', "blue"); hold off;
title("Texture feature - Energy (Depth image)"); legend("Onions","Weeds");

% Plot homogeneity
figure; histogram(homogeneity_onions_r, 'FaceColor', "red");
hold on; histogram(homogeneity_weeds_r, 'FaceColor', "blue"); hold off;
title("Texture feature - Homogeneity (Red colour channel)"); legend("Onions","Weeds");

figure; histogram(homogeneity_onions_g, 'FaceColor', "red");
hold on; histogram(homogeneity_weeds_g, 'FaceColor', "blue"); hold off;
title("Texture feature - Homogeneity (Green colour channel)"); legend("Onions","Weeds");

figure; histogram(homogeneity_onions_b, 'FaceColor', "red");
hold on; histogram(homogeneity_weeds_b, 'FaceColor', "blue"); hold off;
title("Texture feature - Homogeneity (Blue colour channel)"); legend("Onions","Weeds");

figure; histogram(homogeneity_onions_d, 'FaceColor', "red");
hold on; histogram(homogeneity_weeds_d, 'FaceColor', "blue"); hold off;
title("Texture feature - Homogeneity (Depth image)"); legend("Onions","Weeds");

%% Pass features to SVM

% shape features only
% uncomment to enable texture features only for SVM. Comment other features

% X_train = [avg_circularity_onions', avg_eccentricity_onions', avg_solidity_onions', avg_extent_onions';
%            avg_circularity_weeds', avg_eccentricity_weeds', avg_solidity_weeds', avg_extent_weeds'];
% X_test = [X_train(18,:); X_train(19,:); X_train(38,:); X_train(39,:)];
% X_train(39,:) = [];
% X_train(38,:) = [];
% X_train(19,:) = [];
% X_train(18,:) = [];

% texture features only
% uncomment to enable texture features only for SVM. Comment other features

% X_train = [contrast_onions_r', contrast_onions_g', contrast_onions_b', contrast_onions_d', correlation_onions_r', correlation_onions_g', correlation_onions_b', correlation_onions_d', energy_onions_r', energy_onions_g', energy_onions_b', energy_onions_d', homogeneity_onions_r', homogeneity_onions_g', homogeneity_onions_b', homogeneity_onions_d';
%            contrast_weeds_r', contrast_weeds_g', contrast_weeds_b', contrast_weeds_d', correlation_weeds_r', correlation_weeds_g', correlation_weeds_b', correlation_weeds_d', energy_weeds_r', energy_weeds_g', energy_weeds_b', energy_weeds_d', homogeneity_weeds_r', homogeneity_weeds_g', homogeneity_weeds_b', homogeneity_weeds_d'];
% X_test = [X_train(18,:); X_train(19,:); X_train(38,:); X_train(39,:)];
% X_train(39,:) = [];
% X_train(38,:) = [];
% X_train(19,:) = [];
% X_train(18,:) = [];

% shape and texture features 
% uncomment to enable both shape and texture features for SVM

X_train = [avg_circularity_onions', avg_eccentricity_onions', avg_solidity_onions', avg_extent_onions', contrast_onions_r', contrast_onions_g', contrast_onions_b', contrast_onions_d', correlation_onions_r', correlation_onions_g', correlation_onions_b', correlation_onions_d', energy_onions_r', energy_onions_g', energy_onions_b', energy_onions_d', homogeneity_onions_r', homogeneity_onions_g', homogeneity_onions_b', homogeneity_onions_d';
           avg_circularity_weeds', avg_eccentricity_weeds', avg_solidity_weeds', avg_extent_weeds', contrast_weeds_r', contrast_weeds_g', contrast_weeds_b', contrast_weeds_d', correlation_weeds_r', correlation_weeds_g', correlation_weeds_b', correlation_weeds_d', energy_weeds_r', energy_weeds_g', energy_weeds_b', energy_weeds_d', homogeneity_weeds_r', homogeneity_weeds_g', homogeneity_weeds_b', homogeneity_weeds_d'];
X_test = [X_train(18,:); X_train(19,:); X_train(38,:); X_train(39,:)];
X_train(39,:) = [];
X_train(38,:) = [];
X_train(19,:) = [];
X_train(18,:) = [];

for i = 1:18
    Y{i} = 'onions';
    Y{i+18} = 'weeds';
end
Y = Y';

% Plot feature weights
mdl = fscnca(X_train,Y,'Solver','sgd','Verbose',1);
figure()
plot(mdl.FeatureWeights,'ro')
grid on
xlabel('Feature index')
ylabel('Feature weight')

% Features with highest weights only
% Comment to enable previous features
X_train = [contrast_onions_r', contrast_onions_g', contrast_onions_b', contrast_onions_d', energy_onions_b';
           contrast_weeds_r', contrast_weeds_g', contrast_weeds_b', contrast_weeds_d', energy_weeds_b'];
X_test = [X_train(18,:); X_train(19,:); X_train(38,:); X_train(39,:)];
X_train(39,:) = [];
X_train(38,:) = [];
X_train(19,:) = [];
X_train(18,:) = [];

SVM = fitcsvm(X_train,Y)
disp('True values:')
disp('onions')
disp('onions')
disp('weeds')
disp('weeds')
[y_pred, score] = predict(SVM,X_test)


