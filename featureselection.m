%% Load Data from Excel File
data = readtable('Extracted_Colors.xlsx');

% Display column names to check structure
disp('Column Names:');
disp(data.Properties.VariableNames);

% Extract correct X (2nd to 10th columns) and Y (11th or last column)
X = data(:, 2:10); % Select columns 2 to 10
Y = data(:, end);  % Select the last column

% Convert X and Y to numeric arrays
if iscell(X{1,1})  % Check if data is in cell format
    X = cellfun(@str2double, table2cell(X)); % Convert cell to double
else
    X = table2array(X);
end

if iscell(Y{1,1})
    Y = cellfun(@str2double, table2cell(Y));
else
    Y = table2array(Y);
end

% Remove NaN values
X(any(isnan(X), 2), :) = [];
Y(any(isnan(Y), 2), :) = [];

% Verify number of features
disp(['Final number of features in X: ', num2str(size(X,2))]); % Should be 9
disp('First 5 entries of X:');
disp(X(1:5, :)); % Show first 5 rows of X
disp('First 5 entries of Y:');
disp(Y(1:5)); % Show first 5 rows of Y

%% Feature Selection - NCA
mdl_nca = fsrnca(X, Y, 'Verbose', 1);

% Plot NCA Feature Weights
figure();
plot(mdl_nca.FeatureWeights, 'ko', 'MarkerFaceColor', 'k');
xlabel('Feature Index');
ylabel('Feature Weight');
title('NCA Feature Weights');
grid on;

% Set X-axis labels to R, G, B, L, a, B, H, S, V
xticklabels({'R', 'G', 'B', 'L', 'a', 'b', 'H', 'S', 'V'});
xticks(1:9); 

% Normalize NCA Feature Weights
weights_nca = mdl_nca.FeatureWeights;
normalized_weights_nca = weights_nca / sum(weights_nca);

% Display normalized NCA weights
disp('Normalized NCA Feature Weights:');
for i = 1:length(normalized_weights_nca)
    fprintf('Feature %d: %.4f\n', i, normalized_weights_nca(i));
end

%% Feature Selection - mRMR
[idx_mrmr, scores_mrmr] = fscmrmr(X, Y);

% Normalize mRMR Scores
normalized_weights_mrmr = scores_mrmr / sum(scores_mrmr);

% Plot mRMR Feature Weights
figure();
plot(idx_mrmr, normalized_weights_mrmr, 'ko', 'MarkerFaceColor', 'k');
xlabel('Feature Index');
ylabel('Feature Weight');
title('mRMR Feature Weights');
grid on;

% Set X-axis labels to R, G, B, L, a, B, H, S, V
xticklabels({'R', 'G', 'B', 'L', 'a', 'b', 'H', 'S', 'V'});
xticks(1:9); 

% Display normalized mRMR weights
disp('Normalized mRMR Feature Weights:');
for i = 1:length(normalized_weights_mrmr)
    fprintf('Feature %d: %.4f\n', idx_mrmr(i), normalized_weights_mrmr(i));
end

%% Feature Selection - ReliefF
numNeighbors = 10; % Number of neighbors for ReliefF
[idx_reliefF, scores_reliefF] = relieff(X, Y, numNeighbors);

% Normalize ReliefF Scores
normalized_weights_reliefF = scores_reliefF / sum(scores_reliefF);

% Plot ReliefF Feature Weights
figure();
plot(idx_reliefF, normalized_weights_reliefF(idx_reliefF), 'ko', 'MarkerFaceColor', 'k');
xlabel('Feature Index');
ylabel('Feature Weight');
title('ReliefF Feature Weights');
grid on;

% Set X-axis labels to R, G, B, L, a, B, H, S, V
xticklabels({'R', 'G', 'B', 'L', 'a', 'b', 'H', 'S', 'V'});
xticks(1:9); 

% Display normalized ReliefF weights
disp('Normalized ReliefF Feature Weights:');
for i = 1:length(normalized_weights_reliefF)
    fprintf('Feature %d: %.4f\n', idx_reliefF(i), normalized_weights_reliefF(i));
end