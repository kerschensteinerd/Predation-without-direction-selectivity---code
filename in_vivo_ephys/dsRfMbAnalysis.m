%dsRfAnalysis. This scripts computes keye outputs and parameters of
%responses to drifting grating and sparse noise stimuli. Uses as input
%'..._parsed.mat' file.

%% CLEAR MEMORY AND LOAD DATA

clear; clc; close all
[fileName, pathName] = uigetfile('*.mat', 'Select parsed .mat file');
if fileName == 0
    error('No file selected. Exiting.');
end
load([pathName fileName])


%% USER-DEFINED PARAMETERS

mapRes = 1;% final resolution of interpolated RF maps in degrees
sdThresh = 3;% threshold by which an RF pixel has to exceed map SD to count
fractThresh = 0.5;% threshold (relative to max) above which pixels are included in RF map
lowIdx = 1:2;% index for slow speeds
hiIdx = 3:4;% index for fast speeds
smIdx = 1:2;%index for small sizes
laIdx = 3:4;%index for large sizes


%% ANALYZE DIRECTION AND ORIENTATION SELECTIVITY AND SPEED TUNING FOR DRIFITING GRATING STIMULI

speed = double(MatParams.stimIn.Speed);
nSpeeds = length(speed);
direction = double(MatParams.stimIn.Dir);
radDirection = deg2rad(direction);

for i=1:length(unit)
    %average DS responses
    unit(i).dsResp = mean(unit(i).dsRate,[3,4]);
    unit(i).dsRespSub = mean(unit(i).dsRate,[3,4]) - mean(unit(i).preDsRate,[3,4]); %#ok<*SAGROW>

    %preallocate memory for DS and OS parameters
    unit(i).dsi = zeros(nSpeeds,1);
    unit(i).dsPref = zeros(nSpeeds,1);
    unit(i).osi = zeros(nSpeeds,1);
    unit(i).osPref = zeros(nSpeeds,1);

    for j=1:nSpeeds
        dsVar = sum(unit(i).dsResp(:,j) .* exp(1i*radDirection')) / sum(unit(i).dsResp(:,j));
        unit(i).dsi(j) = abs(dsVar);
        unit(i).dsPref(j) = rad2deg(unwrap(angle(dsVar)));
        osVar = sum(unit(i).dsResp(:,j) .* exp(1i*2*radDirection')) / sum(unit(i).dsResp(:,j));
        unit(i).osi(j) = abs(osVar);
        unit(i).osPref(j) = rad2deg(unwrap(angle(osVar))) / 2;
    end
    dsRespAv = mean(unit(i).dsResp,2);
    prefIdx = find(dsRespAv==max(dsRespAv),1,'first');
    unit(i).dsSpeed = unit(i).dsResp(prefIdx,:);
    if isempty(unit(i).dsSpeed)
        unit(i).spi = NaN;
    else
        unit(i).spi = (mean(unit(i).dsSpeed(hiIdx)) - mean(unit(i).dsSpeed(lowIdx))) /...
            (mean(unit(i).dsSpeed(hiIdx)) + mean(unit(i).dsSpeed(lowIdx)));
    end
end


%% ANALYZE DIRECTION AND ORIENTATION SELECTIVITY AND SIZE SELECTIVITY FOR MOVING BAR STIMULI

width = double(MatParams.stimIn.MB_Width);
nWidths = length(width);
direction = double(MatParams.stimIn.MB_Dir);
radDirection = deg2rad(direction);
nDirections = length(direction);

for i=1:length(unit)
    %average DS responses
    unit(i).mbResp = mean(unit(i).mbRate,[3,4]);
    unit(i).mbRespSub = mean(unit(i).mbRate,[3,4]) - mean(unit(i).preMbRate,[3,4]); %#ok<*SAGROW>

    %preallocate memory for DS and OS parameters
    unit(i).mbDsi = zeros(nWidths,1);
    unit(i).mbDsPref = zeros(nWidths,1);
    unit(i).mbOsi = zeros(nWidths,1);
    unit(i).mbOsPref = zeros(nWidths,1);

    for j=1:nWidths
        dsVar = sum(unit(i).mbResp(:,j) .* exp(1i*radDirection')) / sum(unit(i).mbResp(:,j));
        unit(i).mbDsi(j) = abs(dsVar);
        unit(i).mbDsPref(j) = rad2deg(unwrap(angle(dsVar)));
        osVar = sum(unit(i).mbResp(:,j) .* exp(1i*2*radDirection')) / sum(unit(i).mbResp(:,j));
        unit(i).mbOsi(j) = abs(osVar);
        unit(i).mbOsPref(j) = rad2deg(unwrap(angle(osVar))) / 2;
    end
    mbRespAv = mean(unit(i).mbResp,2);
    prefIdx = find(mbRespAv==max(mbRespAv),1,'first');
    unit(i).mbWidth = unit(i).mbResp(prefIdx,:);
    if isempty(unit(i).mbWidth)
        unit(i).wpi = NaN;
    else
        unit(i).wpi = (mean(unit(i).mbWidth(laIdx)) - mean(unit(i).mbWidth(smIdx))) /...
            (mean(unit(i).mbWidth(laIdx)) + mean(unit(i).mbWidth(smIdx)));
    end
end


%% MAP RECEPTIVE FIELDS

stimRf = MatParams.RF_Data;
azimuth = sort(unique(stimRf(:,4)));
nAzimuths = length(azimuth);
elevation = sort(unique(stimRf(:,5)));
nElevations = length(elevation);
nRfRepeats = double(MatParams.stimIn.spotRepeats);

spotSize = double(MatParams.stimIn.spotSize);
origAzi = min(stimRf(:,4)):spotSize:max(stimRf(:,4));
origEle = min(stimRf(:,5)):spotSize:max(stimRf(:,5));
[origX, origY] = meshgrid(origAzi,origEle);

interpAzi = min(stimRf(:,4)):mapRes:max(stimRf(:,4));
interpEle = min(stimRf(:,5)):mapRes:max(stimRf(:,5));
[interpX, interpY] = meshgrid(interpAzi,interpEle);

for i=1:length(unit)
    unit(i).onMap = mean(unit(i).onRate,[3,4]);
    unit(i).onIntMap = interp2(origX,origY,unit(i).onMap',interpX,interpY);%re-orient and interpolate ON map
    if i==1
        [mapRows,mapCols] = size(unit(i).onIntMap);
    else
    end
    onAboveThresh = unit(i).onIntMap(:) > median(unit(i).onIntMap(:),'omitnan')...
        + sdThresh * std(unit(i).onIntMap(:),'omitnan');
    if any(onAboveThresh,'all')
        unit(i).onAllSize = sum(onAboveThresh,'all','omitnan');
        onAboveThresh = reshape(onAboveThresh,mapRows,mapCols);
        onMeasurements = regionprops(onAboveThresh, 'Area');
        onAreas = [onMeasurements.Area];
        unit(i).onMainSize = max(onAreas);
        onMainAboveThresh = bwareaopen(onAboveThresh,max(onAreas));
    else
        unit(i).onAllSize = 0;
        unit(i).onMainSize = 0;
    end

    unit(i).offMap = mean(unit(i).offRate,[3,4]);
    unit(i).offIntMap = interp2(origX,origY,unit(i).offMap',interpX,interpY);%re-orient and interpolate OFF map
    offAboveThresh = unit(i).offIntMap(:) > median(unit(i).offIntMap(:),'omitnan')...
        + sdThresh * std(unit(i).offIntMap(:),'omitnan');
    if any(offAboveThresh,'all')
        unit(i).offAllSize = sum(offAboveThresh,'all','omitnan');
        offAboveThresh = reshape(offAboveThresh,mapRows,mapCols);
        offMeasurements = regionprops(offAboveThresh, 'Area');
        offAreas = [offMeasurements.Area];
        unit(i).offMainSize = max(offAreas);
        offMainAboveThresh = bwareaopen(offAboveThresh,max(offAreas));
    else
        unit(i).offAllSize = 0;
        unit(i).offMainSize = 0;
    end

    if any(onAboveThresh,'all') && any(offAboveThresh,'all')
        onOffIntersect = sum(onMainAboveThresh .* offMainAboveThresh,'all','omitnan');
        unit(i).onOffOverlap = (2 * onOffIntersect) /...
            sum(onMainAboveThresh + offMainAboveThresh,'all','omitnan');
    else
        unit(i).onOffOverlap = NaN;
    end
    unit(i).polIdxAmp = (max(unit(i).onMap,[],'all','omitnan') - max(unit(i).offMap,[],'all','omitnan')) /...
        (max(unit(i).onMap,[],'all','omitnan') + max(unit(i).offMap,[],'all','omitnan'));
    unit(i).polIdxArea = (unit(i).onMainSize - unit(i).offMainSize) /...
        (unit(i).onMainSize + unit(i).offMainSize);
end


%% SAVE RESULTS

underscores = strfind(fileName,'_');
lastUnderscore = underscores(end);
save([pathName fileName(1:lastUnderscore-1) '_analyzed.mat'],'unit','MatParams')

