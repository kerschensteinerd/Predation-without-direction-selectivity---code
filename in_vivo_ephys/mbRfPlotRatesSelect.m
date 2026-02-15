%dsRFPlot plots responses to drifiting grating stimuli (spike rasters and
%polar plots) and RF maps from sparse noise stimuli. Uses as input
%'..._analyzed.mat' file.

%% CLEAR MEMORY AND LOAD DATA

clear; clc; close all

[fileName, pathName] = uigetfile('*.mat', 'Select analyzed .mat file');
if fileName == 0
    error('No file selected. Exiting.');
end
load([pathName fileName])


%% USER-DEFINED PARAMETERS

nPanels = 2;% number of figure panels
bitDepth = 256;% for receptive field maps
offCmap = flipud(colormap(gray(256)));
nYTicks = 2; %approximate number of yticks in DS tuning plot
sdThresh = 3;% threshold by which an RF pixel has to exceed map SD to count
mbStimDur = 2.25;
repRelThresh = 0.1; %threshold for repeat reliability
frThresh = 3; %firing rate threshold (sp/s)


%% SELECT DATA FOR PLOTTING OR SUMMARY ANALYSIS

nUnits = length(unit);
for i=1:length(unit)
    maxResp = max(unit(i).mbResp,[],'all','omitnan');
    if maxResp > frThresh
        maxMbCorr = max(unit(i).mbRepRel(:),[],'omitnan');
        if ~exist('selectIdx','var') && maxMbCorr > repRelThresh
            selectIdx = i;
        elseif exist('selectIdx','var') && maxMbCorr > repRelThresh
            nSoFar = numel(selectIdx);
            selectIdx(nSoFar+1) = i;
        else
        end
    else
    end
end


%% PLOT

width = double(MatParams.stimIn.MB_Width);
nWidths = length(width);
direction = double(MatParams.stimIn.MB_Dir);
radDirection = deg2rad(direction);
nDirections = length(direction);
mbStimInterDur = double(MatParams.stimIn.MB_InterDur);
nMbRepeats = double(MatParams.stimIn.MB_Repeats);

for i=1:length(selectIdx)
    h(i) = figure;
    set(h(i),'Name',num2str(selectIdx(i)),'NumberTitle','off')

    %average and concatanate spike rates
    currPreMbRatesMean = squeeze(mean(unit(selectIdx(i)).preMbRate,3,'omitnan'));
    currPreMbRatesError = squeeze(sem(unit(selectIdx(i)).preMbRate,3));
    currMbRatesMean = squeeze(mean(unit(selectIdx(i)).mbRate,3,'omitnan'));
    currMbRatesError = squeeze(sem(unit(selectIdx(i)).mbRate,3));
    currRatesMean = cat(3,currPreMbRatesMean,currMbRatesMean);
    currRatesError = cat(3,currPreMbRatesError,currMbRatesError);
    maxRate = max(currRatesMean(:));
    minRate = min(currRatesMean(:));

    if i==1
        nBins = size(currRatesMean,3);
        rateTime = linspace(0,mbStimInterDur+mbStimDur,nBins);
    else
    end


    % Plot DS spike rates
    for j=1:nDirections
        for k=1:nWidths
            hAx = subplot(nWidths,(nDirections+nPanels*nWidths+1),...
                j+(k-1)*(nDirections+nPanels*nWidths+1));
            shadePlot(rateTime,squeeze(currRatesMean(j,k,:)),squeeze(currRatesError(j,k,:)),[0 0 0])
            if maxRate > 0
                % ylim([minRate maxRate])
                ylim([0 maxRate])
            else
            end
            box on
            if k==1
                title(num2str(round(direction(j))))
                set(hAx,'YTick', [], 'XTick', [])
            else
                set(hAx,'YTick', [], 'XTick', [])
            end
            if j==1
                ylabel(num2str(width(k)),'Rotation', 0.0,...
                    'HorizontalAlignment', 'right')
            else
            end
        end
    end

    %Plot DS tuning curves
    widthColor = linspace(0.75,0,nWidths);

    for k=1:nWidths
        hAx = subplot(nWidths,(nDirections+nPanels*nWidths+1),...
            [nDirections+2,2*(nDirections+nPanels*nWidths+1)-...
            nWidths]);
        %     polarplot([radDirection radDirection(1)],[unit(i).dsResp(:,k)' unit(i).dsResp(1,k)], ...
        %         '-','Color',[speedColor(k) speedColor(k) speedColor(k)])
        plot(direction,unit(selectIdx(i)).mbResp(:,k),'-','Color',[widthColor(k) widthColor(k) widthColor(k)])
        if k==1
            hold on
        elseif k==nWidths
            plot(direction,mean(unit(selectIdx(i)).mbResp,2),'-r','LineWidth',2)
        else
        end
    end
    box off
    xticks(direction)
    maxResp = ceil(max(unit(selectIdx(i)).mbResp,[],[1,2],'omitnan'));
    if maxResp==0
    elseif floor(maxResp/nYTicks)==0
        yticks([0 maxResp])
    else
        yticks([0:floor(maxResp/nYTicks):(nYTicks*floor(maxResp/nYTicks))]) %#ok<*NBRAK>
    end
    ylim([0 maxResp])


    %Plot Width tuning curve
    hAx = subplot(nWidths,(nDirections+nPanels*nWidths+1),...
        [3*(nDirections+nPanels*nWidths+1)+nDirections+2,nWidths*(nDirections+nPanels*nWidths+1)-...
        nWidths]);
    if ~isempty(unit(selectIdx(i)).mbWidth)
        plot(width,unit(selectIdx(i)).mbWidth,'-r','LineWidth',2)
    else
    end
    box off
    xticks(width)
    maxWidthResp = max(unit(selectIdx(i)).mbWidth);
    if maxWidthResp==0
    elseif floor(maxWidthResp/nYTicks)==0
        yticks([0 maxWidthResp])
    else
        yticks([0:floor(maxWidthResp/nYTicks):(nYTicks*floor(maxWidthResp/nYTicks))]) %#ok<*NBRAK>
    end
    ylim([0 maxWidthResp])

    %Plot ON RF map
    hAxOn = subplot(nWidths,(nDirections+nPanels*nWidths+1),...
        [nDirections+nWidths+2,2*(nDirections+nPanels*nWidths+1)]);

    if i==1
        [mapRows,mapCols] = size(unit(selectIdx(i)).onIntMap);
    else
    end

    medOn = median(unit(selectIdx(i)).onIntMap(:),'omitnan');
    maxOn = max(unit(selectIdx(i)).onIntMap,[],[1,2],'omitnan');
    onBooster = 2 * (maxOn/2 - medOn);

    onMapBoost = unit(selectIdx(i)).onIntMap + onBooster;
    onMapNorm = onMapBoost / max(onMapBoost,[],[1,2],'omitnan');
    onMapImage = uint8(onMapNorm * bitDepth);
    image(onMapImage)
    colormap(hAxOn,gray(256))
    hold on
    onAboveThresh = unit(selectIdx(i)).onIntMap(:) > median(unit(selectIdx(i)).onIntMap(:),'omitnan')...
        + sdThresh * std(unit(selectIdx(i)).onIntMap(:),'omitnan');
    if sum(onAboveThresh,'all') > 0
        onAboveThresh = reshape(onAboveThresh,mapRows,mapCols);
        onMainRf = bwareaopen(onAboveThresh, unit(selectIdx(i)).onMainSize);
        [bOn,~] = bwboundaries(onMainRf,'noholes');
        for j = 1:length(bOn)
            currOnBound = bOn{j};
            plot(currOnBound(:,2), currOnBound(:,1), 'r', 'LineWidth', 2)
        end
    else
    end
    set(hAxOn,'YTick', [], 'XTick', [])
    colorbar(hAxOn,'east','Ticks',[128 256],'TickLabels',{num2str(medOn), num2str(maxOn)})

    %Plot OFF RF map
    hAxOff = subplot(nWidths,(nDirections+nPanels*nWidths+1),...
        [3*(nDirections+nPanels*nWidths+1)+nDirections+nWidths+2,nWidths*(nDirections+nPanels*nWidths+1)]);

    medOff = median(unit(selectIdx(i)).offIntMap(:),'omitnan');
    maxOff = max(unit(selectIdx(i)).offIntMap,[],[1,2],'omitnan');
    offBooster = 2 * (maxOff/2 - medOff);

    offMapBoost = unit(selectIdx(i)).offIntMap + offBooster;
    offMapNorm = offMapBoost / max(offMapBoost,[],[1,2],'omitnan');
    offMapImage = uint8(offMapNorm * bitDepth);
    image(offMapImage)
    colormap(hAxOff,offCmap)
    hold on
    offAboveThresh = unit(selectIdx(i)).offIntMap(:) > median(unit(selectIdx(i)).offIntMap(:),'omitnan')...
        + sdThresh * std(unit(selectIdx(i)).offIntMap(:),'omitnan');
    if sum(offAboveThresh,'all') > 0
        offAboveThresh = reshape(offAboveThresh,mapRows,mapCols);
        offMainRf = bwareaopen(offAboveThresh, unit(selectIdx(i)).offMainSize);
        [bOff,~] = bwboundaries(offMainRf,'noholes');
        for j = 1:length(bOff)
            currOffBound = bOff{j};
            plot(currOffBound(:,2), currOffBound(:,1), 'r', 'LineWidth', 2)
        end
    else
    end
    set(hAxOff,'YTick', [], 'XTick', [])
    colorbar(hAxOff,'east','Ticks',[128 256],'TickLabels',{num2str(medOff), num2str(maxOff)})

end


%% MAKE A TABLE WITH RESPONSE PARAMETERS OF SELECTED UNITS

selectTable = table(selectIdx', [unit(selectIdx).chanNum]', {unit(selectIdx).label}',...
    reshape(cell2mat({unit(selectIdx).mbWidth})',[nWidths, length(selectIdx)])',...
    cell2mat({unit(selectIdx).wpi})', mean(cell2mat({unit(selectIdx).dsi}))', ...
    mean(cell2mat({unit(selectIdx).osi}))', cell2mat({unit(selectIdx).onMainSize})',...
    cell2mat({unit(selectIdx).offMainSize})', cell2mat({unit(selectIdx).onOffOverlap})',...
    cell2mat({unit(selectIdx).polIdxArea})', cell2mat({unit(selectIdx).polIdxAmp})',...
    'VariableNames', {'SelectIdx', 'Channel', 'Sorting', 'widthTuning', 'WPI', 'DSI', 'OSI',...
    'onSize', 'offSize', 'onoffOverlap', 'polIdxArea', 'polIdxAmp'});


%% SAVE FIGURES & TABLE

savefig(h,[pathName fileName(1:end-4) '_mbRatesSelect.fig'],'compact')
writetable(selectTable, [pathName fileName(1:end-4) '_mbSelect.xlsx'])
