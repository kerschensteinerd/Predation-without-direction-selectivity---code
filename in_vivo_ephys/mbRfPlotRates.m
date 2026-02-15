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


%% USER-DEFINED PARAMETERSo

nPanels = 2;% number of figure panels
bitDepth = 256;% for receptive field maps
offCmap = flipud(colormap(gray(256)));
nYTicks = 2; %approximate number of yticks in DS tuning plot
sdThresh = 3;% threshold by which an RF pixel has to exceed map SD to count
mbStimDur = 2.25;


%% PLOT

width = double(MatParams.stimIn.MB_Width);
nWidths = length(width);
direction = double(MatParams.stimIn.MB_Dir);
radDirection = deg2rad(direction);
nDirections = length(direction);
mbStimInterDur = double(MatParams.stimIn.MB_InterDur);
nMbRepeats = double(MatParams.stimIn.MB_Repeats);

for i=1:length(unit)
    h(i) = figure;
    set(h(i),'Name',num2str(i),'NumberTitle','off')

    %average and concatanate spike rates
    currPreMbRatesMean = squeeze(mean(unit(i).preMbRate,3,'omitnan'));
    currPreMbRatesError = squeeze(sem(unit(i).preMbRate,3));
    currMbRatesMean = squeeze(mean(unit(i).mbRate,3,'omitnan'));
    currMbRatesError = squeeze(sem(unit(i).mbRate,3));
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
                ylim([minRate maxRate])
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
        plot(direction,unit(i).mbResp(:,k),'-','Color',[widthColor(k) widthColor(k) widthColor(k)])
        if k==1
            hold on
        elseif k==nWidths
            plot(direction,mean(unit(i).mbResp,2),'-r','LineWidth',2)
        else
        end
    end
    box off
    xticks(direction)
    maxResp = round(max(unit(i).mbResp,[],[1,2],'omitnan'));
    if maxResp==0
    elseif floor(maxResp/nYTicks)==0
        yticks([0 maxResp])
    else
        yticks([0:floor(maxResp/nYTicks):(nYTicks*floor(maxResp/nYTicks))]) %#ok<*NBRAK> 
    end

    %Plot Width tuning curve
    hAx = subplot(nWidths,(nDirections+nPanels*nWidths+1),...
            [3*(nDirections+nPanels*nWidths+1)+nDirections+2,nWidths*(nDirections+nPanels*nWidths+1)-...
            nWidths]);
    if ~isempty(unit(i).mbWidth)
        plot(width,unit(i).mbWidth,'-r','LineWidth',2)
    else
    end
    box off
    xticks(width)
    maxWidthResp = max(unit(i).mbWidth);
    if maxWidthResp==0
    elseif floor(maxWidthResp/nYTicks)==0
        yticks([0 maxWidthResp])
    else
        yticks([0:floor(maxWidthResp/nYTicks):(nYTicks*floor(maxWidthResp/nYTicks))]) %#ok<*NBRAK> 
    end

    %Plot ON RF map
    hAxOn = subplot(nWidths,(nDirections+nPanels*nWidths+1),...
        [nDirections+nWidths+2,2*(nDirections+nPanels*nWidths+1)]);

    if i==1
        [mapRows,mapCols] = size(unit(i).onIntMap);
    else
    end

    medOn = median(unit(i).onIntMap(:),'omitnan');
    maxOn = max(unit(i).onIntMap,[],[1,2],'omitnan');
    onBooster = 2 * (maxOn/2 - medOn);

    onMapBoost = unit(i).onIntMap + onBooster;
    onMapNorm = onMapBoost / max(onMapBoost,[],[1,2],'omitnan');
    onMapImage = uint8(onMapNorm * bitDepth);
    image(onMapImage)
    colormap(hAxOn,gray(256))
    hold on
    onAboveThresh = unit(i).onIntMap(:) > median(unit(i).onIntMap(:),'omitnan')...
        + sdThresh * std(unit(i).onIntMap(:),'omitnan');
    if sum(onAboveThresh) > 0
        onAboveThresh = reshape(onAboveThresh,mapRows,mapCols);
        onMainRf = bwareaopen(onAboveThresh, unit(i).onMainSize);
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

    medOff = median(unit(i).offIntMap(:),'omitnan');
    maxOff = max(unit(i).offIntMap,[],[1,2],'omitnan');
    offBooster = 2 * (maxOff/2 - medOff);

    offMapBoost = unit(i).offIntMap + offBooster;
    offMapNorm = offMapBoost / max(offMapBoost,[],[1,2],'omitnan');
    offMapImage = uint8(offMapNorm * bitDepth);
    image(offMapImage)
    colormap(hAxOff,offCmap)
    hold on
    offAboveThresh = unit(i).offIntMap(:) > median(unit(i).offIntMap(:),'omitnan')...
        + sdThresh * std(unit(i).offIntMap(:),'omitnan');
    if sum(offAboveThresh) > 0
        offAboveThresh = reshape(offAboveThresh,mapRows,mapCols);
        offMainRf = bwareaopen(offAboveThresh, unit(i).offMainSize);
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


%% SAVE FIGURES

savefig(h,[pathName fileName(1:end-4) '_mbRates.fig'],'compact')
