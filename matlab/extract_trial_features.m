function [features, bounds] = extract_trial_features(signal, nTrials, nSegments)
% 15 exhaustive non-overlapping intervals; channel/segment/statistic order.
% Sample variance uses N-1, matching var(x,0) and Python ddof=1.
assert(size(signal,2)==8 && all(isfinite(signal(:))), 'Expected finite N-by-8 data.');
edges = floor((0:nTrials)*size(signal,1)/nTrials);
features = zeros(nTrials,8*nSegments*2);
bounds = [edges(1:end-1)'+1 edges(2:end)'];
for t = 1:nTrials
    trial = signal(bounds(t,1):bounds(t,2),:);
    segmentEdges = floor((0:nSegments)*size(trial,1)/nSegments);
    assert(all(diff(segmentEdges)>=2), 'A segment needs at least two samples.');
    for c = 1:8
        for s = 1:nSegments
            values = trial(segmentEdges(s)+1:segmentEdges(s+1),c);
            j = ((c-1)*nSegments+s-1)*2+1;
            features(t,j:j+1) = [mean(values) var(values,0)];
        end
    end
end
end
