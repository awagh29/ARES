function buffer = emptyHistory(model)
%EMPTYHISTORY A fresh per-UAV buffer with no fabricated healthy observations.
h=double(model.historySteps);
buffer=struct('States',zeros(h,9,'single'),'Goals',zeros(h,3,'single'), ...
    'Summaries',zeros(3,3,90,h,'single'),'Present',false(h,1), ...
    'Ages',zeros(h,1,'single'),'LastTime',-Inf);
end
