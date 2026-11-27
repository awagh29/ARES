function buffer = pushHistory(model,buffer,ranges,valid,available,state,goal,time)
%PUSHHISTORY Measured current/past input only; hidden values are zeroed first.
assert(isfinite(time) && time>buffer.LastTime,'ARES:TemporalTime','Scan times must increase.');
if isfinite(buffer.LastTime) && abs(time-buffer.LastTime-double(model.samplePeriodS))>1e-6
    buffer=ares.mapTemporal.emptyHistory(model);
end
available=logical(available); valid=logical(valid)&available;
ranges=single(ranges); ranges(~available)=0;
assert(isequal(size(ranges),[15 716]) && all(isfinite(ranges),'all') && ...
    all(isfinite([state(:);goal(:)])),'ARES:TemporalObservation','Invalid scan/state/goal.');
age=single(0); if ~any(available,'all'), age=buffer.Ages(end)+single(model.samplePeriodS); end
near=(1-ranges/model.maxRange).*single(valid); starts=double(model.poolStarts(:));
summary=zeros(3,3,90,'single');
for el=1:3
    rows=(el-1)*5+(1:5);
    for az=1:90
        columns=starts(az):starts(az+1)-1;
        summary(1,el,az)=max(near(rows,columns),[],'all');
        summary(2,el,az)=single(nnz(valid(rows,columns)))/(5*numel(columns));
        summary(3,el,az)=single(nnz(available(rows,columns)))/(5*numel(columns));
    end
end
buffer.States=[buffer.States(2:end,:);reshape(single(state),1,9)];
buffer.Goals=[buffer.Goals(2:end,:);reshape(single(goal),1,3)];
buffer.Summaries(:,:,:,1:end-1)=buffer.Summaries(:,:,:,2:end); buffer.Summaries(:,:,:,end)=summary;
buffer.Present=[buffer.Present(2:end);true]; buffer.Ages=[buffer.Ages(2:end);age]; buffer.LastTime=double(time);
end
