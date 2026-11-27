function writeJson(path,value)
%WRITEJSON Write experiment metadata with deterministic cleanup.
fid=fopen(path,'w'); assert(fid>=0,'ARES:TemporalWrite','Cannot write %s',path);
cleanup=onCleanup(@()fclose(fid)); %#ok<NASGU>
fprintf(fid,'%s\n',jsonencode(value,PrettyPrint=true));
end
