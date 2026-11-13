function hash = fileSha256(path)
%FILESHA256 Hash artifact bytes for inference/telemetry provenance.
fid = fopen(path,"rb");
assert(fid~=-1,"ARES:MissingArtifact","Cannot read artifact: %s",path);
cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>
digest = java.security.MessageDigest.getInstance("SHA-256");
while ~feof(fid)
    bytes = fread(fid,1024*1024,"*uint8");
    digest.update(typecast(bytes,'int8'));
end
hash = lower(string(reshape(dec2hex(typecast(digest.digest(),'uint8'),2).',1,[])));
end
