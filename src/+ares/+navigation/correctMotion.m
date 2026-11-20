function [acceleration,info] = correctMotion(position,velocity,nominal,previous,map,options,measuredWorldPoints)
%CORRECTMOTION Hard map/measurement CBF-QP, including acceleration/slew bounds.
% Map distances use supporting planes of convex boxes. Omitting the positive
% distance-curvature term is conservative outside each convex obstacle.
% This software filter assumes bounded translational tracking error. It is
% not a certificate for the full sampled attitude/thrust plant or hardware.
if nargin<7, measuredWorldPoints=zeros(0,3); end
o=options; p=reshape(double(position),1,3); v=reshape(double(velocity),1,3);
radius=o.VehicleRadiusM+o.MinimumSurfaceClearanceM+o.GeometryUncertaintyM+o.TrackingMarginM;
[distance,normals]=ares.navigation.mapClearance(p,map);
h=distance-radius;
bounds=o.WorldBoundsNedM;
normals=[normals;eye(3);-eye(3)];
h=[h;(p-bounds(:,1)')';(bounds(:,2)'-p)'];
if ~isempty(measuredWorldPoints)
    delta=p-measuredWorldPoints; distances=vecnorm(delta,2,2);
    % A return at the vehicle center is already an invalid safety state.
    % Keep its negative h and zero normal so the hard QP rejects it.
    normals=[normals;delta./max(distances,eps)];
    h=[h;distances-radius];
end
hdot=normals*v'; gain=o.BarrierGain;
% Explicit engineering allowance for held controls, in addition to the
% requested acceleration-error bound. Sampling robustness remains empirical.
allowance=o.AccelerationUncertaintyMps2+gain*o.AccelerationLimitMps2*o.ControlSampleTimeS;
A=-normals;
b=2*gain*hdot+gain^2*h-allowance;
change=o.JerkLimitMps3*o.ControlSampleTimeS;
lower=max(-o.AccelerationLimitMps2,reshape(previous,3,1)-change);
upper=min(o.AccelerationLimitMps2,reshape(previous,3,1)+change);
[candidate,solved]=ares.navigation.projectControl(nominal,A,b,lower,upper);
info.MinimumH=min(h); info.MinimumPsi1=min(hdot+gain*h);
info.Feasible=solved && numel(candidate)==3 && all(isfinite(candidate));
info.MaximumViolation=1e6;
if info.Feasible
    info.MaximumViolation=max([0;A*candidate-b;lower-candidate;candidate-upper]);
    info.Feasible=info.MaximumViolation<=1e-6 && info.MinimumH>=-1e-5 && info.MinimumPsi1>=-1e-4;
end
if ~info.Feasible
    % Stop Simulation is asserted by the caller. This bounded brake is a
    % finite output for termination, never relabeled as feasible or safe.
    candidate=min(max(-o.VelocityGain*v',lower),upper);
    info.MaximumViolation=max([0;A*candidate-b;lower-candidate;candidate-upper]);
end
acceleration=candidate';
end
