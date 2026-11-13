# Map-free LiDAR barrier used by the augmented SAITS experiment

This document derives the inequalities implemented in [`scripts/ares_safe_training/barrier.py`](../scripts/ares_safe_training/barrier.py). It describes a conditional horizontal safety filter under explicit sensing, disturbance, and initialization assumptions. The implementation reports invalid or infeasible conditions and uses a separate bounded emergency command; it does not label that command a certified safe solution.

The relative-degree-two construction follows [Xiao and Belta's high-order control barrier functions](https://calinbelta.com/wp-content/uploads/2023/12/High-Order_Control_Barrier_Functions.pdf); differentiating a learned gain through a QP follows the design motivation of [BarrierNet](https://arxiv.org/abs/2111.11277). The sector reduction, range minimization, uncertainty allowances, and operational restrictions below are project-specific. They are not a new proof supplied by those papers.

## 1. Interface and assumptions

Use horizontal NED position `p = [pN,pE]`, velocity `v`, virtual acceleration `a`, and realized horizontal acceleration `a+w`, with `||w||₂ ≤ D`. The true rollout plant retains all 13 guidance states, attitude/yaw dynamics, and actual thrust. The acceleration interface is used only to formulate the filter; it does not replace that plant in training.

Current simulator assumptions are static obstacles and exact internal position, velocity, and attitude. The configured position/velocity uncertainty bounds are zero because the simulated state is exact; nonzero hardware state errors are not implemented. `D` is an empirical horizontal tracking bound from the separately saved MATLAB calibration, including startup and a declared margin. Its measured ramp-domain coverage is not a uniform guarantee for arbitrary command histories.

The online filter receives only current measured/reconstructed sector features, current motion state, previous issued acceleration, a nominal acceleration, and a goal-derived reference. It never reads obstacle centers, an occupancy map, hidden clean targets, or future measurements. Simulator geometry is confined to sensor rendering, supervision, and evaluation. Absolute position in the reconstructor can still encode scene-specific associations learned from data.

Seven sectors cover body-horizontal bearings −60° to +60°. Their lower/upper boundaries come from eight equally spaced edges. The actual sensor samples `linspace(-179,179,716)` in azimuth and 15 elevations from −15° to +15°. The installed roll-180° mount makes body bearing the negative of sensor azimuth. The min-distance and return-fraction pair from every affected sector is masked together during a fault. A healthy no-return sector is measured `[7 m,0]`, not missing evidence of an obstacle.

The model concerns the visible, horizontally projected static surfaces represented by these sectors. Thin obstacles between rays, occlusion, objects outside elevation support, moving obstacles, and unknown side/rear objects are not certified by a finite scan. The 0.27 m beam allowance is engineering conservatism, not a proof that every continuous surface was sampled.

## 2. Conservative sector distance

Let `r̂j` be predicted/measured sector minimum range and `bj` its one-sided overestimation allowance. For a fully measured distance/fraction pair, reconstruction allowance is zero. For an unobserved pair, the allowance is the fixed calibrated value for that sector. It is not a learned confidence score, and return fraction is not treated as confidence.

The implemented lower horizontal distance is

\[
L_j=(\hat r_j-b_j)c_h-\ell_{mount}-m_{beam}-m_{sample},
\qquad
m_{sample}=\|v\|\Delta t+\tfrac12(\sqrt2 a_{max}+D)\Delta t^2.
\]

Here `c_h = cos(tilt + elevation_half_angle + orientation_margin)` lower-bounds the horizontal projection while that angle is below 89°. The mount lever allowance is 0.4 m, the recorded sensor offset norm. `m_beam = 0.27 m` is approximately the full diagonal angular-pixel chord at 7 m for this grid. `m_sample` is a one-interval movement allowance using the horizontal per-axis acceleration box. These allowances reduce the usable horizon; they do not turn sparse returns into a continuous free-space certificate or prove arbitrary sampled-data invariance.

Empirical `bj` is computed on 12 reserved training trajectories for the current reconstructor: the chosen quantile of `max(predicted_distance − clean_distance, 0)`, plus 0.10 m, with a 0.05 m floor. Defaults use the finite sample maximum (`quantile=1`). Bounds are checkpoint-specific and recalibrated after each Stage-2 candidate, without gradients. They are not distribution-free population coverage guarantees, and closed-loop states can differ from these recorded calibration states.

The inflated vehicle radius is

\[
R=r_{vehicle}+r_{clearance}=0.1+1.0=1.1\;\mathrm m.
\]

If any sector has `Lj < R`, or angular/state validity fails, the filter declares its CBF conditions invalid. It does not clamp the observation into an apparently valid clearance.

## 3. Relative-degree-two obstacle inequality

Consider a static horizontal obstacle point `q`, relative vector `q−p=d n`, unit direction `n`, and distance `d≥Lj`. Define

\[
h=\|q-p\|^2-R^2=d^2-R^2,
\qquad \dot h=-2d\,n^T v,
\qquad \ddot h=2\|v\|^2-2d\,n^T(a+w).
\]

With constant positive `k1`, let `ψ1 = ḣ+k1 h`. Enforce `ψ̇1+k2 ψ1≥0` with positive `k2`. Therefore

\[
n^T[a+(k_1+k_2)v]
\leq \frac{k_1 k_2}{2}d+
\frac{\|v\|^2-\frac12 k_1 k_2R^2}{d}-D.
\]

The last term uses `nᵀw≤||w||≤D`. Keeping `k1` fixed is deliberate: the learned state-dependent `k2` multiplies `ψ1`; no derivative of a learned gain is omitted from `ψ1` itself. The code uses `k1=2`, and `k2=0.5+3.5 sigmoid(fθ(features))`, bounded in `(0.5,4)`. Its small gain network sees horizontal velocity, attitude sine/cosine, and sector features. It does not learn `D`, range bounds, radius, or geometric margins.

### Unknown distance: minimize over every `d ≥ Lj`

Write `K=k1 k2`, `C=||v||²−K R²/2`, and `β(d)=Kd/2+C/d`. If `C>0`, its unconstrained minimizer is `sqrt(2||v||²/K−R²)`; if `C≤0`, it increases for positive distance. Thus

\[
d_j^*=\max\left(L_j,\sqrt{\max(2\|v\|^2/K-R^2,0)}\right),
\qquad B_j=\beta(d_j^*)-D.
\]

This minimization is essential. Substituting only the nearest range into an inequality that must hold for every unknown farther point can be incorrect when `β` decreases over part of the interval. No finite maximum obstacle distance is needed for this algebra; the sensing assumption still limits which directions/surfaces are represented.

### Unknown direction: outer triangle of the sector arc

After rotating the sector's actual azimuth columns, edges, and elevation extremes by current attitude, let the projected horizontal center/half-angle be `(θj,φj)`, with `φj<π/2`. Define three vertices

\[
u_-=n(\theta_j-\phi_j),\quad
u_+=n(\theta_j+\phi_j),\quad
u_0=\frac{n(\theta_j)}{\cos\phi_j},\qquad n(\theta)=[\cos\theta,\sin\theta]^T.
\]

The unit-direction arc lies inside their convex hull. A linear inequality holding at all three vertices therefore holds on that arc. The filter adds three hard rows per sector:

\[
u^T a\leq B_j-(k_1+k_2)u^T v,
\qquad u\in\{u_-,u_+,u_0\}.
\]

There are 21 obstacle rows in total. At nonzero tilt the angular interval is computed from rotated ray support, not from yaw alone. Invalid projected angles or a collapsed common field of view are explicitly rejected.

### Initialization is checked separately

The HOCBF implication requires both `h≥0` and `ψ1≥0` initially; a feasible acceleration row cannot repair an already invalid premise. For each outer triangle, the code takes `αj=max_u uᵀv` and minimizes

\[
\psi_1(d)=k_1(d^2-R^2)-2d\alpha_j,
\quad d\geq L_j,
\quad d_{\psi}=\max(L_j,\alpha_j/k_1).
\]

It requires `Lj≥R` and `ψ1(dψ)≥0` for every sector at each solve. A failed condition is logged as invalid/infeasible, rather than treated as a valid barrier certificate. Recomputed local scan barriers do not by themselves prove consistent obstacle association across time.

The rollout enrolls an initial state only after both an offline simulator truth-clearance check and the initial sensed-sector/input/angular/HOCBF and hard-QP feasibility checks pass. A rejected trajectory executes zero counted plant steps and remains in the results with `initial_state_safe=false`, its individual truth/CBF/QP flags, and an invalid-initial-state termination. This is an explicit assessment of whether the initial premises hold, not a guarantee that every requested start satisfies them. Simulator truth used for admission is not supplied to the online barrier or learned network, and rejected starts are not silently excluded from aggregate outcomes.

## 4. Visibility, software control bounds, and tracking tube

The common projected forward field of view defines a cone with center direction `nc`, tangent `tc`, and half-angle `φc`. The next requested horizontal velocity `v+Δt a` must satisfy

\[
\pm t_c^T(v+\Delta t a)\leq \tan\phi_c\,n_c^T(v+\Delta t a),
\qquad n_c^T(v+\Delta t a)\geq0.
\]

These three hard rows restrict the requested motion direction. They do not create rear/side obstacle knowledge or guarantee that disturbances preserve the exact velocity cone continuously between samples.

Horizontal software bounds are `−1≤aN,aE≤1 m/s²` and `|a−aprev|≤1·Δt m/s²` per axis. Their intersection is enforced inside the QP. Vertical acceleration is controlled separately within `[−4,4] m/s²` and slewed by at most `0.1 m/s²` per 0.1 s step. Yaw-rate allocation is bounded by 0.5 rad/s, with roll/pitch at most 25°. These are declared design limits, not measured motor limits. Acceleration-to-attitude/thrust allocation uses the calibrated chart formula with no post-QP clipping of a feasible horizontal solution.

The fixed nominal controller tracks a smooth goal reference and repels from perceived sectors. With displacement `Δp=goal−p0`, duration `T=max(1.875||Δp||/2.5,10)` s, and `τ=clip(t/T,0,1)`, the reference is

\[
p_r=p_0+\Delta p(10\tau^3-15\tau^4+6\tau^5).
\]

Its analytic velocity and acceleration are continuous at both endpoints. It targets the actual mission goal, not the recorded VFH-adjusted local reference. This is a new frozen controller for co-design; deploying it requires implementing the same controller, not substituting the original PID/VFH path.

For horizontal tracking error `e=p−pr`, `ė=v−vr` and radius `ρ=2.5 m`, define `ht=ρ²−||e||²`. Its robust HOCBF row is

\[
2e^Ta-\delta\leq
2(\dot\rho^2+\rho\ddot\rho)-2\|\dot e\|^2+2e^Ta_r
+(k_1+k_2)\dot h_t+k_1k_2 h_t-2\|e\|D,
\qquad \delta\geq0.
\]

Here `ρ̇=ρ̈=0` by default. Tracking alone has slack. The obstacle, visibility, and control-bound rows have no slack.

## 5. Differentiable QP and explicit failure semantics

With `z=[aN,aE,δ]`, solve

\[
\min_z\;\tfrac12\|a-a_{nom}\|^2+\lambda\delta^2
\quad\text{subject to}\quad Gz\leq b,\qquad \lambda=100.
\]

The current problem has three decision variables and 30 inequality rows. [`qp.py`](../scripts/ares_safe_training/qp.py) implements a batched FP64 primal-dual interior-point method and an active-KKT implicit backward pass. Neural-network FP16 autocast is disabled for the QP. Gradients are almost-everywhere local derivatives of the selected solution; active-set transitions and degenerate constraints need numerical care. The solver checks convergence and residuals against the original hard inequalities.

The returned control is considered feasible only when input/angular/initial CBF checks, solver feasibility, and convergence all pass. Otherwise it chooses a separate acceleration/slew-bounded emergency brake, sets `feasible=false`, and retains the true inequality residual of that selected command. An infeasible brake is not relabeled a safe QP solution. Training/evaluation records and terminates filter failures; a low aggregate loss must not hide them.

The training objective can backpropagate tracking, control-correction, constraint, and geometric losses through this filter and the true guidance dynamics. Geometry-derived losses are supervision only. They do not supply map coordinates to the online controller or establish a runtime geometry certificate.

## 6. What the calibration and evaluation establish

The saved MATLAB calibration measures the actual installed guidance model and the declared acceleration ramp envelope. Current horizontal `D` is approximately `0.219249 m/s²`, including a `0.05 m/s²` margin; its provenance and coverage remain in the artifact. The validator checks derivatives, continuous/discrete control Jacobians, all 13 state trajectories, native collision positions, allocation, and step-size convergence against saved reference cases. It seals calibration, fixture, and source hashes before Stage 2.

Those checks establish a finite numerical agreement claim. They do not prove that empirical reconstruction/tracking bounds hold uniformly, that sparse rays observe all relevant surfaces, that the QP stays feasible, or that continuous-time HOCBF premises survive sensor switching, sampling, occlusion, and disturbances. The horizontal layer also does not certify altitude, ceiling/ground clearance, moving objects, or arbitrary 3D collision avoidance. Evaluate those outcomes separately on the full 1 ms collision-position grid and report failures, outage exposure, and actual termination reasons.

The appropriate current claim is **a tested implementation of a conditional, map-free differentiable LiDAR safety filter for a calibrated software plant**. A completed Colab experiment must provide its measured held-out reconstruction and closed-loop results. Deployment to Simulink and hardware validation are subsequent tasks.
