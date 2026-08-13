# TORCS AI

Research tooling for training and evaluating racing agents against the native
Windows TORCS installation. The current development target is a competitive,
hierarchical discrete policy: the learned policy chooses tactical racing
intents, while a deterministic controller handles smooth steering, speed,
gears, and safety recovery. A PPO entry point is provided for the canonical
nine-action environment; the older DQN implementation remains a legacy
checkpoint-compatible path until its feature schema is versioned.

The project is not yet a production-trained champion. Performance claims are
made only from the versioned benchmark reports described below.

## Native simulator layout

The supported local installation is:

```text
TORCS game:       C:\torcs\torcs
Executable:       C:\torcs\torcs\wtorcs.exe
AI project:       C:\torcs\gym_torcs
Runtime staging:  C:\torcs\gym_torcs\.runtime\
```

The installed game is treated as read-only. The runtime manager stages an
isolated copy before starting a race, writes logs/configuration only into that
copy, and verifies the installed executable and simulator assets by checksum.
Do not place checkpoints, race outputs, or generated XML in the installation
directory.

The installed distribution currently provides the SCR server, ten SCR driver
slots, ports 3001–3010, the `car1-trb1` SCR car, the benchmark tracks, and the
built-in drivers used by the competitive suite.

## Setup

Use Python 3.11 or 3.12. From PowerShell:

```powershell
cd C:\torcs\gym_torcs
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

The simulator path can be overridden for a different local installation:

```powershell
$env:TORCS_HOME = 'C:\torcs\torcs'
```

## Verify the installation

Run the read-only doctor before training:

```powershell
python scripts\torcs_doctor.py --torcs-home C:\torcs\torcs
python scripts\torcs_doctor.py --torcs-home C:\torcs\torcs --json
```

The doctor verifies `wtorcs.exe`, the SCR server DLL/configuration, the ten
driver slots, required tracks, required opponent modules, and records SHA-256
identities. It does not start TORCS or modify files.

## Native smoke test

The first runtime probe stages a copy and checks process startup, SCR
identification, telemetry, one bounded action, and cleanup. The command below
is intentionally bounded and leaves `C:\torcs\torcs` untouched:

```powershell
python scripts\native_smoke.py --torcs-home C:\torcs\torcs --steps 1000
# Full local release gate, including an installed-tree fingerprint check:
.\scripts\verify_native_release.ps1 -TorcsHome C:\torcs\torcs -Steps 1000
```

The process manager uses `shell=False` and owns the child PID. Global process
kills such as `pkill torcs` or `taskkill /IM wtorcs.exe` are not part of the
supported workflow.

## Current agent contract

The canonical tactical action space has exactly nine actions:

```text
0 left + brake       1 left + hold        2 left + push
3 center + brake     4 center + hold      5 center + push
6 right + brake      7 right + hold       8 right + push
```

The policy must always emit an integer in `[0, 8]`. Longitudinal actions never
apply throttle and brake simultaneously. A slew limiter and safety shield
convert the tactical intent into valid TORCS actuator values. Shield
interventions are logged and reported; they are not hidden as model skill.

The competitive telemetry encoder is versioned as
`competitive-telemetry-v1`. It produces a finite 118-value `float32`
observation containing ego telemetry, track rays, opponent ranges and closing
rates, previous applied controls, traffic clearance, and race context. Frame
stacking is not silently assumed; an experiment that adds it must version the
observation schema and checkpoint contract.

## Development commands

```text
torcs-ai doctor
torcs-ai simulator prepare
torcs-ai simulator probe
torcs-ai collect
torcs-ai train --config <file>
torcs-ai evaluate --checkpoint <file>
torcs-ai benchmark --suite competitive-v1
torcs-ai report --run <run-id>
```

The command surface is being introduced incrementally. Until the full CLI is
available, `scripts\torcs_doctor.py` and the native smoke script are the
authoritative runtime checks.

The first real-policy path is explicit and bounded:

```powershell
python scripts\train_native_agent.py --timesteps 100000 --max-steps 15000 --output runs\ppo_native
# Optional warm start from the audited tactical teacher:
python scripts\train_native_agent.py --timesteps 100000 --max-steps 15000 `
  --expert-episodes 1 --bc-epochs 8 --teacher-guidance 0.25 `
  --output runs\ppo_teacher
python scripts\evaluate_native_agent.py --model runs\ppo_native.zip --episodes 3 --max-steps 10000
# Add --visual to watch the PPO model drive the TORCS window:
python scripts\evaluate_native_agent.py --model runs\ppo_native.zip --track road/forza --episodes 1 --max-steps 15000 --visual
# Track-specific runs use only the doctor-approved allowlist:
python scripts\train_native_agent.py --track road/alpine-1 --timesteps 100000
# Seeded multi-track training reuses one isolated runtime per track:
python scripts\train_native_agent.py `
  --track road/alpine-1 --track road/forza --track oval/michigan `
  --timesteps 300000 --max-steps 15000 --expert-episodes 1 `
  --bc-epochs 8 --teacher-guidance 0.25 --output runs\ppo_matrix
python scripts\benchmark_native.py --model runs\ppo_native.zip --track road/ruudskogen --episodes 3 --max-steps 10000
# Matrix mode keeps one isolated runtime per track:
python scripts\benchmark_native.py --model runs\ppo_native.zip `
  --track road/alpine-1 --track road/forza --track oval/michigan `
  --episodes 3 --max-steps 10000
```

These commands require the `rl` extra, stage a private runtime copy, and close
the simulator they started. They do not claim competitiveness until a frozen
benchmark report compares completion, pace, damage, and position outcomes to
the fixed-driver baseline.

Teacher guidance is an explicit training-only reward term: it rewards matching
the deterministic telemetry-driven tactical controller and records its
coefficient in the run manifest. Benchmark environments disable guidance, so
reported learned-policy results remain independent of the teacher reward.
Repeated `--track` arguments produce a matrix artifact while retaining the
single-track fields for existing consumers.

Competitive training rejects short episode horizons and insufficient total
experience by default. A multi-track run must permit at least 5,000 steps per
episode and allocate at least one maximum-length episode per selected track.
`--allow-smoke-training` exists only for transport/CI smoke checks and its
artifacts are not competitive evidence. The behavioural-cloning warm start
collects complete teacher races, balances the nine tactical classes, records
its action counts and training accuracy, and then lets PPO optimize the v3
progress/position/safety reward. Evaluation remains teacher-free.

The low-level controller uses an explicit, manifest-recorded driving profile.
The generic forward-ray speed envelope is used for Alpine and Michigan. Forza
uses a calibrated short-horizon hairpin brake floor (`speed_limit_scale=0.72`)
because its forward rays collapse before a tight corner; this profile is
applied consistently to the teacher, actuator controller, and benchmark
environment. Unknown tracks keep the generic profile and must be revalidated
before being used for competitive claims.

## Evaluation policy

Racing policies are not scored by classification accuracy. The benchmark
reports:

- completion, finish, podium, and win rates;
- median and interquartile-mean return;
- conditional lap time and distance before failure;
- off-track events, collision events, and damage per kilometre;
- clean overtakes and lost positions;
- safety-shield interventions per kilometre;
- action distribution, dominant-action share, teacher agreement, and an
  explicit action-collapse flag;
- p50/p95 inference latency; and
- bootstrap confidence intervals over independent seeds.

Training tracks are `road/alpine-1`, `road/forza`, and `oval/michigan`.
Validation uses `road/ruudskogen`. Held-out testing uses `road/spring` and
`road/street-1`. The built-in roster is `berniw`, `bt`, `inferno`, `olethros`,
and `tita`. Hyperparameters and reward weights are selected only on training
and validation tracks.

The first competitive release requires reliable solo completion before pace,
then traffic completion before overtaking, and finally held-out races against
fixed bots and frozen self-play opponents. Negative results remain valid
research outcomes.

## Artifacts and reproducibility

Runs are stored under `runs/<run-id>/` and must contain the resolved
configuration, Git commit, simulator and track checksums, environment/action/
reward versions, seed, hardware/dependency metadata, episode metrics,
checkpoint checksum, and logs. Checkpoints are written atomically and loaded
only through the trusted state-dict path.

## CI

Pull-request CI runs Python compilation, tests, package installation, and the
non-simulator checks on Python 3.11 and 3.12. A Windows runner with access to
`C:\torcs\torcs` is required for the native launch, SCR handshake, multi-slot,
lap-completion, and process-cleanup release gate. Hosted CI must not claim that
native simulator tests ran when the installation is unavailable.

## Licensing

The repository’s Python code and the installed TORCS distribution are separate
artifacts. TORCS documentation identifies GPL-covered components and artwork
with additional restrictions. The simulator tree and a public simulator image
must not be redistributed until a component-by-component licence and
attribution audit is complete.

## Status

The native runtime, read-only installation doctor, isolated staging, strict
SCR client, telemetry schema, and nine-action contract are under active
development. The full competitive benchmark and trained policy have not yet
been frozen or published.
