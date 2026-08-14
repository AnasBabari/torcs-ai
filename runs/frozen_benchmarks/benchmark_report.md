# TORCS Racing Agent Benchmark Report

| Track | Model / Controller | Finish Rate | Win Rate | Med Position | Damage/km | Shield/km | Dom Action % | Collapsed | p50 Latency (ms) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `road/alpine-1` | **Learned PPO** | 0% | 0% | 1.0 | 1188.3 | 0.0 | 58.9% | NO | 0.79 |
| `road/alpine-1` | **Fixed Center** | 0% | 0% | 1.0 | 987.9 | 35.7 | 100.0% | YES | 0.00 |
| `road/alpine-1` | **Expert Teacher** | 0% | 0% | 1.0 | 885.1 | 0.0 | 58.2% | NO | 0.07 |
| `road/ruudskogen` | **Learned PPO** | 0% | 0% | 1.0 | 0.0 | 17.9 | 75.0% | NO | 0.69 |
| `road/ruudskogen` | **Fixed Center** | 0% | 0% | 1.0 | 0.0 | 15.4 | 100.0% | YES | 0.00 |
| `road/ruudskogen` | **Expert Teacher** | 0% | 0% | 1.0 | 0.0 | 18.2 | 69.6% | NO | 0.07 |
| `road/spring` | **Learned PPO** | 0% | 0% | 1.0 | 0.0 | 1.8 | 59.6% | NO | 0.72 |
| `road/spring` | **Fixed Center** | 0% | 0% | 1.0 | 0.0 | 1.3 | 100.0% | YES | 0.00 |
| `road/spring` | **Expert Teacher** | 0% | 0% | 1.0 | 0.0 | 1.8 | 55.8% | NO | 0.05 |
| `road/street-1` | **Learned PPO** | 0% | 0% | 1.0 | 517.2 | 5.3 | 50.9% | NO | 0.64 |
| `road/street-1` | **Fixed Center** | 0% | 0% | 1.0 | 481.8 | 97.3 | 100.0% | YES | 0.00 |
| `road/street-1` | **Expert Teacher** | 0% | 0% | 1.0 | 354.2 | 8.5 | 53.2% | NO | 0.06 |

## Competitiveness Gate Hierarchy
### Track: `road/alpine-1` (FAILED)
- **gate_1_completion**: FAIL
- **gate_2_traffic_completion**: PASS
- **gate_3_damage_control**: FAIL
- **gate_4_pace_near_baseline**: FAIL
- **gate_5_position_overtaking**: PASS
- **gate_6_action_diversity**: PASS
### Track: `road/ruudskogen` (FAILED)
- **gate_1_completion**: FAIL
- **gate_2_traffic_completion**: PASS
- **gate_3_damage_control**: PASS
- **gate_4_pace_near_baseline**: FAIL
- **gate_5_position_overtaking**: PASS
- **gate_6_action_diversity**: PASS
### Track: `road/spring` (FAILED)
- **gate_1_completion**: FAIL
- **gate_2_traffic_completion**: PASS
- **gate_3_damage_control**: PASS
- **gate_4_pace_near_baseline**: FAIL
- **gate_5_position_overtaking**: PASS
- **gate_6_action_diversity**: PASS
### Track: `road/street-1` (FAILED)
- **gate_1_completion**: FAIL
- **gate_2_traffic_completion**: PASS
- **gate_3_damage_control**: FAIL
- **gate_4_pace_near_baseline**: FAIL
- **gate_5_position_overtaking**: PASS
- **gate_6_action_diversity**: PASS
