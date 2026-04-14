# Compliance & Standards Alignment

HDNA Workbench provides built-in capabilities that map directly to requirements in the EU AI Act, NIST AI Risk Management Framework, ISO/IEC 42001, and emerging US regulatory guidance. This document maps specific Workbench features to specific regulatory requirements.

This is not legal advice. Consult qualified legal counsel for compliance decisions specific to your organization and use case.

---

## EU AI Act

The EU AI Act (Regulation 2024/1689) establishes requirements for high-risk AI systems. HDNA Workbench addresses several key articles.

### Article 12 — Record-Keeping

**Requirement:** High-risk AI systems shall allow for automatic recording of events (logs) over their lifetime. Logging capabilities must ensure traceability of the AI system's functioning throughout its lifecycle.

**Workbench Feature:** `AuditLog` — append-only decision log that records every prediction with:
- Timestamp and step number
- Chosen class, confidence, and alternatives considered
- Which neurons fired and their activation levels
- The routing path (causal chain of the decision)
- Source path (fast or shadow) and the reason for routing
- Ground truth outcome (backfilled after evaluation)

```python
from workbench.core.audit import AuditLog
audit = AuditLog(capacity=100000)
# Every prediction automatically logged
# Query with: audit.accuracy(), audit.explain(step), audit.save("log.json")
```

**Coverage:** Full. Every decision is logged with sufficient detail to reconstruct the reasoning chain.

### Article 13 — Transparency and Provision of Information to Deployers

**Requirement:** High-risk AI systems shall be designed and developed in such a way to ensure that their operation is sufficiently transparent to enable deployers to interpret the system's output and use it appropriately. An appropriate type and degree of transparency shall be ensured.

**Workbench Features:**

| Requirement | Feature | Detail |
|------------|---------|--------|
| Interpret system output | `DecisionReplay` | Full causal chain replay for any prediction |
| Understand capabilities and limitations | `Inspector` | Model summary, health check, anomaly detection |
| Detect potential misuse | `StressMonitor` | Detects dead neurons, jitter, drift, degradation |
| Understand input characteristics | `Curriculum` + feature extraction | Documented feature engineering for each domain |

```python
from workbench.tools import Inspector, DecisionReplay
inspector = Inspector(adapter)
inspector.print_summary()  # capabilities, limitations, health

replayer = DecisionReplay(adapter)
replayer.print_trace(input_data=sample)  # full transparency into one decision
```

**Coverage:** Full for HDNA models (Tier 3). Partial for external models (Tier 1-2, limited by framework access).

### Article 14 — Human Oversight

**Requirement:** High-risk AI systems shall be designed and developed in such a way that they can be effectively overseen by natural persons during the period in which the system is in use. Human oversight measures shall aim to prevent or minimize the risks to health, safety, or fundamental rights.

**Workbench Features:**

| Requirement | Feature | Detail |
|------------|---------|--------|
| Understand system capacities | `Inspector.summary()` | Full model overview with capability assessment |
| Correctly interpret outputs | `DecisionReplay.print_trace()` | Human-readable explanation of each decision |
| Decide not to use / override | `Breakpoints` | Halt execution when outputs meet anomaly criteria |
| Intervene in operation | `ModelAdapter.intervene()` | Modify activations at any layer in real time |
| Monitor for anomalies | `StressMonitor`, `workbench.anomalies()` | Continuous health monitoring with alerts |

```python
# Set breakpoint: halt if confidence drops below threshold
layer.add_breakpoint(lambda l, inp, out: out.max() < 0.3)

# Monitor health continuously
report = monitor.snapshot(net, episode)
if report.warnings:
    print(f"ALERT: {report.warnings}")
```

**Coverage:** Full for HDNA. The daemon phase progression system explicitly models trust (Apprentice to Independent), providing a built-in human oversight ramp.

### Article 15 — Accuracy, Robustness and Cybersecurity

**Requirement:** High-risk AI systems shall be designed and developed in such a way that they achieve an appropriate level of accuracy, robustness, and cybersecurity, and perform consistently in those respects throughout their lifecycle.

**Workbench Features:**

| Requirement | Feature | Detail |
|------------|---------|--------|
| Accuracy measurement | `AuditLog.accuracy()`, `Curriculum.progress` | Rolling and lifetime accuracy tracking |
| Robustness monitoring | `StressMonitor` | Dead neuron detection, jitter, drift, weight explosion |
| Lifecycle consistency | `ShadowHDNA` | Two-path architecture with stress-gated graduation/degradation |
| Forgetting detection | `Curriculum.check_forgetting()` | Detects catastrophic forgetting of previously mastered capabilities |
| Bias detection | `DaemonStudio.analyze()` | Calibration analysis, error pattern breakdown by class |

```python
# Check for catastrophic forgetting
forgotten = curriculum.check_forgetting()
if forgotten:
    for f in forgotten:
        print(f"DEGRADATION: {f['name']} dropped to {f['accuracy']:.1%}")

# Continuous robustness monitoring
if not monitor.is_healthy():
    print("ALERT: Network health degraded")
```

**Coverage:** Accuracy and robustness are well covered. Cybersecurity is outside the scope of this tool (address at the infrastructure level).

---

## NIST AI Risk Management Framework (AI RMF 1.0)

The NIST AI RMF organizes AI risk management into four functions: Govern, Map, Measure, and Manage.

### GOVERN — Establishing AI governance

| NIST Requirement | Workbench Feature |
|-----------------|------------------|
| Document AI system behavior | `ARCHITECTURE.md`, `AuditLog`, `Exporter` |
| Maintain audit trails | `AuditLog` — append-only with full decision chains |
| Define roles and responsibilities | Daemon `Phase` progression documents system trust levels |

### MAP — Contextualizing AI risks

| NIST Requirement | Workbench Feature |
|-----------------|------------------|
| Identify intended use and limitations | `Inspector.summary()` reports capabilities per adapter tier |
| Document data characteristics | `Curriculum` system documents training task properties |
| Assess potential impacts | `Experiment Forge` A/B tests configurations for impact analysis |

### MEASURE — Analyzing and assessing AI risks

| NIST Requirement | Workbench Feature |
|-----------------|------------------|
| Assess accuracy | `AuditLog.accuracy()`, per-level mastery in `Curriculum` |
| Evaluate fairness | `DaemonStudio.analyze()` — error pattern analysis by class |
| Monitor for drift | `StressMonitor` — continuous weight drift and jitter tracking |
| Test robustness | `Experiment Forge` — controlled A/B testing under different conditions |
| Measure interpretability | Native to HDNA (Tier 3); `DecisionReplay` for all tiers |

### MANAGE — Prioritizing and acting on AI risks

| NIST Requirement | Workbench Feature |
|-----------------|------------------|
| Mitigate identified risks | `HomeostasisDaemon` — proposes network repairs (prune, spawn, dampen) |
| Monitor in deployment | `ShadowHDNA` — continuous background learning with degradation detection |
| Document incidents | `AuditLog.record_event()` — logs disagreements, degradation, anomalies |
| Enable human intervention | `Breakpoints`, `intervene()`, scaffold decay with human-controlled parameters |

---

## ISO/IEC 42001 — AI Management System

ISO/IEC 42001 specifies requirements for establishing, implementing, maintaining, and continually improving an AI management system.

| ISO/IEC 42001 Area | Workbench Feature |
|-------------------|------------------|
| **A.6.2.5 — AI system documentation** | `ARCHITECTURE.md`, `Inspector.summary()`, `Exporter.summary_report()` |
| **A.6.2.6 — Data for AI systems** | `Curriculum` system documents all training data characteristics |
| **A.8.4 — AI system operation and monitoring** | `StressMonitor`, `AuditLog`, `ShadowHDNA` degradation detection |
| **A.8.5 — Recording and reporting** | `AuditLog` with `save()`, `Exporter` for CSV/JSON/text reports |
| **A.9.3 — Management review** | `Inspector.health()`, `Experiment.print_report()`, `Exporter.summary_report()` |
| **A.10.1 — Continual improvement** | `ShadowHDNA` continuous learning, `Curriculum.check_forgetting()` |

---

## Practical Compliance Workflow

For organizations using HDNA Workbench in a regulated environment:

### 1. Before Deployment

```python
from workbench.tools import Inspector, Exporter
from workbench.adapters import HDNAAdapter

adapter = HDNAAdapter(network=net, brain=brain, coordinator=coordinator)
inspector = Inspector(adapter)

# Generate pre-deployment documentation
exporter = Exporter("./compliance_docs")
exporter.summary_report(inspector, "model_card.txt", input_data=sample)
exporter.network_state(net, "initial_state.json")
exporter.table(curriculum.snapshot(), "training_progress.csv")
```

### 2. During Operation

```python
from workbench.core.stress import StressMonitor
from workbench.core.audit import AuditLog

monitor = StressMonitor()
audit = AuditLog(capacity=1000000)

# Every prediction is logged automatically
# Periodic health checks:
report = monitor.snapshot(net, episode)
if report.warnings:
    log_alert(report.warnings)  # your alerting system

# Periodic compliance reports:
exporter.table({"accuracy": audit.accuracy(), 
                "novelty_rate": audit.novelty_rate(),
                "predictions": audit.stats()["total_predictions"]},
               f"compliance_report_{date}.csv")
```

### 3. Incident Investigation

```python
# Replay any specific decision
print(audit.explain(step=12345))

# Full causal chain
from workbench.tools import DecisionReplay
replayer = DecisionReplay(adapter)
trace = replayer.trace(incident_input)
replayer.print_trace(trace)

# Export evidence
exporter.trace_log([trace], "incident_12345_trace.json")
```

### 4. Periodic Review

```python
# Check for degradation
forgotten = curriculum.check_forgetting()
health = inspector.health()

# Compare current vs baseline
from workbench.tools import ModelComparison
comp = ModelComparison()
comp.add("current", current_adapter)
comp.add("baseline", baseline_adapter)
comp.run(validation_inputs, labels=validation_labels)
comp.print_report()
```

---

## Feature-to-Regulation Matrix

| Workbench Feature | EU AI Act Art. 12 | EU AI Act Art. 13 | EU AI Act Art. 14 | EU AI Act Art. 15 | NIST Govern | NIST Map | NIST Measure | NIST Manage | ISO 42001 |
|---|---|---|---|---|---|---|---|---|---|
| AuditLog | X | X | | X | X | | X | X | X |
| DecisionReplay | | X | X | | | | X | | |
| Inspector | | X | X | | X | X | | X | X |
| StressMonitor | | | X | X | | | X | X | X |
| HomeostasisDaemon | | | | X | | | | X | X |
| ShadowHDNA | | | | X | | | X | X | X |
| Breakpoints | | | X | | | | | X | |
| Curriculum | | | | X | | X | X | | X |
| Experiment Forge | | | | X | | X | X | | |
| DaemonStudio | | | | X | | | X | | |
| Exporter | X | X | | | X | X | X | X | X |
| Neuron Inspection | | X | X | | | | X | | |

---

## Limitations

HDNA Workbench addresses transparency, interpretability, record-keeping, and monitoring requirements. It does not address:

- **Cybersecurity requirements** (Art. 15) — address at infrastructure level
- **Data governance** (Art. 10) — use dedicated data governance tools
- **Conformity assessment procedures** (Art. 43) — requires a notified body
- **Registration obligations** (Art. 49) — administrative, not technical
- **Fundamental rights impact assessments** (Art. 27) — requires organizational process

The compliance features work at full depth (Tier 3) with HDNA models. External models connected via adapters provide compliance capabilities proportional to their inspection depth (Tier 1-2).

For commercial compliance deployments, [contact us for a commercial license](mailto:chris@hdna-workbench.com) with compliance documentation support.
