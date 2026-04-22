# Copyright (c) 2026 Chris. All rights reserved.
# Licensed under the Business Source License 1.1 — see LICENSE file.

"""
Daemon Framework — Pluggable reasoning modules for HDNA networks.

A daemon is a specialized reasoner that proposes actions. The coordinator
collects proposals from all daemons and the brain selects among them.

Researchers can create custom daemons by subclassing Daemon and implementing
the reason() method. That's it — the framework handles registration, proposal
routing, phase progression, and audit logging.

Proven daemon types across projects:
- Curiosity (exploration), Arithmetic, Comparison, Probability (math domain)
- Planning, Causality, Repair, Chain (spatial domain)
- Sentiment, Topic, Emotion, Intent (language domain)
- Homeostasis (network health — domain-agnostic)
"""

import time
import numpy as np
from dataclasses import dataclass, field
from typing import Any, Optional, Callable
from enum import IntEnum


class Phase(IntEnum):
    """Daemon maturity phases — quality-gated, not time-gated."""
    APPRENTICE = 0    # learning, low confidence
    JOURNEYMAN = 1    # reliable on known patterns
    COMPETENT = 2     # can handle novel inputs
    EXPERT = 3        # trusted for autonomous decisions
    INDEPENDENT = 4   # minimal oversight needed


@dataclass
class Proposal:
    """
    A daemon's output — a proposed action with reasoning.

    The coordinator collects these from all active daemons and the brain
    routes to the best one. Full traceability: you can see exactly which
    daemon proposed what and why.
    """
    action: Any                    # domain-specific (int, str, dict, etc.)
    confidence: float              # 0.0 to 1.0
    reasoning: str                 # human-readable explanation
    source: str                    # daemon name
    features_used: list = None     # which input features drove this proposal
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "action": self.action if not hasattr(self.action, 'tolist') else self.action.tolist(),
            "confidence": round(self.confidence, 4),
            "reasoning": self.reasoning,
            "source": self.source,
            "metadata": self.metadata,
        }


class Daemon:
    """
    Base class for all reasoning daemons.

    Subclass this and implement reason() to create a new daemon.
    The framework handles everything else.

    Example:
        class MyDaemon(Daemon):
            def reason(self, state, features, rng=None):
                if features[0] > 0.5:
                    return Proposal(action=1, confidence=0.8,
                                    reasoning="feature[0] is high",
                                    source=self.name)
                return None  # abstain
    """

    def __init__(self, name: str, domain: str = "general",
                 description: str = ""):
        self.name = name
        self.domain = domain
        self.description = description
        self.phase = Phase.APPRENTICE
        self.proposals_made = 0
        self.proposals_accepted = 0
        self.total_reward = 0.0
        self.enabled = True
        self._phase_history: list = []

    def reason(self, state: Any, features: np.ndarray,
               rng: np.random.Generator = None) -> Optional[Proposal]:
        """
        Propose an action given the current state and features.

        Override this in your subclass. Return a Proposal or None to abstain.

        Args:
            state: Domain-specific state object (game board, text, etc.)
            features: Numeric feature vector extracted from state
            rng: Random number generator for stochastic reasoning
        """
        raise NotImplementedError("Subclass must implement reason()")

    def record_outcome(self, accepted: bool, reward: float):
        """Called by the coordinator after a decision is made."""
        self.proposals_made += 1
        if accepted:
            self.proposals_accepted += 1
            self.total_reward += reward

    @property
    def acceptance_rate(self) -> float:
        if self.proposals_made == 0:
            return 0.0
        return self.proposals_accepted / self.proposals_made

    @property
    def avg_reward(self) -> float:
        if self.proposals_accepted == 0:
            return 0.0
        return self.total_reward / self.proposals_accepted

    def advance_phase(self) -> bool:
        """
        Try to advance to the next phase. Returns True if advanced.

        Default criteria (override for custom progression):
        - APPRENTICE → JOURNEYMAN: 50+ proposals, acceptance > 30%
        - JOURNEYMAN → COMPETENT: 200+ proposals, acceptance > 50%, avg_reward > 0
        - COMPETENT → EXPERT: 500+ proposals, acceptance > 60%, avg_reward > 0.1
        - EXPERT → INDEPENDENT: 1000+ proposals, acceptance > 70%, avg_reward > 0.2
        """
        thresholds = {
            Phase.APPRENTICE: (50, 0.3, -999),
            Phase.JOURNEYMAN: (200, 0.5, 0.0),
            Phase.COMPETENT: (500, 0.6, 0.1),
            Phase.EXPERT: (1000, 0.7, 0.2),
        }

        if self.phase >= Phase.INDEPENDENT:
            return False

        min_proposals, min_acceptance, min_reward = thresholds[self.phase]
        if (self.proposals_made >= min_proposals and
            self.acceptance_rate >= min_acceptance and
            self.avg_reward >= min_reward):
            old = self.phase
            self.phase = Phase(self.phase + 1)
            self._phase_history.append({
                "from": old.name, "to": self.phase.name,
                "at_proposal": self.proposals_made,
                "acceptance_rate": round(self.acceptance_rate, 3),
                "avg_reward": round(self.avg_reward, 4),
            })
            return True
        return False

    def snapshot(self) -> dict:
        """Full daemon state for inspection."""
        return {
            "name": self.name,
            "domain": self.domain,
            "description": self.description,
            "phase": self.phase.name,
            "enabled": self.enabled,
            "proposals_made": self.proposals_made,
            "proposals_accepted": self.proposals_accepted,
            "acceptance_rate": round(self.acceptance_rate, 4),
            "avg_reward": round(self.avg_reward, 4),
            "total_reward": round(self.total_reward, 4),
            "phase_history": self._phase_history,
        }

    def to_dict(self) -> dict:
        return self.snapshot()


class Coordinator:
    """
    Collects proposals from all daemons and routes to the brain for selection.

    The coordinator implements scaffold decay: early in training, it favors
    high-confidence proposals (handholding). Later, it trusts the brain's
    Q-values to route between daemons (autonomy).
    """

    def __init__(self, scaffold_decay_rate: float = 0.001,
                 scaffold_floor: float = 0.4):
        self.daemons: dict[str, Daemon] = {}
        self.scaffold_strength: float = 1.0  # 1.0 = full scaffold, 0.0 = brain only
        self.scaffold_decay_rate = scaffold_decay_rate
        self.scaffold_floor = scaffold_floor  # never go below this
        self.decisions_made: int = 0
        self._decision_log: list = []
        self._log_capacity: int = 1000
        # Names of daemons that actually produced a proposal in the most
        # recent collect_proposals() call. record_outcome() uses this so
        # abstainers are not charged with a "made proposal".
        self._last_proposers: set[str] = set()

    def register(self, daemon: Daemon):
        """Add a daemon to the coordinator."""
        self.daemons[daemon.name] = daemon

    def unregister(self, name: str):
        """Remove a daemon."""
        self.daemons.pop(name, None)

    def collect_proposals(self, state: Any, features: np.ndarray,
                          rng: np.random.Generator = None) -> list:
        """Ask all enabled daemons for proposals."""
        proposals = []
        self._last_proposers = set()
        for daemon in self.daemons.values():
            if not daemon.enabled:
                continue
            try:
                proposal = daemon.reason(state, features, rng)
                if proposal is not None:
                    proposals.append(proposal)
                    self._last_proposers.add(daemon.name)
            except Exception as e:
                # Daemons should not crash the system. A crash still counts
                # as participation (the daemon tried), so add to proposers.
                proposals.append(Proposal(
                    action=None, confidence=0.0,
                    reasoning=f"ERROR: {e}", source=daemon.name,
                ))
                self._last_proposers.add(daemon.name)
        return proposals

    def select(self, proposals: list, brain_q_values: np.ndarray = None,
               rng: np.random.Generator = None) -> Optional[Proposal]:
        """
        Select the best proposal using scaffold + brain routing.

        Early: highest confidence wins (scaffold).
        Late: brain Q-values route between daemon proposals.
        Transition: blended scoring.
        """
        if not proposals:
            return None

        r = rng or np.random.default_rng()

        # Score each proposal
        scores = []
        for i, p in enumerate(proposals):
            # Scaffold component: raw confidence
            scaffold_score = p.confidence

            # Brain component: Q-value for this daemon's action
            brain_score = 0.0
            if brain_q_values is not None and p.action is not None:
                try:
                    idx = int(p.action) if isinstance(p.action, (int, float)) else hash(str(p.action)) % len(brain_q_values)
                    if 0 <= idx < len(brain_q_values):
                        brain_score = brain_q_values[idx]
                except (ValueError, TypeError):
                    brain_score = 0.0

            # Blended score
            score = (self.scaffold_strength * scaffold_score +
                     (1 - self.scaffold_strength) * brain_score)
            scores.append(score)

        # Select (with small epsilon for exploration)
        best_idx = int(np.argmax(scores))
        selected = proposals[best_idx]

        # Record the decision
        self.decisions_made += 1
        log_entry = {
            "step": self.decisions_made,
            "selected": selected.to_dict(),
            "alternatives": [p.to_dict() for p in proposals if p is not selected],
            "scores": [round(s, 4) for s in scores],
            "scaffold_strength": round(self.scaffold_strength, 4),
        }
        self._decision_log.append(log_entry)
        if len(self._decision_log) > self._log_capacity:
            self._decision_log.pop(0)

        # Decay scaffold (but never below floor — daemons always have influence)
        self.scaffold_strength = max(self.scaffold_floor,
                                     self.scaffold_strength - self.scaffold_decay_rate)

        return selected

    def record_outcome(self, proposal: Proposal, reward: float):
        """Propagate outcome back to the daemon that made the proposal.

        Only daemons that actually proposed this round are charged with a
        non-selection outcome. Abstaining daemons (reason() returned None)
        are not counted as having made a proposal — otherwise their
        proposals_made would inflate and make phase thresholds meaningless.
        """
        if proposal.source in self.daemons:
            daemon = self.daemons[proposal.source]
            daemon.record_outcome(accepted=True, reward=reward)
            daemon.advance_phase()

        # Record non-selection only for the other daemons that actually
        # proposed this round.
        for name in self._last_proposers:
            if name != proposal.source and name in self.daemons:
                self.daemons[name].record_outcome(accepted=False, reward=0.0)

    def snapshot(self) -> dict:
        """Full coordinator state."""
        return {
            "num_daemons": len(self.daemons),
            "decisions_made": self.decisions_made,
            "scaffold_strength": round(self.scaffold_strength, 4),
            "daemons": {name: d.snapshot() for name, d in self.daemons.items()},
            "recent_decisions": self._decision_log[-10:],
        }
