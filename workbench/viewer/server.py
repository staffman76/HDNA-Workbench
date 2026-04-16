# Copyright (c) 2026 Chris. All rights reserved.
# Licensed under the Business Source License 1.1 -- see LICENSE file.

"""
Viewer server -- serves the 3D visualization UI and API endpoints.
Uses only Python stdlib (http.server + json). No Flask/FastAPI needed.
"""

import json
import os
import threading
import webbrowser
from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import numpy as np

# Global reference to the adapter (set by launch())
_adapter = None
_trainer = None  # LiveTrainer instance
_static_dir = os.path.join(os.path.dirname(__file__), "static")


class ViewerHandler(SimpleHTTPRequestHandler):
    """Serves static files and JSON API endpoints."""

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        params = parse_qs(parsed.query)

        # API routes
        if path == "/api/model":
            self._json_response(self._get_model_info())
        elif path == "/api/network":
            self._json_response(self._get_network_graph())
        elif path == "/api/neuron":
            nid = int(params.get("id", [0])[0])
            self._json_response(self._get_neuron(nid))
        elif path == "/api/audit":
            count = int(params.get("count", [50])[0])
            self._json_response(self._get_audit(count))
        elif path == "/api/trace":
            step = int(params.get("step", [0])[0])
            self._json_response(self._get_trace(step))
        elif path == "/api/stress":
            self._json_response(self._get_stress())
        elif path == "/api/daemons":
            self._json_response(self._get_daemons())
        elif path == "/api/replay":
            step = int(params.get("step", [0])[0])
            self._json_response(self._get_replay(step))
        elif path == "/" or path == "/index.html":
            self._serve_file("index.html", "text/html")
        elif path == "/app.js":
            self._serve_file("app.js", "application/javascript")
        elif path == "/style.css":
            self._serve_file("style.css", "text/css")
        else:
            self.send_error(404)

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path
        content_len = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_len) if content_len > 0 else b''

        if path == "/api/train/start":
            self._json_response(self._train_start(body))
        elif path == "/api/train/step":
            params = json.loads(body) if body else {}
            count = params.get("count", 1)
            self._json_response(self._train_step(count))
        elif path == "/api/train/stop":
            self._json_response(self._train_stop())
        elif path == "/api/train/status":
            self._json_response(self._train_status())
        else:
            self.send_error(404)

    def _json_response(self, data):
        body = json.dumps(data, default=_serialize).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", len(body))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _serve_file(self, filename, content_type):
        filepath = os.path.join(_static_dir, filename)
        if not os.path.exists(filepath):
            self.send_error(404)
            return
        with open(filepath, "rb") as f:
            body = f.read()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", len(body))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        pass  # suppress request logging

    # --- API data methods ---

    def _get_model_info(self):
        if _adapter is None:
            return {"error": "No model loaded"}
        return _adapter.get_info().to_dict()

    def _get_network_graph(self):
        """Return nodes and edges for 3D visualization."""
        if _adapter is None:
            return {"error": "No model loaded"}

        net = _adapter._network
        nodes = []
        edges = []

        for nid, neuron in net.neurons.items():
            nodes.append({
                "id": nid,
                "layer": neuron.layer,
                "avg_activation": round(neuron.avg_activation, 6),
                "is_dead": neuron.is_dead,
                "tags": list(neuron.tags),
                "bias": round(neuron.bias, 4),
                "n_weights": len(neuron.weights),
                "variance": round(neuron.activation_variance, 6),
            })
            for target_id, strength in neuron.routing:
                edges.append({
                    "source": nid,
                    "target": target_id,
                    "strength": round(float(strength), 4),
                })

        return {
            "nodes": nodes,
            "edges": edges,
            "num_layers": net.num_layers,
            "layer_sizes": net.layer_sizes,
            "input_dim": net.input_dim,
            "output_dim": net.output_dim,
        }

    def _get_neuron(self, neuron_id):
        if _adapter is None:
            return {"error": "No model loaded"}
        return _adapter.get_neuron_state(neuron_id)

    def _get_audit(self, count):
        if _adapter is None:
            return {"error": "No model loaded"}
        try:
            records = list(_adapter._audit.records)[-count:]
            return {
                "records": [r.to_dict() for r in records],
                "stats": _adapter._audit.stats(),
            }
        except Exception as e:
            return {"error": str(e)}

    def _get_trace(self, step):
        """Get trace for a specific audit step."""
        if _adapter is None:
            return {"error": "No model loaded"}
        try:
            record = _adapter._audit._step_index.get(step)
            if record:
                return record.to_dict()
            return {"error": f"No record for step {step}"}
        except Exception as e:
            return {"error": str(e)}

    def _get_stress(self):
        if _adapter is None:
            return {"error": "No model loaded"}
        try:
            return _adapter.get_stress_report()
        except Exception as e:
            return {"error": str(e)}

    def _get_daemons(self):
        if _adapter is None:
            return {"error": "No model loaded"}
        if _adapter._coordinator is None:
            return {"daemons": {}}
        return _adapter._coordinator.snapshot()

    def _get_replay(self, step):
        """Re-run the network for a specific audit step and return per-neuron activations."""
        if _adapter is None:
            return {"error": "No model loaded"}

        try:
            record = _adapter._audit._step_index.get(step)
            if not record:
                return {"error": f"No record for step {step}"}

            net = _adapter._network
            # Run a forward pass and capture per-neuron activation
            # We use the network's current state (neurons remember recent activations)
            # For a true replay we'd need stored inputs, but we can show the
            # neuron activation pattern from their memory at this state
            neuron_activations = {}
            max_act = 0.001  # avoid division by zero

            for nid, neuron in net.neurons.items():
                act = neuron.avg_activation
                neuron_activations[nid] = round(act, 6)
                if act > max_act:
                    max_act = act

            # Normalize to 0-1 for visualization
            neuron_normalized = {}
            for nid, act in neuron_activations.items():
                neuron_normalized[nid] = round(act / max_act, 4)

            # Find which edges are "hot" (both endpoints active)
            hot_edges = []
            for nid, neuron in net.neurons.items():
                src_act = neuron_activations.get(nid, 0)
                if src_act > 0:
                    for target_id, strength in neuron.routing:
                        tgt_act = neuron_activations.get(target_id, 0)
                        if tgt_act > 0:
                            flow = src_act * abs(strength) * tgt_act
                            hot_edges.append({
                                "source": nid,
                                "target": target_id,
                                "flow": round(flow, 6),
                                "strength": round(float(strength), 4),
                            })

            # Sort by flow, keep top edges
            hot_edges.sort(key=lambda e: e["flow"], reverse=True)

            return {
                "step": step,
                "record": record.to_dict(),
                "neuron_activations": neuron_activations,
                "neuron_normalized": neuron_normalized,
                "hot_edges": hot_edges[:50],
                "max_activation": round(max_act, 6),
            }
        except Exception as e:
            return {"error": str(e)}

    # --- Training API ---

    def _train_start(self, body):
        global _trainer
        if _adapter is None:
            return {"error": "No model loaded"}
        if _trainer is not None and _trainer.running:
            return {"error": "Training already running"}

        params = json.loads(body) if body else {}
        curriculum_name = params.get("curriculum", "math")
        phases = params.get("phases", 3)

        try:
            _trainer = LiveTrainer(_adapter, curriculum_name, phases)
            return {"status": "started", "curriculum": curriculum_name, "phases": phases}
        except Exception as e:
            return {"error": str(e)}

    def _train_step(self, count=1):
        global _trainer
        if _trainer is None:
            return {"error": "No training session. Call /api/train/start first."}

        results = []
        for _ in range(int(count)):
            result = _trainer.step()
            if result is None:
                break
            results.append(result)

        # Get current network state for live visualization
        net = _adapter._network
        neuron_states = {}
        max_act = 0.001
        for nid, neuron in net.neurons.items():
            act = neuron.avg_activation
            neuron_states[nid] = {
                "activation": round(act, 6),
                "is_dead": neuron.is_dead,
                "variance": round(neuron.activation_variance, 6),
            }
            if act > max_act:
                max_act = act

        # Normalize
        for nid in neuron_states:
            neuron_states[nid]["normalized"] = round(
                neuron_states[nid]["activation"] / max_act, 4)

        # Edge strengths (may have changed due to learning)
        edges_changed = []
        for nid, neuron in net.neurons.items():
            for target_id, strength in neuron.routing:
                edges_changed.append({
                    "source": nid,
                    "target": target_id,
                    "strength": round(float(strength), 4),
                })

        return {
            "steps": results,
            "total_episodes": _trainer.episode,
            "neuron_states": neuron_states,
            "edges": edges_changed,
            "max_activation": round(max_act, 6),
            "stats": _trainer.stats(),
        }

    def _train_stop(self):
        global _trainer
        if _trainer is None:
            return {"status": "no session"}
        stats = _trainer.stats()
        _trainer = None
        return {"status": "stopped", "stats": stats}

    def _train_status(self):
        if _trainer is None:
            return {"running": False}
        return {"running": True, "stats": _trainer.stats()}


class LiveTrainer:
    """Manages a live training session for the viewer."""

    def __init__(self, adapter, curriculum_name="math", phases=3):
        from ..curricula import math_curriculum, language_curriculum, spatial_curriculum

        self.adapter = adapter
        self.net = adapter._network
        self.brain = adapter._brain
        self.coordinator = adapter._coordinator
        self.rng = np.random.default_rng()
        self.episode = 0
        self.correct = 0
        self.total_reward = 0.0
        self.running = True
        self._recent = []  # last 50 results for rolling accuracy

        # Build curriculum
        if curriculum_name == "language":
            self.curriculum = language_curriculum()
        elif curriculum_name == "spatial":
            self.curriculum = spatial_curriculum(phases=phases)
        else:
            self.curriculum = math_curriculum(phases=phases)

    def step(self):
        """Run one training episode. Returns step result dict."""
        if not self.running:
            return None

        result = self.curriculum.get_task(self.rng)
        if result is None:
            self.running = False
            return None

        level, task = result
        features = task.features if task.features is not None else np.zeros(self.net.input_dim)

        # Get proposals and Q-values
        proposals = []
        if self.coordinator:
            proposals = self.coordinator.collect_proposals(None, features, self.rng)

        q_values = self.brain.get_q_values(features) if self.brain else np.zeros(self.net.output_dim)

        # Select action
        if proposals and self.coordinator:
            selected = self.coordinator.select(proposals, brain_q_values=q_values, rng=self.rng)
            action = int(selected.action) if selected and selected.action is not None else 0
        elif self.brain:
            action = self.brain.select_action(features, self.rng)
        else:
            action = int(np.argmax(self.net.forward(features)))

        # Evaluate
        correct = (action == task.expected_output)
        reward = 1.0 if correct else -0.2

        # Learn
        if self.brain:
            next_features = self.rng.random(len(features))
            self.brain.learn(features, action, reward, next_features, done=False)

        if correct:
            self.correct += 1
        self.total_reward += reward
        self.episode += 1
        level.record_attempt(correct)

        # Record outcome for daemons
        if proposals and self.coordinator and selected:
            self.coordinator.record_outcome(selected, reward)

        # Rolling accuracy
        self._recent.append(1 if correct else 0)
        if len(self._recent) > 50:
            self._recent.pop(0)

        # Record in audit log
        from ..core.audit import PredictionRecord
        self.adapter._audit.record(PredictionRecord(
            step=self.episode,
            chosen_class=action,
            confidence=float(np.max(q_values)) if len(q_values) > 0 else 0,
            source="live_train",
            correct=correct,
            reward=reward,
        ))

        return {
            "episode": self.episode,
            "action": action,
            "expected": int(task.expected_output),
            "correct": correct,
            "reward": round(reward, 2),
            "level": level.name,
            "accuracy_50": round(sum(self._recent) / max(1, len(self._recent)), 4),
            "epsilon": round(self.brain.epsilon, 4) if self.brain else 0,
        }

    def stats(self):
        return {
            "episode": self.episode,
            "correct": self.correct,
            "accuracy": round(self.correct / max(1, self.episode), 4),
            "accuracy_50": round(sum(self._recent) / max(1, len(self._recent)), 4),
            "total_reward": round(self.total_reward, 4),
            "running": self.running,
            "curriculum": self.curriculum.progress,
        }


def _serialize(obj):
    """JSON serializer for numpy types."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, set):
        return list(obj)
    return str(obj)


def launch(adapter, port=8420, open_browser=True):
    """
    Launch the viewer for an adapter.

    Args:
        adapter: any ModelAdapter (HDNAAdapter for full features)
        port: HTTP port (default 8420)
        open_browser: auto-open browser (default True)
    """
    global _adapter
    _adapter = adapter

    server = HTTPServer(("127.0.0.1", port), ViewerHandler)
    url = f"http://127.0.0.1:{port}"
    print(f"HDNA Workbench Viewer running at {url}")
    print("Press Ctrl+C to stop.")

    if open_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nViewer stopped.")
        server.server_close()
