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
from pathlib import Path
from urllib.parse import urlparse, parse_qs
import numpy as np

# Global references (set by launch())
_adapter = None          # primary model (HDNA)
_models = {}             # name -> ModelAdapter (all loaded models)
_trainer = None          # LiveTrainer instance
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
        elif path == "/api/save":
            self._json_response(self._save_model())
        elif path == "/api/load":
            self._json_response(self._load_model())
        elif path == "/api/models":
            self._json_response(self._list_models())
        elif path == "/api/curricula":
            self._json_response(self._list_curricula())
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

        if path == "/api/models/load":
            self._json_response(self._load_external_model(body))
        elif path == "/api/models/list":
            self._json_response(self._list_models())
        elif path == "/api/models/inspect":
            params = json.loads(body) if body else {}
            self._json_response(self._inspect_model(params))
        elif path == "/api/models/compare":
            params = json.loads(body) if body else {}
            self._json_response(self._compare_models(params))
        elif path == "/api/curricula/list":
            self._json_response(self._list_curricula())
        elif path == "/api/curricula/load_file":
            self._json_response(self._load_curriculum_file(body))
        elif path == "/api/network/rebuild":
            self._json_response(self._rebuild_network(body))
        elif path == "/api/train/start":
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

    # --- Save / Load ---

    def _save_model(self):
        if _adapter is None:
            return {"error": "No model loaded"}
        try:
            save_dir = Path(_static_dir).parent / "saves"
            save_dir.mkdir(exist_ok=True)
            save_path = save_dir / "model_state.json"

            data = {
                "network": _adapter._network.to_dict(),
                "brain": {
                    "epsilon": _adapter._brain.epsilon if _adapter._brain else 0.3,
                    "episodes": _adapter._brain.episodes if _adapter._brain else 0,
                    "total_reward": _adapter._brain.total_reward if _adapter._brain else 0,
                    "lr": _adapter._brain.lr if _adapter._brain else 0.01,
                },
                "audit_stats": _adapter._audit.stats() if _adapter._audit else {},
            }

            save_path.write_text(json.dumps(data, default=_serialize, indent=2))
            return {"status": "saved", "path": str(save_path),
                    "neurons": len(_adapter._network.neurons)}
        except Exception as e:
            return {"error": str(e)}

    def _load_model(self):
        if _adapter is None:
            return {"error": "No model loaded"}
        try:
            save_dir = Path(_static_dir).parent / "saves"
            save_path = save_dir / "model_state.json"

            if not save_path.exists():
                return {"error": "No saved model found"}

            data = json.loads(save_path.read_text())

            from ..core.neuron import HDNANetwork
            loaded_net = HDNANetwork.from_dict(data["network"])

            # Replace the adapter's network
            _adapter._network = loaded_net
            if _adapter._brain:
                _adapter._brain.net = loaded_net
                brain_data = data.get("brain", {})
                _adapter._brain.epsilon = brain_data.get("epsilon", 0.3)
                _adapter._brain.episodes = brain_data.get("episodes", 0)
                _adapter._brain.total_reward = brain_data.get("total_reward", 0)

            return {"status": "loaded", "neurons": len(loaded_net.neurons),
                    "path": str(save_path)}
        except Exception as e:
            return {"error": str(e)}

    # --- Curriculum Management ---

    def _list_curricula(self):
        from ..curricula.registry import list_curricula
        return {"curricula": list_curricula()}

    def _load_curriculum_file(self, body):
        params = json.loads(body) if body else {}
        file_path = params.get("path", "")
        name = params.get("name", "")

        if not file_path:
            return {"error": "No file path provided"}

        try:
            from ..curricula.registry import load_curriculum_file, register_curriculum
            curriculum = load_curriculum_file(file_path)
            reg_name = name or curriculum.name
            register_curriculum(
                reg_name,
                lambda **kw: load_curriculum_file(file_path),
                description=f"Loaded from {file_path}",
                tags=["custom", "file"],
            )
            return {
                "status": "loaded",
                "name": reg_name,
                "levels": len(curriculum.levels),
                "total_tasks": sum(len(l.tasks) for l in curriculum.levels),
            }
        except Exception as e:
            return {"error": str(e)}

    # --- Network Configuration ---

    def _rebuild_network(self, body):
        """Rebuild the HDNA network with new dimensions."""
        global _adapter
        if _adapter is None:
            return {"error": "No model loaded"}

        params = json.loads(body) if body else {}
        input_dim = params.get("input_dim", 24)
        output_dim = params.get("output_dim", 5)
        hidden_dims = params.get("hidden_dims", [32, 16])
        lr = params.get("learning_rate", 0.01)
        epsilon = params.get("epsilon", 0.3)

        try:
            from ..core.neuron import HDNANetwork
            from ..core.brain import Brain

            rng = np.random.default_rng()
            net = HDNANetwork(input_dim=input_dim, output_dim=output_dim,
                              hidden_dims=hidden_dims, rng=rng)
            brain = Brain(net, epsilon=epsilon, learning_rate=lr)

            # Warm up
            for _ in range(50):
                net.forward(rng.random(input_dim))

            _adapter._network = net
            _adapter._brain = brain
            _adapter._brain.net = net

            return {
                "status": "rebuilt",
                "input_dim": input_dim,
                "output_dim": output_dim,
                "hidden_dims": hidden_dims,
                "neurons": len(net.neurons),
                "connections": sum(len(n.routing) for n in net.neurons.values()),
            }
        except Exception as e:
            return {"error": str(e)}

    # --- Multi-Model Management ---

    def _load_external_model(self, body):
        """Load a PyTorch, HuggingFace, or ONNX model."""
        global _models
        params = json.loads(body) if body else {}
        model_type = params.get("type", "pytorch")  # pytorch, huggingface, onnx
        model_path = params.get("path", "")
        model_name = params.get("name", model_path or "external_model")

        try:
            if model_type == "huggingface":
                from ..adapters.huggingface_adapter import HuggingFaceAdapter
                adapter = HuggingFaceAdapter.from_pretrained(model_path)
                adapter._name = model_name

            elif model_type == "pytorch":
                import torch
                from ..adapters.pytorch_adapter import PyTorchAdapter
                import workbench

                if model_path and os.path.exists(model_path):
                    model = torch.load(model_path, map_location="cpu",
                                       weights_only=False)
                else:
                    # Demo: create a small model
                    model = torch.nn.Sequential(
                        torch.nn.Linear(24, 32),
                        torch.nn.ReLU(),
                        torch.nn.Linear(32, 16),
                        torch.nn.ReLU(),
                        torch.nn.Linear(16, 5),
                    )
                    model_name = model_name or "PyTorch Demo MLP"

                model = workbench.inspect(model)
                adapter = PyTorchAdapter(model, name=model_name, inspected=True)

            elif model_type == "onnx":
                from ..adapters.onnx_adapter import ONNXAdapter
                adapter = ONNXAdapter(model_path, name=model_name)

            else:
                return {"error": f"Unknown model type: {model_type}"}

            _models[model_name] = adapter
            info = adapter.get_info()
            return {
                "status": "loaded",
                "name": model_name,
                "info": info.to_dict(),
                "capabilities": str(adapter.capabilities()),
                "total_models": len(_models),
            }

        except Exception as e:
            return {"error": str(e)}

    def _list_models(self):
        """List all loaded models with their info and capabilities."""
        global _models

        models = {}
        # Always include the primary HDNA model
        if _adapter:
            info = _adapter.get_info()
            models["HDNA (primary)"] = {
                "info": info.to_dict(),
                "capabilities": str(_adapter.capabilities()),
                "type": "hdna",
                "is_primary": True,
            }

        for name, adapter in _models.items():
            info = adapter.get_info()
            models[name] = {
                "info": info.to_dict(),
                "capabilities": str(adapter.capabilities()),
                "type": info.framework,
                "is_primary": False,
            }

        return {"models": models, "count": len(models)}

    def _inspect_model(self, params):
        """Run inspection on a specific loaded model."""
        model_name = params.get("name", "")
        input_data = params.get("input")

        adapter = _models.get(model_name)
        if not adapter:
            return {"error": f"Model '{model_name}' not found"}

        from ..tools.inspector import Inspector
        inspector = Inspector(adapter)
        result = inspector.summary()

        # Run activation flow if input provided
        if input_data is not None:
            try:
                inp = np.array(input_data, dtype=np.float32)
                result["activation_flow"] = inspector.activation_flow(inp)
                result["attention"] = inspector.attention_analysis(inp)
            except Exception as e:
                result["activation_error"] = str(e)

        # Layer list
        try:
            result["layers"] = adapter.list_layers()
        except Exception:
            pass

        return result

    def _compare_models(self, params):
        """Compare two or more models on the same input."""
        model_names = params.get("models", [])
        input_data = params.get("input")

        if not input_data:
            # Generate a random test input
            input_data = np.random.random(24).tolist()

        inp = np.array(input_data, dtype=np.float32)

        results = {}

        # Include primary HDNA model
        if _adapter:
            try:
                output = _adapter.predict(inp)
                results["HDNA (primary)"] = {
                    "output": _safe_output(output),
                    "framework": "hdna",
                }
            except Exception as e:
                results["HDNA (primary)"] = {"error": str(e)}

        # Include requested models
        for name in model_names:
            adapter = _models.get(name)
            if not adapter:
                results[name] = {"error": "not found"}
                continue
            try:
                output = adapter.predict(inp.reshape(1, -1))
                results[name] = {
                    "output": _safe_output(output),
                    "framework": adapter.get_info().framework,
                }

                # Get activations if available
                from ..adapters.protocol import Capability
                if adapter.has(Capability.ACTIVATIONS):
                    acts = adapter.get_activations(inp.reshape(1, -1))
                    results[name]["activations"] = [
                        {"layer": a.layer_name, "shape": list(a.shape),
                         "mean": round(float(np.mean(a.values)), 6),
                         "std": round(float(np.std(a.values)), 6)}
                        for a in acts
                    ]

                if adapter.has(Capability.ATTENTION):
                    attns = adapter.get_attention(inp.reshape(1, -1))
                    results[name]["attention"] = [
                        {"layer": a.layer_name, "heads": a.num_heads,
                         "metadata": a.metadata}
                        for a in attns
                    ]

            except Exception as e:
                results[name] = {"error": str(e)}

        return {"input_shape": list(inp.shape), "results": results}

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

    def __init__(self, adapter, curriculum_name="classification", phases=3):
        from ..curricula.registry import get_curriculum
        from ..core.stress import StressMonitor, HomeostasisDaemon, apply_interventions

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
        self._interventions = []  # log of homeostasis actions

        # Stress monitoring and homeostasis
        self.monitor = StressMonitor()
        self.homeostasis = HomeostasisDaemon(monitor=self.monitor)

        # Build curriculum from registry
        self.curriculum = get_curriculum(curriculum_name, phases=phases)
        if self.curriculum is None:
            # Fallback
            from ..curricula import classification_curriculum
            self.curriculum = classification_curriculum()

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
        selected = None
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

        # Learn (with stronger signal)
        if self.brain:
            # Use actual next task features instead of random noise
            next_result = self.curriculum.get_task(self.rng)
            if next_result:
                _, next_task = next_result
                next_features = next_task.features if next_task.features is not None else self.rng.random(len(features))
            else:
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

        # Teach daemons from correct answers
        # (The brain routes, but daemons learn the actual patterns)
        for daemon in self.coordinator.daemons.values():
            if hasattr(daemon, 'learn_from_outcome'):
                daemon.learn_from_outcome(features, task.expected_output, True)

        # Rolling accuracy
        self._recent.append(1 if correct else 0)
        if len(self._recent) > 50:
            self._recent.pop(0)

        # --- Homeostasis: health check and intervention every 50 episodes ---
        intervention_info = None
        if self.episode % 50 == 0 and self.episode > 20:
            from ..core.stress import apply_interventions
            report = self.monitor.snapshot(self.net, self.episode)

            # Check for dead neurons and fix them
            dead_count = sum(1 for n in self.net.neurons.values()
                            if n.is_dead and "output" not in n.tags)
            total_hidden = sum(1 for n in self.net.neurons.values()
                              if "output" not in n.tags)

            if dead_count > total_hidden * 0.3:
                # Too many dead neurons — prune and spawn fresh ones
                pruned = self.net.prune_dead_neurons()
                spawned = 0
                # Spawn replacements in the most depleted layer
                layer_sizes = self.net.layer_sizes
                for layer_idx in range(1, self.net.num_layers - 1):
                    current = layer_sizes.get(layer_idx, 0)
                    # Spawn up to the original size
                    needed = max(0, min(len(pruned), 4) - spawned)
                    for _ in range(needed):
                        prev_layer = layer_idx - 1
                        prev_neurons = self.net.get_layer_neurons(prev_layer) if prev_layer > 0 else []
                        n_in = len(prev_neurons) if prev_neurons else self.net.input_dim
                        nid = self.net.add_neuron(
                            n_inputs=n_in, layer=layer_idx,
                            tags={"hidden", "spawned"}, rng=self.rng
                        )
                        # Connect to/from adjacent layers
                        if prev_neurons:
                            for pn in prev_neurons:
                                strength = self.rng.normal(0, np.sqrt(2.0 / n_in))
                                self.net.connect(pn.neuron_id, nid, float(strength))
                        next_neurons = self.net.get_layer_neurons(layer_idx + 1)
                        for nn in next_neurons:
                            strength = self.rng.normal(0, np.sqrt(2.0 / max(1, current + 1)))
                            self.net.connect(nid, nn.neuron_id, float(strength))
                        spawned += 1

                intervention_info = {
                    "type": "homeostasis",
                    "pruned": len(pruned),
                    "spawned": spawned,
                    "dead_before": dead_count,
                }
                self._interventions.append(intervention_info)

            # Adaptive learning rate: boost if accuracy is stuck
            if self.brain and len(self._recent) >= 50:
                acc = sum(self._recent) / len(self._recent)
                if acc < 0.25:
                    # Stuck at random chance — increase exploration and learning rate
                    self.brain.epsilon = min(0.6, self.brain.epsilon + 0.05)
                    self.brain.lr = min(0.05, self.brain.lr * 1.2)
                elif acc > 0.6:
                    # Doing well — tighten exploration
                    self.brain.epsilon = max(0.05, self.brain.epsilon * 0.95)

        # Decay epsilon normally
        if self.brain and self.episode % 10 == 0:
            self.brain.epsilon = max(
                self.brain.epsilon_min,
                self.brain.epsilon * self.brain.epsilon_decay
            )

        # Record in audit log
        from ..core.audit import PredictionRecord
        self.adapter._audit.record(PredictionRecord(
            step=self.episode,
            chosen_class=action,
            confidence=float(np.exp(q_values[action]) / (np.sum(np.exp(q_values - np.max(q_values))) + 1e-8)) if len(q_values) > 0 else 0,
            source="live_train",
            correct=correct,
            reward=reward,
        ))

        result_dict = {
            "episode": self.episode,
            "action": action,
            "expected": int(task.expected_output),
            "correct": correct,
            "reward": round(reward, 2),
            "level": level.name,
            "accuracy_50": round(sum(self._recent) / max(1, len(self._recent)), 4),
            "epsilon": round(self.brain.epsilon, 4) if self.brain else 0,
            "lr": round(self.brain.lr, 5) if self.brain else 0,
        }
        if intervention_info:
            result_dict["intervention"] = intervention_info
        return result_dict

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


def _safe_output(output):
    """Convert model output to JSON-safe format."""
    arr = np.asarray(output).flatten()
    if len(arr) <= 20:
        return arr.round(4).tolist()
    return {"shape": list(np.asarray(output).shape),
            "mean": round(float(arr.mean()), 4),
            "std": round(float(arr.std()), 4)}


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
