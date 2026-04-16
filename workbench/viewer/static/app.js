// HDNA Workbench Viewer - 3D Model Visualization
// Copyright (c) 2026 Chris. All rights reserved.

const LAYER_COLORS = [
    0x00d4ff, // input - cyan
    0x00e676, // hidden 1 - green
    0xffab40, // hidden 2 - orange
    0xb388ff, // hidden 3 - purple
    0xff5252, // output - red
    0xffd740, // extra layers...
    0x69f0ae,
    0xff80ab,
];

const DEAD_COLOR = 0x444444;
const EDGE_COLOR = 0x1a3a5e;
const EDGE_HIGHLIGHT = 0x00d4ff;

class HDNAViewer {
    constructor() {
        this.scene = null;
        this.camera = null;
        this.renderer = null;
        this.neuronMeshes = {};    // neuron_id -> mesh
        this.edgeLines = [];
        this.labels = [];
        this.networkData = null;
        this.auditData = null;
        this.selectedNeuron = null;
        this.showEdges = true;
        this.showDead = false;
        this.showLabels = false;
        this.traceStep = 0;
        this.maxStep = 0;
        this.isReplaying = false;

        // Training state
        this.isTraining = false;
        this.trainInterval = null;
        this.trainSpeed = 5;

        // Mouse interaction
        this.raycaster = new THREE.Raycaster();
        this.mouse = new THREE.Vector2();
        this.isDragging = false;
        this.isRightDrag = false;
        this.prevMouse = { x: 0, y: 0 };
        this.rotX = 0.3;
        this.rotY = 0;
        this.panX = 0;
        this.panY = 0;
        this.zoom = 5;

        this.init();
        this.loadData();
    }

    init() {
        const canvas = document.getElementById('canvas3d');
        const viewport = document.getElementById('viewport');

        // Scene
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0x1a1a2e);

        // Camera
        this.camera = new THREE.PerspectiveCamera(60, 1, 0.1, 1000);
        this.camera.position.set(0, 2, 5);

        // Renderer
        this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
        this.renderer.setPixelRatio(window.devicePixelRatio);

        // Lights
        const ambient = new THREE.AmbientLight(0x404060, 0.6);
        this.scene.add(ambient);

        const directional = new THREE.DirectionalLight(0xffffff, 0.8);
        directional.position.set(5, 10, 5);
        this.scene.add(directional);

        const point = new THREE.PointLight(0x00d4ff, 0.4, 50);
        point.position.set(0, 5, 0);
        this.scene.add(point);

        // Events
        canvas.addEventListener('mousedown', (e) => this.onMouseDown(e));
        canvas.addEventListener('mousemove', (e) => this.onMouseMove(e));
        canvas.addEventListener('mouseup', (e) => this.onMouseUp(e));
        canvas.addEventListener('wheel', (e) => this.onWheel(e));
        canvas.addEventListener('click', (e) => this.onClick(e));
        canvas.addEventListener('contextmenu', (e) => e.preventDefault());
        window.addEventListener('resize', () => this.onResize());

        this.onResize();
        this.animate();
    }

    onResize() {
        const viewport = document.getElementById('viewport');
        const w = viewport.clientWidth;
        const h = viewport.clientHeight;
        this.camera.aspect = w / h;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(w, h);
    }

    // --- Camera controls ---

    onMouseDown(e) {
        this.isDragging = true;
        this.isRightDrag = e.button === 2;
        this.prevMouse = { x: e.clientX, y: e.clientY };
    }

    onMouseMove(e) {
        if (!this.isDragging) return;
        const dx = e.clientX - this.prevMouse.x;
        const dy = e.clientY - this.prevMouse.y;

        if (this.isRightDrag) {
            this.panX += dx * 0.01;
            this.panY -= dy * 0.01;
        } else {
            this.rotY += dx * 0.005;
            this.rotX += dy * 0.005;
            this.rotX = Math.max(-Math.PI / 2, Math.min(Math.PI / 2, this.rotX));
        }
        this.prevMouse = { x: e.clientX, y: e.clientY };
    }

    onMouseUp(e) {
        this.isDragging = false;
    }

    onWheel(e) {
        this.zoom += e.deltaY * 0.005;
        this.zoom = Math.max(1, Math.min(20, this.zoom));
        e.preventDefault();
    }

    onClick(e) {
        const viewport = document.getElementById('viewport');
        const rect = viewport.getBoundingClientRect();
        this.mouse.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
        this.mouse.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;

        this.raycaster.setFromCamera(this.mouse, this.camera);
        const meshes = Object.values(this.neuronMeshes);
        const intersects = this.raycaster.intersectObjects(meshes);

        if (intersects.length > 0) {
            const mesh = intersects[0].object;
            this.selectNeuron(mesh.userData.neuronId);
        }
    }

    resetCamera() {
        this.rotX = 0.3;
        this.rotY = 0;
        this.panX = 0;
        this.panY = 0;
        this.zoom = 5;
    }

    updateCamera() {
        const x = this.zoom * Math.sin(this.rotY) * Math.cos(this.rotX) + this.panX;
        const y = this.zoom * Math.sin(this.rotX) + this.panY;
        const z = this.zoom * Math.cos(this.rotY) * Math.cos(this.rotX);
        this.camera.position.set(x, y, z);
        this.camera.lookAt(this.panX, this.panY, 0);
    }

    // --- Data loading ---

    async loadData() {
        try {
            const [modelRes, networkRes, auditRes, stressRes, daemonRes] = await Promise.all([
                fetch('/api/model').then(r => r.json()),
                fetch('/api/network').then(r => r.json()),
                fetch('/api/audit?count=200').then(r => r.json()),
                fetch('/api/stress').then(r => r.json()),
                fetch('/api/daemons').then(r => r.json()),
            ]);

            // Header stats
            document.getElementById('model-name').textContent = modelRes.name || 'HDNA Model';
            document.getElementById('stat-neurons').textContent = networkRes.nodes ? networkRes.nodes.length : 0;
            document.getElementById('stat-connections').textContent = networkRes.edges ? networkRes.edges.length : 0;
            document.getElementById('stat-layers').textContent = networkRes.num_layers || 0;

            const healthy = stressRes.is_healthy !== false;
            const healthEl = document.getElementById('stat-health');
            healthEl.textContent = healthy ? 'Healthy' : 'Warning';
            healthEl.style.color = healthy ? 'var(--green)' : 'var(--orange)';

            this.networkData = networkRes;
            this.auditData = auditRes;

            this.buildNetwork(networkRes);
            this.buildNeuronList(networkRes);
            this.buildAuditTrail(auditRes);
            this.buildHealthPanel(stressRes, networkRes);
            this.buildDaemonPanel(daemonRes);

            // Set trace slider range
            if (auditRes.records && auditRes.records.length > 0) {
                this.maxStep = auditRes.records.length - 1;
                document.getElementById('trace-slider').max = this.maxStep;
                document.getElementById('trace-slider').value = this.maxStep;
                this.traceStep = this.maxStep;
                this.updateTraceDisplay();
            }
        } catch (err) {
            console.error('Failed to load data:', err);
        }
    }

    // --- 3D Network ---

    buildNetwork(data) {
        if (!data.nodes) return;

        const numLayers = data.num_layers || 1;

        // Layout: layers spread on X axis, neurons spread on Y within each layer
        const layerGroups = {};
        data.nodes.forEach(n => {
            if (!layerGroups[n.layer]) layerGroups[n.layer] = [];
            layerGroups[n.layer].push(n);
        });

        const layerSpacing = 2.5;
        const neuronSpacing = 0.6;

        data.nodes.forEach(node => {
            const layer = node.layer;
            const layerNodes = layerGroups[layer] || [];
            const idx = layerNodes.indexOf(node);
            const yOffset = (layerNodes.length - 1) / 2;

            const x = (layer - (numLayers - 1) / 2) * layerSpacing;
            const y = (idx - yOffset) * neuronSpacing;
            const z = (Math.sin(idx * 0.7) * 0.3); // slight depth variation

            // Size based on activation
            const baseSize = 0.12;
            const actSize = Math.min(0.3, baseSize + node.avg_activation * 0.4);
            const size = node.is_dead ? baseSize * 0.6 : actSize;

            // Color
            const color = node.is_dead ? DEAD_COLOR : (LAYER_COLORS[layer] || LAYER_COLORS[0]);

            const geometry = new THREE.SphereGeometry(size, 16, 12);
            const material = new THREE.MeshPhongMaterial({
                color: color,
                emissive: color,
                emissiveIntensity: node.is_dead ? 0.05 : 0.2 + node.avg_activation * 0.3,
                transparent: node.is_dead && !this.showDead,
                opacity: (node.is_dead && !this.showDead) ? 0.15 : 1.0,
            });

            const mesh = new THREE.Mesh(geometry, material);
            mesh.position.set(x, y, z);
            mesh.userData = { neuronId: node.id, node: node };
            this.scene.add(mesh);
            this.neuronMeshes[node.id] = mesh;
        });

        // Edges
        this.buildEdges(data.edges);

        // Layer legend
        this.buildLayerLegend(layerGroups);
    }

    buildEdges(edges) {
        if (!edges) return;

        edges.forEach(edge => {
            const srcMesh = this.neuronMeshes[edge.source];
            const tgtMesh = this.neuronMeshes[edge.target];
            if (!srcMesh || !tgtMesh) return;

            const points = [srcMesh.position.clone(), tgtMesh.position.clone()];
            const geometry = new THREE.BufferGeometry().setFromPoints(points);

            const strength = Math.abs(edge.strength);
            const opacity = Math.min(0.6, 0.05 + strength * 0.5);
            const color = edge.strength > 0 ? 0x00d4ff : 0xff5252;

            const material = new THREE.LineBasicMaterial({
                color: color,
                transparent: true,
                opacity: this.showEdges ? opacity : 0,
            });

            const line = new THREE.Line(geometry, material);
            line.userData = { edge: edge };
            this.scene.add(line);
            this.edgeLines.push(line);
        });
    }

    buildLayerLegend(layerGroups) {
        const legend = document.getElementById('layer-legend');
        legend.innerHTML = '';
        Object.keys(layerGroups).sort((a, b) => a - b).forEach(layer => {
            const color = LAYER_COLORS[layer] || LAYER_COLORS[0];
            const hex = '#' + color.toString(16).padStart(6, '0');
            const count = layerGroups[layer].length;
            legend.innerHTML += `
                <div class="layer-dot">
                    <div class="dot" style="background:${hex}"></div>
                    L${layer} (${count})
                </div>`;
        });
    }

    // --- Neuron selection ---

    async selectNeuron(neuronId) {
        // Highlight in 3D
        Object.values(this.neuronMeshes).forEach(m => {
            m.material.emissiveIntensity = m.userData.node.is_dead ? 0.05 :
                0.2 + m.userData.node.avg_activation * 0.3;
        });

        const mesh = this.neuronMeshes[neuronId];
        if (mesh) {
            mesh.material.emissiveIntensity = 1.0;
        }

        // Highlight edges
        this.edgeLines.forEach(line => {
            const e = line.userData.edge;
            if (e.source === neuronId || e.target === neuronId) {
                line.material.color.setHex(EDGE_HIGHLIGHT);
                line.material.opacity = 0.8;
            } else {
                const str = Math.abs(e.strength);
                line.material.color.setHex(e.strength > 0 ? 0x00d4ff : 0xff5252);
                line.material.opacity = this.showEdges ? Math.min(0.6, 0.05 + str * 0.5) : 0;
            }
        });

        // Fetch full neuron data
        const res = await fetch(`/api/neuron?id=${neuronId}`).then(r => r.json());
        this.selectedNeuron = neuronId;

        // If replaying, also fetch replay context for this neuron
        if (this.isReplaying && this.auditData && this.auditData.records[this.traceStep]) {
            const record = this.auditData.records[this.traceStep];
            try {
                const replay = await fetch(`/api/replay?step=${record.step}`).then(r => r.json());
                this.showNeuronDetail(res, replay);
            } catch (err) {
                this.showNeuronDetail(res);
            }
        } else {
            this.showNeuronDetail(res);
        }
    }

    showNeuronDetail(data, replayData) {
        const panel = document.getElementById('neuron-detail');
        const title = document.getElementById('nd-title');
        const body = document.getElementById('nd-body');

        title.textContent = `Neuron #${data.id}`;

        // Replay context (if stepping)
        let html = '';
        if (replayData) {
            const act = replayData.neuron_activations[data.id] || 0;
            const norm = replayData.neuron_normalized[data.id] || 0;
            const record = replayData.record || {};
            const isChosen = this.networkData &&
                data.layer === this.networkData.num_layers - 1 &&
                this.networkData.nodes.filter(n => n.layer === data.layer)
                    .findIndex(n => n.id === data.id) === record.chosen_class;

            const actBarW = Math.min(120, norm * 120);
            const actColor = act > 0 ? 'var(--green)' : 'var(--red)';

            html += `<div style="background:var(--bg-dark);padding:6px 8px;border-radius:4px;margin-bottom:8px">
                <div style="color:var(--orange);font-weight:600;font-size:11px;margin-bottom:4px">
                    Step ${replayData.step}${isChosen ? (record.correct ? ' -- CHOSEN (correct)' : ' -- CHOSEN (wrong)') : ''}
                </div>
                <div class="stat-row"><span class="label">Activation</span>
                    <span class="value" style="color:${actColor}">${act.toFixed(6)}
                        <span class="act-bar" style="width:${actBarW}px;background:${actColor}"></span>
                    </span>
                </div>
                <div class="stat-row"><span class="label">Normalized</span>
                    <span class="value">${(norm * 100).toFixed(1)}%</span>
                </div>`;

            // Show which hot edges involve this neuron
            const hotIn = (replayData.hot_edges || []).filter(e => e.target === data.id);
            const hotOut = (replayData.hot_edges || []).filter(e => e.source === data.id);
            if (hotIn.length > 0) {
                html += `<div style="font-size:10px;color:var(--text-dim);margin-top:4px">Signal from: ${hotIn.map(e => '#' + e.source).join(', ')}</div>`;
            }
            if (hotOut.length > 0) {
                html += `<div style="font-size:10px;color:var(--text-dim)">Signal to: ${hotOut.map(e => '#' + e.target).join(', ')}</div>`;
            }

            html += `</div>`;
        }

        // Static neuron info
        html += `
            <div class="stat-row"><span class="label">Layer</span><span class="value">${data.layer}</span></div>
            <div class="stat-row"><span class="label">Tags</span><span class="value">${(data.tags || []).join(', ') || 'none'}</span></div>
            <div class="stat-row"><span class="label">Avg Activation</span><span class="value">${(data.avg_activation || 0).toFixed(6)}</span></div>
            <div class="stat-row"><span class="label">Variance</span><span class="value">${(data.activation_variance || 0).toFixed(6)}</span></div>
            <div class="stat-row"><span class="label">Dead</span><span class="value ${data.is_dead ? 'bad' : 'good'}">${data.is_dead ? 'YES' : 'No'}</span></div>
            <div class="stat-row"><span class="label">Weights</span><span class="value">${data.n_weights || 0}</span></div>
            <div class="stat-row"><span class="label">Bias</span><span class="value">${(data.bias || 0).toFixed(4)}</span></div>
            <div class="stat-row"><span class="label">Routes Out</span><span class="value">${data.n_routes || 0}</span></div>
        `;

        if (data.weight_stats) {
            const ws = data.weight_stats;
            html += `
                <div style="margin-top:8px;color:var(--accent);font-weight:600;font-size:11px">Weight Stats</div>
                <div class="stat-row"><span class="label">Mean</span><span class="value">${ws.mean.toFixed(6)}</span></div>
                <div class="stat-row"><span class="label">Std</span><span class="value">${ws.std.toFixed(6)}</span></div>
                <div class="stat-row"><span class="label">Range</span><span class="value">[${ws.min.toFixed(4)}, ${ws.max.toFixed(4)}]</span></div>
            `;
        }

        if (data.routing && data.routing.length > 0) {
            html += `<div style="margin-top:8px;color:var(--accent);font-weight:600;font-size:11px">Routing (top 5)</div>`;
            data.routing.slice(0, 5).forEach(([tid, str]) => {
                const barW = Math.min(60, Math.abs(str) * 100);
                const barColor = str > 0 ? 'var(--accent)' : 'var(--red)';
                html += `<div class="stat-row">
                    <span class="label">&rarr; #${tid}</span>
                    <span class="value">${str.toFixed(4)} <span class="act-bar" style="width:${barW}px;background:${barColor}"></span></span>
                </div>`;
            });
        }

        body.innerHTML = html;
        panel.classList.add('visible');
    }

    async refreshSelectedNeuron(replayData) {
        // Refresh the neuron detail panel with current replay context
        if (this.selectedNeuron === null) return;
        const panel = document.getElementById('neuron-detail');
        if (!panel.classList.contains('visible')) return;

        try {
            const res = await fetch(`/api/neuron?id=${this.selectedNeuron}`).then(r => r.json());
            this.showNeuronDetail(res, replayData);
        } catch (err) {
            console.error('Failed to refresh neuron:', err);
        }
    }

    closeNeuronDetail() {
        document.getElementById('neuron-detail').classList.remove('visible');
        // Reset highlights
        Object.values(this.neuronMeshes).forEach(m => {
            m.material.emissiveIntensity = m.userData.node.is_dead ? 0.05 :
                0.2 + m.userData.node.avg_activation * 0.3;
        });
        this.edgeLines.forEach(line => {
            const e = line.userData.edge;
            const str = Math.abs(e.strength);
            line.material.color.setHex(e.strength > 0 ? 0x00d4ff : 0xff5252);
            line.material.opacity = this.showEdges ? Math.min(0.6, 0.05 + str * 0.5) : 0;
        });
        this.selectedNeuron = null;
    }

    // --- Side panel builders ---

    buildNeuronList(data) {
        const container = document.getElementById('neuron-list');
        if (!data.nodes) { container.innerHTML = '<div class="loading">No data</div>'; return; }

        // Group by layer
        const layers = {};
        data.nodes.forEach(n => {
            if (!layers[n.layer]) layers[n.layer] = [];
            layers[n.layer].push(n);
        });

        let html = `<div class="neuron-row" style="border-bottom:1px solid var(--border);padding-bottom:6px;margin-bottom:6px;font-weight:600;color:var(--text-dim);cursor:default">
            <span class="nid">ID</span>
            <span class="layer-tag">Layer</span>
            <span class="activation">Activation</span>
            <span class="status">Status</span>
        </div>`;
        Object.keys(layers).sort((a, b) => a - b).forEach(layer => {
            const neurons = layers[layer];
            const alive = neurons.filter(n => !n.is_dead).length;
            const color = LAYER_COLORS[layer] || LAYER_COLORS[0];
            const hex = '#' + color.toString(16).padStart(6, '0');

            html += `<div class="card">
                <div class="card-header" onclick="this.nextElementSibling.classList.toggle('collapsed')">
                    <h3 style="color:${hex}">Layer ${layer} (${alive}/${neurons.length} active)</h3>
                    <span class="toggle">&#9660;</span>
                </div>
                <div class="card-body">`;

            neurons.sort((a, b) => b.avg_activation - a.avg_activation).forEach(n => {
                const barW = Math.min(40, n.avg_activation * 80);
                html += `<div class="neuron-row" onclick="app.selectNeuron(${n.id})">
                    <span class="nid">#${n.id}</span>
                    <span class="layer-tag">L${n.layer}</span>
                    <span class="activation">${n.avg_activation.toFixed(4)}
                        <span class="act-bar" style="width:${barW}px"></span>
                    </span>
                    <span class="status ${n.is_dead ? 'dead' : 'active'}">${n.is_dead ? 'DEAD' : 'OK'}</span>
                </div>`;
            });

            html += `</div></div>`;
        });

        container.innerHTML = html;
    }

    buildAuditTrail(data) {
        // Stats
        const statsEl = document.getElementById('audit-stats');
        if (data.stats) {
            const s = data.stats;
            statsEl.innerHTML = `<div class="card-body">
                <div class="stat-row"><span class="label">Total Predictions</span><span class="value">${s.total_predictions}</span></div>
                <div class="stat-row"><span class="label">Accuracy (last 100)</span><span class="value ${s.accuracy_100 > 0.5 ? 'good' : s.accuracy_100 > 0.25 ? 'warn' : 'bad'}">${(s.accuracy_100 * 100).toFixed(1)}%</span></div>
                <div class="stat-row"><span class="label">Novelty Rate</span><span class="value">${(s.novelty_rate * 100).toFixed(1)}%</span></div>
            </div>`;
        }

        // Records
        const listEl = document.getElementById('audit-list');
        if (!data.records || data.records.length === 0) {
            listEl.innerHTML = '<div class="loading">No audit records</div>';
            return;
        }

        let html = `<div class="audit-row" style="border-bottom:1px solid var(--border);padding-bottom:6px;margin-bottom:6px;font-weight:600;color:var(--text-dim);cursor:default">
            <span class="step">Step</span>
            <span class="prediction">Class</span>
            <span class="confidence">Conf</span>
            <span class="result">Result</span>
            <span style="font-size:10px">Source</span>
        </div>`;
        data.records.slice().reverse().forEach((r, i) => {
            html += `<div class="audit-row" onclick="app.seekTrace(${data.records.length - 1 - i})">
                <span class="step">${r.step}</span>
                <span class="prediction">C${r.chosen_class}</span>
                <span class="confidence">${(r.confidence * 100).toFixed(0)}%</span>
                <span class="result ${r.correct ? 'correct' : 'wrong'}">${r.correct ? 'OK' : 'X'}</span>
                <span style="color:var(--text-dim);font-size:10px">${r.source}</span>
            </div>`;
        });

        listEl.innerHTML = html;
    }

    buildHealthPanel(stress, network) {
        const el = document.getElementById('health-content');
        let html = '<div class="card"><div class="card-body">';

        const healthy = stress.is_healthy !== false;
        html += `<div class="stat-row"><span class="label">Status</span>
            <span class="value ${healthy ? 'good' : 'bad'}">${healthy ? 'Healthy' : 'Warning'}</span></div>`;
        html += `<div class="stat-row"><span class="label">Dead Neurons</span>
            <span class="value ${stress.dead_pct > 25 ? 'bad' : stress.dead_pct > 10 ? 'warn' : 'good'}">${(stress.dead_pct || 0).toFixed(1)}%</span></div>`;
        html += `<div class="stat-row"><span class="label">Avg Jitter</span>
            <span class="value">${(stress.avg_jitter || 0).toFixed(6)}</span></div>`;
        html += `<div class="stat-row"><span class="label">Weight Drift</span>
            <span class="value">${(stress.avg_weight_drift || 0).toFixed(6)}</span></div>`;

        if (stress.warnings && stress.warnings.length > 0) {
            html += '<div style="margin-top:8px">';
            stress.warnings.forEach(w => {
                html += `<div style="color:var(--orange);font-size:11px">&#9888; ${w}</div>`;
            });
            html += '</div>';
        }

        html += '</div></div>';

        // Per-layer stats
        if (stress.layer_stats) {
            html += '<div class="card"><div class="card-header"><h3>Per-Layer Stats</h3></div><div class="card-body">';
            Object.keys(stress.layer_stats).sort().forEach(layer => {
                const s = stress.layer_stats[layer];
                const color = LAYER_COLORS[layer] || LAYER_COLORS[0];
                const hex = '#' + color.toString(16).padStart(6, '0');
                html += `<div style="margin-bottom:6px">
                    <span style="color:${hex};font-weight:600">Layer ${layer}</span>
                    <span style="color:var(--text-dim);font-size:11px;margin-left:8px">${s.count} neurons, ${s.dead_pct.toFixed(0)}% dead, avg=${s.avg_activation.toFixed(4)}</span>
                </div>`;
            });
            html += '</div></div>';
        }

        el.innerHTML = html;
    }

    buildDaemonPanel(data) {
        const el = document.getElementById('daemon-content');
        if (!data.daemons || Object.keys(data.daemons).length === 0) {
            el.innerHTML = '<div class="loading">No daemons registered</div>';
            return;
        }

        let html = `<div class="card"><div class="card-body">
            <div class="stat-row"><span class="label">Total Decisions</span><span class="value">${data.decisions_made || 0}</span></div>
            <div class="stat-row"><span class="label">Scaffold Strength</span><span class="value">${(data.scaffold_strength || 0).toFixed(3)}</span></div>
        </div></div>`;

        Object.values(data.daemons).forEach(d => {
            const phaseColors = {
                'APPRENTICE': 'var(--text-dim)',
                'JOURNEYMAN': 'var(--orange)',
                'COMPETENT': 'var(--accent)',
                'EXPERT': 'var(--green)',
                'INDEPENDENT': 'var(--purple)',
            };
            const phaseColor = phaseColors[d.phase] || 'var(--text)';

            html += `<div class="card">
                <div class="card-header" onclick="this.nextElementSibling.classList.toggle('collapsed')">
                    <h3>${d.name}</h3>
                    <span style="color:${phaseColor};font-size:11px">${d.phase}</span>
                </div>
                <div class="card-body">
                    <div class="stat-row"><span class="label">Domain</span><span class="value">${d.domain}</span></div>
                    <div class="stat-row"><span class="label">Proposals</span><span class="value">${d.proposals_made}</span></div>
                    <div class="stat-row"><span class="label">Accepted</span><span class="value">${d.proposals_accepted}</span></div>
                    <div class="stat-row"><span class="label">Accept Rate</span><span class="value">${(d.acceptance_rate * 100).toFixed(1)}%</span></div>
                    <div class="stat-row"><span class="label">Avg Reward</span><span class="value ${d.avg_reward > 0 ? 'good' : 'bad'}">${d.avg_reward.toFixed(4)}</span></div>
                </div>
            </div>`;
        });

        el.innerHTML = html;
    }

    // --- Tab switching ---

    switchTab(tabName) {
        document.querySelectorAll('.tab-bar button').forEach(b => b.classList.remove('active'));
        document.querySelectorAll('.tab-pane').forEach(p => p.classList.remove('active'));

        event.target.classList.add('active');
        document.getElementById('tab-' + tabName).classList.add('active');
    }

    // --- Trace stepper ---

    stepTrace(delta) {
        this.traceStep = Math.max(0, Math.min(this.maxStep, this.traceStep + delta));
        document.getElementById('trace-slider').value = this.traceStep;
        this.updateTraceDisplay();
    }

    seekTrace(step) {
        this.traceStep = parseInt(step);
        document.getElementById('trace-slider').value = this.traceStep;
        this.updateTraceDisplay();
    }

    async updateTraceDisplay() {
        const stepLabel = document.getElementById('trace-step');

        if (!this.auditData || !this.auditData.records || !this.auditData.records[this.traceStep]) {
            stepLabel.textContent = `Step ${this.traceStep}`;
            return;
        }

        const record = this.auditData.records[this.traceStep];
        const correct = record.correct;
        stepLabel.textContent = `Step ${record.step} ${correct ? '  OK' : '  X'}`;
        stepLabel.style.color = correct ? 'var(--green)' : 'var(--red)';

        // Fetch replay data with per-neuron activations
        try {
            const replay = await fetch(`/api/replay?step=${record.step}`).then(r => r.json());
            if (replay.error) return;

            this.isReplaying = true;
            const norm = replay.neuron_normalized || {};
            const acts = replay.neuron_activations || {};
            const hotEdgeSet = new Set();

            // Mark hot edges for quick lookup
            (replay.hot_edges || []).forEach(e => {
                hotEdgeSet.add(`${e.source}-${e.target}`);
            });

            // Light up neurons based on activation
            Object.values(this.neuronMeshes).forEach(m => {
                const nid = m.userData.neuronId;
                const intensity = norm[nid] || 0;
                const isOutput = m.userData.node.layer === this.networkData.num_layers - 1;
                const layerColor = LAYER_COLORS[m.userData.node.layer] || LAYER_COLORS[0];

                if (m.userData.node.is_dead || intensity < 0.01) {
                    // Dead or inactive: dim
                    m.material.emissive.setHex(0x222222);
                    m.material.emissiveIntensity = 0.05;
                    m.scale.setScalar(0.7);
                } else {
                    // Active: glow proportional to activation
                    m.material.emissive.setHex(layerColor);
                    m.material.emissiveIntensity = 0.3 + intensity * 0.7;
                    m.scale.setScalar(0.8 + intensity * 0.6);
                }

                // Output neuron: green/red for correct/wrong
                if (isOutput) {
                    const outputNeurons = this.networkData.nodes.filter(n =>
                        n.layer === this.networkData.num_layers - 1);
                    const outputIdx = outputNeurons.findIndex(n => n.id === nid);
                    if (outputIdx === record.chosen_class) {
                        m.material.emissive.setHex(correct ? 0x00e676 : 0xff5252);
                        m.material.emissiveIntensity = 1.0;
                        m.scale.setScalar(1.6);
                    }
                }
            });

            // Light up hot edges
            this.edgeLines.forEach(line => {
                const e = line.userData.edge;
                const key = `${e.source}-${e.target}`;

                if (hotEdgeSet.has(key)) {
                    // Hot edge: bright, thick feel
                    line.material.color.setHex(0x00ffcc);
                    line.material.opacity = 0.7;
                } else {
                    // Cold edge: very dim
                    line.material.color.setHex(0x1a2a3e);
                    line.material.opacity = 0.03;
                }
            });

            // Update neuron detail panel if one is selected
            this.refreshSelectedNeuron(replay);

        } catch (err) {
            console.error('Replay fetch failed:', err);
        }
    }

    resetTraceHighlights() {
        // Restore neurons to default appearance
        this.isReplaying = false;
        Object.values(this.neuronMeshes).forEach(m => {
            const node = m.userData.node;
            const color = node.is_dead ? DEAD_COLOR : (LAYER_COLORS[node.layer] || LAYER_COLORS[0]);
            m.material.emissive.setHex(color);
            m.material.emissiveIntensity = node.is_dead ? 0.05 : 0.2 + node.avg_activation * 0.3;
            m.scale.setScalar(1.0);
        });

        // Restore edges
        this.edgeLines.forEach(line => {
            const e = line.userData.edge;
            const str = Math.abs(e.strength);
            line.material.color.setHex(e.strength > 0 ? 0x00d4ff : 0xff5252);
            line.material.opacity = this.showEdges ? Math.min(0.6, 0.05 + str * 0.5) : 0;
        });
    }

    // --- Toggle controls ---

    toggleEdges() {
        this.showEdges = !this.showEdges;
        document.getElementById('btn-edges').classList.toggle('active');
        this.edgeLines.forEach(line => {
            if (this.showEdges) {
                const str = Math.abs(line.userData.edge.strength);
                line.material.opacity = Math.min(0.6, 0.05 + str * 0.5);
            } else {
                line.material.opacity = 0;
            }
        });
    }

    toggleDead() {
        this.showDead = !this.showDead;
        document.getElementById('btn-dead').classList.toggle('active');
        Object.values(this.neuronMeshes).forEach(m => {
            if (m.userData.node.is_dead) {
                m.material.opacity = this.showDead ? 0.6 : 0.15;
            }
        });
    }

    toggleLabels() {
        this.showLabels = !this.showLabels;
        document.getElementById('btn-labels').classList.toggle('active');
        // Labels handled in render loop via CSS labels or sprite text
    }

    // --- Live Training ---

    async toggleTraining() {
        if (this.isTraining) {
            this.stopTraining();
        } else {
            this.startTraining();
        }
    }

    async startTraining() {
        const curriculum = document.getElementById('train-curriculum').value;
        this.trainSpeed = parseInt(document.getElementById('train-speed').value);

        try {
            await fetch('/api/train/start', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ curriculum: curriculum, phases: 5 }),
            });
        } catch (err) {
            console.error('Failed to start training:', err);
            return;
        }

        this.isTraining = true;
        this.isReplaying = false;
        document.getElementById('btn-train').classList.add('active');
        document.getElementById('btn-train').textContent = 'Stop';
        document.getElementById('train-overlay').style.display = 'block';

        // Start polling for training steps
        this.trainLoop();
    }

    async stopTraining() {
        this.isTraining = false;
        if (this.trainInterval) {
            clearTimeout(this.trainInterval);
            this.trainInterval = null;
        }

        try {
            await fetch('/api/train/stop', { method: 'POST' });
        } catch (err) {}

        document.getElementById('btn-train').classList.remove('active');
        document.getElementById('btn-train').textContent = 'Train';
        document.getElementById('train-overlay').style.display = 'none';

        // Refresh side panels with new data
        const [auditRes, stressRes, daemonRes] = await Promise.all([
            fetch('/api/audit?count=200').then(r => r.json()),
            fetch('/api/stress').then(r => r.json()),
            fetch('/api/daemons').then(r => r.json()),
        ]);
        this.auditData = auditRes;
        this.buildAuditTrail(auditRes);
        this.buildHealthPanel(stressRes, this.networkData);
        this.buildDaemonPanel(daemonRes);

        // Update trace slider
        if (auditRes.records) {
            this.maxStep = auditRes.records.length - 1;
            document.getElementById('trace-slider').max = this.maxStep;
        }
    }

    async trainLoop() {
        if (!this.isTraining) return;

        try {
            const res = await fetch('/api/train/step', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ count: this.trainSpeed }),
            });
            const data = await res.json();

            if (data.error || !data.steps || data.steps.length === 0) {
                this.stopTraining();
                return;
            }

            // Update 3D network with live neuron states
            this.updateNetworkLive(data);

            // Update training overlay
            const lastStep = data.steps[data.steps.length - 1];
            const stats = data.stats || {};
            document.getElementById('train-ep').textContent = stats.episode || 0;

            const acc50 = stats.accuracy_50 || 0;
            const accEl = document.getElementById('train-acc');
            accEl.textContent = (acc50 * 100).toFixed(1) + '%';
            accEl.style.color = acc50 > 0.5 ? 'var(--green)' : acc50 > 0.25 ? 'var(--orange)' : 'var(--red)';

            document.getElementById('train-eps').textContent = lastStep.epsilon || 0;
            document.getElementById('train-level').textContent = lastStep.level || '-';

            // Show last few results
            const lastN = data.steps.slice(-5).reverse();
            document.getElementById('train-last').innerHTML = lastN.map(s =>
                `<span style="color:${s.correct ? 'var(--green)' : 'var(--red)'}">` +
                `${s.correct ? 'OK' : 'X'}</span>`
            ).join(' ');

            // Pulse the training indicator
            const pulse = document.getElementById('train-pulse');
            pulse.style.opacity = pulse.style.opacity === '0.3' ? '1' : '0.3';

            // Update header stats
            document.getElementById('stat-neurons').textContent =
                Object.keys(data.neuron_states).length;

        } catch (err) {
            console.error('Train step failed:', err);
        }

        // Schedule next batch (60ms gap gives smooth visual updates)
        this.trainInterval = setTimeout(() => this.trainLoop(), 60);
    }

    updateNetworkLive(data) {
        // Update neuron glow/size based on live activations
        const states = data.neuron_states || {};

        Object.values(this.neuronMeshes).forEach(m => {
            const nid = m.userData.neuronId;
            const state = states[nid];
            if (!state) return;

            const norm = state.normalized || 0;
            const layerColor = LAYER_COLORS[m.userData.node.layer] || LAYER_COLORS[0];

            // Update the stored node data for when training stops
            m.userData.node.avg_activation = state.activation;
            m.userData.node.is_dead = state.is_dead;

            if (state.is_dead) {
                m.material.emissive.setHex(DEAD_COLOR);
                m.material.emissiveIntensity = 0.05;
                m.scale.setScalar(0.7);
            } else {
                m.material.emissive.setHex(layerColor);
                m.material.emissiveIntensity = 0.2 + norm * 0.8;
                m.scale.setScalar(0.8 + norm * 0.5);
            }
        });

        // Update edge opacities based on current strengths
        if (data.edges) {
            const edgeMap = {};
            data.edges.forEach(e => {
                edgeMap[`${e.source}-${e.target}`] = e.strength;
            });

            this.edgeLines.forEach(line => {
                const e = line.userData.edge;
                const key = `${e.source}-${e.target}`;
                const newStrength = edgeMap[key];
                if (newStrength !== undefined) {
                    e.strength = newStrength;
                    const str = Math.abs(newStrength);
                    line.material.color.setHex(newStrength > 0 ? 0x00d4ff : 0xff5252);
                    line.material.opacity = this.showEdges ? Math.min(0.8, 0.05 + str * 0.6) : 0;
                }
            });
        }
    }

    // --- Animation loop ---

    animate() {
        requestAnimationFrame(() => this.animate());
        this.updateCamera();

        // Gentle pulse on active neurons (only when idle - not replaying or training)
        if (!this.isReplaying && !this.isTraining) {
            const time = Date.now() * 0.001;
            Object.values(this.neuronMeshes).forEach(m => {
                if (!m.userData.node.is_dead && m.userData.neuronId !== this.selectedNeuron) {
                    const base = 0.2 + m.userData.node.avg_activation * 0.3;
                    m.material.emissiveIntensity = base + Math.sin(time * 2 + m.userData.neuronId) * 0.05;
                }
            });
        } else if (this.isReplaying) {
            // During replay: pulse the chosen output neuron
            const time = Date.now() * 0.003;
            Object.values(this.neuronMeshes).forEach(m => {
                if (m.scale.x > 1.4) {
                    m.material.emissiveIntensity = 0.7 + Math.sin(time) * 0.3;
                }
            });
        }
        // During training: updateNetworkLive handles the visuals

        this.renderer.render(this.scene, this.camera);
    }
}

// Launch
const app = new HDNAViewer();
