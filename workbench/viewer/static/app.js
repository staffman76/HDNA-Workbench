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
        this.showNeuronDetail(res);
        this.selectedNeuron = neuronId;
    }

    showNeuronDetail(data) {
        const panel = document.getElementById('neuron-detail');
        const title = document.getElementById('nd-title');
        const body = document.getElementById('nd-body');

        title.textContent = `Neuron #${data.id}`;

        let html = `
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

        let html = '';
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

        let html = '';
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

    updateTraceDisplay() {
        document.getElementById('trace-step').textContent = `Step ${this.traceStep}`;

        // Highlight the corresponding audit row
        if (this.auditData && this.auditData.records && this.auditData.records[this.traceStep]) {
            const record = this.auditData.records[this.traceStep];

            // Flash the correct/incorrect on the neurons
            // Highlight output neuron for chosen class
            if (this.networkData) {
                const outputNeurons = this.networkData.nodes.filter(n =>
                    n.layer === this.networkData.num_layers - 1);
                outputNeurons.forEach((n, i) => {
                    const mesh = this.neuronMeshes[n.id];
                    if (mesh) {
                        if (i === record.chosen_class) {
                            mesh.material.emissive.setHex(record.correct ? 0x00e676 : 0xff5252);
                            mesh.material.emissiveIntensity = 0.8;
                        } else {
                            const color = LAYER_COLORS[n.layer] || LAYER_COLORS[0];
                            mesh.material.emissive.setHex(color);
                            mesh.material.emissiveIntensity = 0.2;
                        }
                    }
                });
            }
        }
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

    // --- Animation loop ---

    animate() {
        requestAnimationFrame(() => this.animate());
        this.updateCamera();

        // Gentle pulse on active neurons
        const time = Date.now() * 0.001;
        Object.values(this.neuronMeshes).forEach(m => {
            if (!m.userData.node.is_dead && m.userData.neuronId !== this.selectedNeuron) {
                const base = 0.2 + m.userData.node.avg_activation * 0.3;
                m.material.emissiveIntensity = base + Math.sin(time * 2 + m.userData.neuronId) * 0.05;
            }
        });

        this.renderer.render(this.scene, this.camera);
    }
}

// Launch
const app = new HDNAViewer();
