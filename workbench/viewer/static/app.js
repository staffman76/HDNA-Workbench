// HDNA Workbench Viewer - 3D Model Visualization
// Copyright (c) 2026 Chris. All rights reserved.

const LAYER_COLORS = [
    0x00d4ff, // layer 1 - cyan
    0xff6b35, // layer 2 - burnt orange
    0xb388ff, // layer 3 - purple
    0xff5252, // layer 4 / output - red
    0xffd740, // extra layers...
    0x69f0ae,
    0xff80ab,
    0x40c4ff,
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
        this.layerMeshes = [];     // for non-HDNA models
        this.labels = [];
        this.networkData = null;
        this.auditData = null;
        this.selectedNeuron = null;
        this.activeModel = null;   // name of currently displayed model
        this.governanceMode = false;
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
        this.chartData = {
            accuracy: [],       // raw rolling accuracy values
            smoothed: [],       // smoothed (moving average of rolling avg)
            epsilon: [],        // epsilon values over time
            maxPoints: 200,     // max data points to show
            bestAccuracy: 0,
            allCorrect: 0,      // total correct ever
            allTotal: 0,        // total episodes ever
        };

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
            this.panX -= dx * 0.01;
            this.panY += dy * 0.01;
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
            this.activeModel = 'HDNA (primary)';
            this.updateHeader(
                modelRes.name || 'HDNA Model',
                networkRes.nodes ? networkRes.nodes.length : 0,
                networkRes.edges ? networkRes.edges.length : 0,
                networkRes.num_layers || 0,
                'hdna'
            );

            this.networkData = networkRes;
            this.auditData = auditRes;

            this.buildNetwork(networkRes);
            this.buildNeuronList(networkRes);
            this.buildAuditTrail(auditRes);
            this.buildHealthPanel(stressRes, networkRes);
            this.buildDaemonPanel(daemonRes);
            this.refreshModelsList();
            this.refreshCurriculaDropdown();
            this.refreshDaemonList();
            this.updateDaemonForm();

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

        // Scale spacing based on largest layer
        const maxLayerSize = Math.max(...Object.values(layerGroups).map(g => g.length));
        const layerSpacing = 4.0;
        const neuronSpacing = Math.min(0.5, 8.0 / Math.max(1, maxLayerSize));

        data.nodes.forEach(node => {
            const layer = node.layer;
            const layerNodes = layerGroups[layer] || [];
            const idx = layerNodes.indexOf(node);
            const yOffset = (layerNodes.length - 1) / 2;

            const x = (layer - (numLayers - 1) / 2) * layerSpacing;
            const y = (idx - yOffset) * neuronSpacing;
            const z = (Math.sin(idx * 0.7) * 0.3); // slight depth variation

            const isOutput = (layer === numLayers - 1);

            // Size: scale with spacing so neurons don't overlap
            const sizeScale = Math.min(1.0, neuronSpacing / 0.5);
            const baseSize = (isOutput ? 0.15 : 0.08) * sizeScale;
            const actSize = Math.min(0.2 * sizeScale, baseSize + node.avg_activation * 0.15);
            const size = node.is_dead ? baseSize * 0.5 : actSize;

            // Color: use layer color, output layer always red/warm
            let color;
            if (node.is_dead) {
                color = DEAD_COLOR;
            } else if (isOutput) {
                color = 0xff5252; // output always red — distinct from hidden layers
            } else {
                color = LAYER_COLORS[layer % LAYER_COLORS.length] || LAYER_COLORS[0];
            }

            // Output neurons use octahedron (diamond shape), hidden use spheres
            const geometry = isOutput ?
                new THREE.OctahedronGeometry(size * 1.2) :
                new THREE.SphereGeometry(size, 16, 12);
            const material = new THREE.MeshPhongMaterial({
                color: color,
                emissive: color,
                emissiveIntensity: node.is_dead ? 0.05 : 0.2 + node.avg_activation * 0.3,
                transparent: node.is_dead && !this.showDead,
                opacity: (node.is_dead && !this.showDead) ? 0.15 : 1.0,
            });

            const mesh = new THREE.Mesh(geometry, material);
            mesh.position.set(x, y, z);
            mesh.userData = { neuronId: node.id, node: node, isOutput: isOutput };
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

        // Only show the strongest edges
        const sortedEdges = [...edges].sort((a, b) => Math.abs(b.strength) - Math.abs(a.strength));
        const maxEdges = Math.min(sortedEdges.length, 80);
        const topEdges = sortedEdges.slice(0, maxEdges);
        const maxStrength = topEdges.length > 0 ? Math.abs(topEdges[0].strength) : 1;

        topEdges.forEach(edge => {
            const srcMesh = this.neuronMeshes[edge.source];
            const tgtMesh = this.neuronMeshes[edge.target];
            if (!srcMesh || !tgtMesh) return;

            const src = srcMesh.position.clone();
            const tgt = tgtMesh.position.clone();
            const strength = Math.abs(edge.strength);
            const normalizedStrength = strength / maxStrength;
            const color = edge.strength > 0 ? 0x8899aa : 0xcc4444;

            const points = [src, tgt];
            const geometry = new THREE.BufferGeometry().setFromPoints(points);
            const material = new THREE.LineBasicMaterial({
                color: color,
                transparent: true,
                opacity: this.showEdges ? Math.min(0.4, 0.03 + normalizedStrength * 0.3) : 0,
            });

            const line = new THREE.Line(geometry, material);
            this.scene.add(line);

            this.edgeLines.push({
                line: line,
                edge: edge,
                strength: normalizedStrength,
                color: color,
                baseOpacity: Math.min(0.4, 0.03 + normalizedStrength * 0.3),
            });
        });
    }

    // Helper: set edge visibility/color
    setEdgeStyle(edgeGroup, color, opacity) {
        if (!edgeGroup.line) return;
        if (color !== undefined) edgeGroup.line.material.color.setHex(color);
        edgeGroup.line.material.opacity = opacity;
    }

    setAllEdgesVisible(visible) {
        this.edgeLines.forEach(eg => {
            if (!eg.line) return;
            eg.line.material.opacity = visible ? Math.min(0.5, 0.05 + eg.strength * 0.3) : 0;
        });
    }

    animateEdges(time) {
        // No animation needed — edges update via setEdgeStyle calls
        // from training loop, replay, and selection handlers
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

        // Highlight edges connected to this neuron
        this.edgeLines.forEach(eg => {
            if (!eg.edge) return;
            const e = eg.edge;
            if (e.source === neuronId || e.target === neuronId) {
                this.setEdgeStyle(eg, EDGE_HIGHLIGHT, 0.8);
            } else {
                this.setEdgeStyle(eg, eg.color, this.showEdges ? Math.min(0.5, 0.05 + eg.strength * 0.3) : 0);
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
        this.edgeLines.forEach(eg => {
            if (eg.edge) {
                this.setEdgeStyle(eg, eg.color, this.showEdges ? Math.min(0.5, 0.05 + eg.strength * 0.3) : 0);
            }
        });
        this.selectedNeuron = null;
    }

    // --- Model switching ---

    clearScene() {
        // Remove all neuron meshes
        Object.values(this.neuronMeshes).forEach(m => this.scene.remove(m));
        this.neuronMeshes = {};

        // Remove all edges
        this.edgeLines.forEach(eg => {
            if (eg.line) this.scene.remove(eg.line);
            if (eg.dots) eg.dots.forEach(d => this.scene.remove(d));
        });
        this.edgeLines = [];

        // Remove layer meshes (for PyTorch models)
        this.layerMeshes.forEach(m => this.scene.remove(m));
        this.layerMeshes = [];

        this.closeNeuronDetail();
    }

    async switchToModel(name) {
        this.clearScene();
        this.activeModel = name;

        if (name === 'HDNA (primary)' || name === null) {
            // Switch back to HDNA — reload the network graph
            const networkRes = await fetch('/api/network').then(r => r.json());
            const modelRes = await fetch('/api/model').then(r => r.json());
            this.networkData = networkRes;
            this.buildNetwork(networkRes);
            this.buildNeuronList(networkRes);
            this.updateHeader(modelRes.name, networkRes.nodes.length,
                networkRes.edges.length, networkRes.num_layers, 'hdna');
        } else {
            // External model — fetch inspection and build layer graph
            try {
                const res = await fetch('/api/models/inspect', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ name: name }),
                }).then(r => r.json());

                if (res.error) {
                    console.error(res.error);
                    return;
                }

                const info = res.info || {};
                const layers = res.layers || [];
                this.buildLayerGraph(name, layers, info);
                this.buildExternalNeuronList(name, layers);
                this.updateHeader(name, info.parameter_count || 0,
                    layers.length, layers.length, info.framework || '?');
            } catch (err) {
                console.error('Failed to switch model:', err);
            }
        }

        this.resetCamera();
    }

    updateHeader(name, neurons, connections, layers, framework) {
        document.getElementById('model-name').textContent = name;
        document.getElementById('stat-neurons').textContent =
            typeof neurons === 'number' ? neurons.toLocaleString() : neurons;
        document.getElementById('stat-connections').textContent = connections;
        document.getElementById('stat-layers').textContent = layers;

        const healthEl = document.getElementById('stat-health');
        if (framework === 'hdna') {
            healthEl.textContent = 'HDNA';
            healthEl.style.color = 'var(--accent)';
        } else {
            healthEl.textContent = framework;
            healthEl.style.color = 'var(--purple)';
        }
    }

    buildLayerGraph(modelName, layers, info) {
        // Build a 3D graph where each layer is a box/sphere
        // Sized by parameter count, connected in sequence
        if (!layers || layers.length === 0) return;

        const maxParams = Math.max(1, ...layers.map(l => l.parameter_count || 1));
        const spacing = 2.0;
        const totalWidth = (layers.length - 1) * spacing;

        layers.forEach((layer, i) => {
            const x = (i - (layers.length - 1) / 2) * spacing;
            const params = layer.parameter_count || 0;
            const size = 0.15 + (params / maxParams) * 0.5;
            const colorIdx = i % LAYER_COLORS.length;
            const color = LAYER_COLORS[colorIdx];

            // Use boxes for non-HDNA layers (visually distinct from HDNA spheres)
            const geometry = new THREE.BoxGeometry(size, size * 1.5, size);
            const material = new THREE.MeshPhongMaterial({
                color: color,
                emissive: color,
                emissiveIntensity: 0.3,
                transparent: params === 0,
                opacity: params === 0 ? 0.3 : 1.0,
            });

            const mesh = new THREE.Mesh(geometry, material);
            mesh.position.set(x, 0, 0);
            mesh.userData = {
                layerName: layer.name,
                layerType: layer.type,
                paramCount: params,
                index: i,
                inspectable: layer.inspectable || false,
            };
            this.scene.add(mesh);
            this.layerMeshes.push(mesh);

            // Connection to previous layer
            if (i > 0) {
                const prev = this.layerMeshes[i - 1];
                const points = [prev.position.clone(), mesh.position.clone()];
                const lineGeo = new THREE.BufferGeometry().setFromPoints(points);
                const lineMat = new THREE.LineBasicMaterial({
                    color: 0x00d4ff,
                    transparent: true,
                    opacity: 0.4,
                });
                const line = new THREE.Line(lineGeo, lineMat);
                this.scene.add(line);
                this.edgeLines.push(line);
            }

            // Text label (using a sprite — simple approach)
            const canvas = document.createElement('canvas');
            canvas.width = 256;
            canvas.height = 64;
            const ctx = canvas.getContext('2d');
            ctx.fillStyle = '#e0e0e0';
            ctx.font = '14px sans-serif';
            ctx.textAlign = 'center';

            // Shorten long names
            let label = layer.type || layer.name;
            if (label.length > 20) label = label.substring(0, 18) + '..';
            ctx.fillText(label, 128, 20);

            ctx.fillStyle = '#8892a4';
            ctx.font = '11px sans-serif';
            ctx.fillText(params > 0 ? params.toLocaleString() + ' params' : 'no params', 128, 40);

            if (layer.inspectable) {
                ctx.fillStyle = '#00e676';
                ctx.font = '10px sans-serif';
                ctx.fillText('[INSPECTABLE]', 128, 55);
            }

            const texture = new THREE.CanvasTexture(canvas);
            const spriteMat = new THREE.SpriteMaterial({ map: texture, transparent: true });
            const sprite = new THREE.Sprite(spriteMat);
            sprite.position.set(x, -size - 0.4, 0);
            sprite.scale.set(1.5, 0.4, 1);
            this.scene.add(sprite);
            this.layerMeshes.push(sprite);
        });
    }

    buildExternalNeuronList(modelName, layers) {
        // Repurpose the neuron list for external model layers
        const container = document.getElementById('neuron-list');
        if (!layers || layers.length === 0) {
            container.innerHTML = '<div class="loading">No layers</div>';
            return;
        }

        let html = `<div class="neuron-row" style="border-bottom:1px solid var(--border);padding-bottom:6px;margin-bottom:6px;font-weight:600;color:var(--text-dim);cursor:default">
            <span class="nid">Layer</span>
            <span class="layer-tag">Type</span>
            <span class="activation">Params</span>
            <span class="status">Insp.</span>
        </div>`;

        layers.forEach((layer, i) => {
            const color = LAYER_COLORS[i % LAYER_COLORS.length];
            const hex = '#' + color.toString(16).padStart(6, '0');
            const inspectable = layer.inspectable ? '<span style="color:var(--green)">YES</span>' : '<span style="color:var(--text-dim)">---</span>';

            html += `<div class="neuron-row">
                <span class="nid" style="color:${hex}">${layer.name}</span>
                <span class="layer-tag">${layer.type}</span>
                <span class="activation">${(layer.parameter_count || 0).toLocaleString()}</span>
                <span class="status">${inspectable}</span>
            </div>`;
        });

        container.innerHTML = html;
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
                <span class="confidence">${(Math.min(1, Math.max(0, r.confidence)) * 100).toFixed(0)}%</span>
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

    toggleReplayMode() {
        const btn = document.getElementById('btn-replay-toggle');
        if (this.isReplaying) {
            // Exit replay mode — back to live view
            this.resetTraceHighlights();
            btn.textContent = 'Replay';
            btn.classList.remove('active');
            btn.style.color = 'var(--text)';
            btn.style.borderColor = 'var(--border)';
        } else {
            // Enter replay mode
            this.isReplaying = true;
            btn.textContent = 'Live View';
            btn.classList.add('active');
            btn.style.color = 'var(--accent)';
            btn.style.borderColor = 'var(--accent)';
            this.updateTraceDisplay();
        }
    }

    stepTrace(delta) {
        if (!this.isReplaying) {
            this.isReplaying = true;
            const btn = document.getElementById('btn-replay-toggle');
            btn.textContent = 'Live View';
            btn.classList.add('active');
            btn.style.color = 'var(--accent)';
            btn.style.borderColor = 'var(--accent)';
        }
        this.traceStep = Math.max(0, Math.min(this.maxStep, this.traceStep + delta));
        document.getElementById('trace-slider').value = this.traceStep;
        this.updateTraceDisplay();
    }

    seekTrace(step) {
        if (!this.isReplaying) {
            this.isReplaying = true;
            const btn = document.getElementById('btn-replay-toggle');
            btn.textContent = 'Live View';
            btn.classList.add('active');
            btn.style.color = 'var(--accent)';
            btn.style.borderColor = 'var(--accent)';
        }
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

            // Light up the active path — bright for hot edges, dim everything else
            this.edgeLines.forEach(eg => {
                if (!eg.edge) return;
                const e = eg.edge;
                const key = `${e.source}-${e.target}`;

                if (hotEdgeSet.has(key)) {
                    // Active path: bright cyan, clearly visible
                    this.setEdgeStyle(eg, 0x00ffcc, 0.7);
                } else {
                    // Inactive: very dim so the active path stands out
                    this.setEdgeStyle(eg, 0x222233, 0.04);
                }
            });

            // Update neuron detail panel if one is selected
            this.refreshSelectedNeuron(replay);

        } catch (err) {
            console.error('Replay fetch failed:', err);
        }
    }

    resetTraceHighlights() {
        // Restore neurons to default appearance — back to live view
        this.isReplaying = false;
        const btn = document.getElementById('btn-replay-toggle');
        if (btn) {
            btn.textContent = 'Replay';
            btn.classList.remove('active');
            btn.style.color = 'var(--text)';
            btn.style.borderColor = 'var(--border)';
        }
        Object.values(this.neuronMeshes).forEach(m => {
            const node = m.userData.node;
            const color = node.is_dead ? DEAD_COLOR : (LAYER_COLORS[node.layer] || LAYER_COLORS[0]);
            m.material.emissive.setHex(color);
            m.material.emissiveIntensity = node.is_dead ? 0.05 : 0.2 + node.avg_activation * 0.3;
            m.scale.setScalar(1.0);
        });

        // Restore edges
        this.edgeLines.forEach(eg => {
            if (eg.edge) {
                eg._isHot = false;
                this.setEdgeStyle(eg, eg.color, this.showEdges ? Math.min(0.5, 0.05 + eg.strength * 0.3) : 0);
            }
        });
    }

    // --- Toggle controls ---

    toggleEdges() {
        this.showEdges = !this.showEdges;
        document.getElementById('btn-edges').classList.toggle('active');
        this.setAllEdgesVisible(this.showEdges);
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

    // --- Save / Load ---

    async saveModel() {
        try {
            const res = await fetch('/api/save').then(r => r.json());
            if (res.error) {
                alert('Save failed: ' + res.error);
            } else {
                alert('Model saved! (' + res.neurons + ' neurons)');
            }
        } catch (err) {
            alert('Save failed: ' + err);
        }
    }

    async loadModel() {
        try {
            const res = await fetch('/api/load').then(r => r.json());
            if (res.error) {
                alert('Load failed: ' + res.error);
            } else {
                alert('Model loaded! (' + res.neurons + ' neurons). Refreshing...');
                // Reload the entire view with new network
                location.reload();
            }
        } catch (err) {
            alert('Load failed: ' + err);
        }
    }

    // --- Live neuron refresh during training ---

    async refreshSelectedNeuronLive(neuronStates) {
        // Quick update of the neuron detail panel during training
        // without a full API call — use the states we already have
        if (this.selectedNeuron === null) return;
        const panel = document.getElementById('neuron-detail');
        if (!panel.classList.contains('visible')) return;

        const state = neuronStates[this.selectedNeuron];
        if (!state) return;

        // Fetch full neuron data (routing etc) from server
        try {
            const res = await fetch(`/api/neuron?id=${this.selectedNeuron}`).then(r => r.json());
            // Build a fake replay context with current training state
            const fakeReplay = {
                step: 'LIVE',
                record: {},
                neuron_activations: {},
                neuron_normalized: {},
                hot_edges: [],
            };
            // Fill from all neuron states
            let maxAct = 0.001;
            Object.entries(neuronStates).forEach(([nid, s]) => {
                fakeReplay.neuron_activations[parseInt(nid)] = s.activation;
                if (s.activation > maxAct) maxAct = s.activation;
            });
            Object.entries(neuronStates).forEach(([nid, s]) => {
                fakeReplay.neuron_normalized[parseInt(nid)] = s.activation / maxAct;
            });

            this.showNeuronDetail(res, fakeReplay);
        } catch (err) {}
    }

    // --- Inspectable Transformer ---

    async launchTransformer() {
        const btn = document.getElementById('btn-transformer');

        if (this._transformerRunning) {
            this.stopTransformerTraining();
            return;
        }

        btn.textContent = 'Creating...';
        btn.disabled = true;

        try {
            // Create the transformer
            const res = await fetch('/api/transformer/create', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    vocab_size: 1000, d_model: 64,
                    n_heads: 4, n_layers: 2, n_experts: 4, d_ff: 128,
                }),
            }).then(r => r.json());

            if (res.error) {
                alert('Failed: ' + res.error);
                btn.textContent = 'Transformer';
                btn.disabled = false;
                return;
            }

            console.log('Transformer created:', res);

            // Load and display the transformer graph
            await this.showTransformerView();

            // Start training loop
            this._transformerRunning = true;
            this._transformerLosses = [];
            btn.textContent = 'Stop Transformer';
            btn.classList.add('active');
            btn.disabled = false;
            this.transformerTrainLoop();

        } catch (err) {
            alert('Failed: ' + err);
            btn.textContent = 'Transformer';
            btn.disabled = false;
        }
    }

    async showTransformerView() {
        // Fetch transformer graph data
        const data = await fetch('/api/transformer').then(r => r.json());
        if (!data.exists) return;

        // Clear current 3D scene
        this.clearScene();

        const graph = data.graph;
        const snap = data.snapshot;

        // Update header
        this.updateHeader(
            `Inspectable Transformer (${snap.parameters.toLocaleString()} params)`,
            `${snap.n_heads * snap.n_layers} heads`,
            `${snap.n_experts * snap.n_layers} experts`,
            snap.n_layers + ' layers',
            'transformer'
        );

        // Build 3D visualization
        // Heads as spheres, experts as boxes, layered layout
        const layerSpacing = 3.0;
        const headSpacing = 1.2;
        const expertSpacing = 1.2;

        graph.nodes.forEach(node => {
            const layerX = (node.layer - (snap.n_layers - 1) / 2) * layerSpacing;

            if (node.type === 'head') {
                const yOffset = (snap.n_heads - 1) / 2;
                const y = (node.index - yOffset) * headSpacing + 1;

                const size = 0.15 + (node.gate || 0.5) * 0.2;
                const color = node.is_dead ? 0x444444 : LAYER_COLORS[node.layer] || 0x00d4ff;

                const geometry = new THREE.SphereGeometry(size, 16, 12);
                const material = new THREE.MeshPhongMaterial({
                    color: color,
                    emissive: color,
                    emissiveIntensity: node.is_dead ? 0.05 : 0.3 + (node.sharpness || 0) * 0.5,
                });
                const mesh = new THREE.Mesh(geometry, material);
                mesh.position.set(layerX, y, 0);
                mesh.userData = { nodeData: node, type: 'head' };
                this.scene.add(mesh);
                this.layerMeshes.push(mesh);

                // Label
                this.addLabel(layerX, y - size - 0.3, 0, node.tag || `H${node.index}`, color);

            } else if (node.type === 'expert') {
                const yOffset = (snap.n_experts - 1) / 2;
                const y = (node.index - yOffset) * expertSpacing - 1.5;

                const usage = (node.usage_pct || 25) / 100;
                const size = 0.1 + usage * 0.25;
                const color = 0xffab40;

                const geometry = new THREE.BoxGeometry(size, size * 1.2, size);
                const material = new THREE.MeshPhongMaterial({
                    color: color,
                    emissive: color,
                    emissiveIntensity: 0.2 + usage * 0.4,
                });
                const mesh = new THREE.Mesh(geometry, material);
                mesh.position.set(layerX, y, 0);
                mesh.userData = { nodeData: node, type: 'expert' };
                this.scene.add(mesh);
                this.layerMeshes.push(mesh);

                // Label
                this.addLabel(layerX, y - size - 0.3, 0, `E${node.index} ${node.usage_pct?.toFixed(0) || '?'}%`, color);
            }
        });

        // Edges
        graph.edges.forEach(edge => {
            const srcNode = graph.nodes.find(n => n.id === edge.source);
            const tgtNode = graph.nodes.find(n => n.id === edge.target);
            if (!srcNode || !tgtNode) return;

            const srcMesh = this.layerMeshes.find(m =>
                m.userData.nodeData && m.userData.nodeData.id === edge.source);
            const tgtMesh = this.layerMeshes.find(m =>
                m.userData.nodeData && m.userData.nodeData.id === edge.target);
            if (!srcMesh || !tgtMesh) return;

            const points = [srcMesh.position.clone(), tgtMesh.position.clone()];
            const geometry = new THREE.BufferGeometry().setFromPoints(points);
            const material = new THREE.LineBasicMaterial({
                color: 0x00d4ff,
                transparent: true,
                opacity: Math.min(0.6, edge.strength * 0.8),
            });
            const line = new THREE.Line(geometry, material);
            this.scene.add(line);
            this.edgeLines.push(line);
        });

        // Layer separators (subtle vertical lines)
        for (let i = 0; i < snap.n_layers; i++) {
            const x = (i - (snap.n_layers - 1) / 2) * layerSpacing;
            this.addLabel(x, 2.5, 0, `Layer ${i}`, 0x556677);
        }
        this.addLabel(-(snap.n_layers - 1) / 2 * layerSpacing, 2.0, 0, 'Heads', 0x00d4ff);
        this.addLabel(-(snap.n_layers - 1) / 2 * layerSpacing, -0.8, 0, 'Experts', 0xffab40);

        // Update neurons tab with head/expert info
        this.buildTransformerPanel(data);
        this.resetCamera();
    }

    addLabel(x, y, z, text, color) {
        const canvas = document.createElement('canvas');
        canvas.width = 256;
        canvas.height = 48;
        const ctx = canvas.getContext('2d');
        ctx.fillStyle = '#' + (color || 0xffffff).toString(16).padStart(6, '0');
        ctx.font = '14px sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText(text, 128, 28);

        const texture = new THREE.CanvasTexture(canvas);
        const material = new THREE.SpriteMaterial({ map: texture, transparent: true });
        const sprite = new THREE.Sprite(material);
        sprite.position.set(x, y, z);
        sprite.scale.set(1.2, 0.3, 1);
        this.scene.add(sprite);
        this.layerMeshes.push(sprite);
    }

    buildTransformerPanel(data) {
        const container = document.getElementById('neuron-list');
        const headReport = data.head_report || [];
        const expertReport = data.expert_report || [];

        let html = '<div style="font-size:12px;color:var(--accent);font-weight:600;margin-bottom:8px">Attention Heads</div>';

        headReport.forEach(h => {
            const tagColor = h.is_dead ? 'var(--red)' :
                h.tag === 'sharp_selector' ? 'var(--green)' :
                h.tag === 'local_focus' ? 'var(--accent)' :
                h.tag === 'position_tracker' ? 'var(--purple)' :
                h.tag === 'global_mixer' ? 'var(--orange)' :
                'var(--text-dim)';

            html += `<div class="neuron-row">
                <span class="nid" style="color:var(--accent)">L${h.layer}H${h.head}</span>
                <span style="color:${tagColor};font-size:10px;min-width:90px">${h.tag}</span>
                <span class="activation">${h.avg_entropy.toFixed(2)} ent</span>
                <span class="status ${h.is_dead ? 'dead' : 'active'}">${h.is_dead ? 'DEAD' : 'OK'}</span>
            </div>`;
        });

        html += '<div style="font-size:12px;color:var(--orange);font-weight:600;margin:12px 0 8px">Expert Routing</div>';
        expertReport.forEach(e => {
            html += `<div style="margin-bottom:6px">
                <span style="color:var(--text-dim);font-size:11px">Layer ${e.layer}:</span>`;
            e.usage_pct.forEach((pct, i) => {
                const barW = Math.min(60, pct * 0.6);
                html += ` <span style="font-size:10px;color:var(--orange)">E${i}:${pct.toFixed(0)}%</span>
                    <span class="act-bar" style="width:${barW}px;background:var(--orange)"></span>`;
            });
            html += '</div>';
        });

        container.innerHTML = html;
    }

    async transformerTrainLoop() {
        if (!this._transformerRunning) return;

        try {
            const res = await fetch('/api/transformer/train_step', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ steps: 5 }),
            }).then(r => r.json());

            if (res.error) {
                console.error('Transformer train error:', res.error);
                this.stopTransformerTraining();
                return;
            }

            // Track losses
            this._transformerLosses.push(res.avg_loss);
            if (this._transformerLosses.length > 200) this._transformerLosses.shift();

            // Update the 3D view with new head tags/gate values
            await this.showTransformerView();

            // Update training stats display
            const acc = res.accuracy || 0;
            document.getElementById('train-ep').textContent = res.total_forwards;
            document.getElementById('train-acc').textContent = `${(acc * 100).toFixed(1)}%`;
            document.getElementById('train-acc').style.color =
                acc > 0.7 ? 'var(--green)' : acc > 0.3 ? 'var(--orange)' : 'var(--red)';
            document.getElementById('train-overlay').style.display = 'block';
            document.getElementById('train-lifetime').textContent =
                `loss: ${res.avg_loss.toFixed(3)}`;
            document.getElementById('train-eps').textContent =
                res.avg_loss.toFixed(2);

            // Track best
            if (acc > this.chartData.bestAccuracy) {
                this.chartData.bestAccuracy = acc;
            }
            document.getElementById('train-best').textContent =
                (this.chartData.bestAccuracy * 100).toFixed(1) + '%';

            // Show head specialization
            const specialized = (res.head_report || []).filter(h => h.tag !== 'balanced' && !h.tag.startsWith('head_'));
            document.getElementById('train-level').textContent =
                `${specialized.length}/${(res.head_report || []).length} heads specialized`;

            // Show sample prediction with actual English
            if (res.sample) {
                const s = res.sample;
                document.getElementById('train-last').innerHTML =
                    `<div style="font-size:10px;margin-top:2px">` +
                    `<div style="color:var(--text-dim)">Input: <span style="color:var(--accent)">${s.input}</span></div>` +
                    `<div style="color:var(--text-dim)">Expected: <span style="color:var(--text)">${s.target}</span></div>` +
                    `<div style="color:var(--text-dim)">Predicted: <span style="color:${s.correct > s.total/2 ? 'var(--green)' : 'var(--orange)'}">${s.predicted}</span></div>` +
                    `<div style="color:var(--text-dim)">${s.correct}/${s.total} tokens correct</div>` +
                    `</div>`;
            }

            document.getElementById('chart-episodes').textContent = res.total_forwards;

            // Push to chart (use accuracy for green line, loss/7 for orange line as proxy)
            this.chartData.accuracy.push(acc);
            this.chartData.epsilon.push(Math.min(1, res.avg_loss / 7));  // normalize loss to 0-1

            const raw = this.chartData.accuracy;
            const smoothWin = Math.min(10, raw.length);
            const smoothed = raw.slice(-smoothWin).reduce((a, b) => a + b, 0) / smoothWin;
            this.chartData.smoothed.push(smoothed);

            // Track lifetime
            this.chartData.allTotal += 5;
            this.chartData.allCorrect += Math.round(acc * 5 * 16);  // approx

            if (this.chartData.accuracy.length > this.chartData.maxPoints) {
                this.chartData.accuracy.shift();
                this.chartData.epsilon.shift();
                this.chartData.smoothed.shift();
            }
            this.drawChart();

            // Update bottom graphs with transformer data
            this.updateTransformerGraphs(res);

        } catch (err) {
            console.error('Transformer train failed:', err);
        }

        this._transformerInterval = setTimeout(() => this.transformerTrainLoop(), 200);
    }

    stopTransformerTraining() {
        this._transformerRunning = false;
        if (this._transformerInterval) {
            clearTimeout(this._transformerInterval);
            this._transformerInterval = null;
        }
        const btn = document.getElementById('btn-transformer');
        btn.textContent = 'Transformer';
        btn.classList.remove('active');
        document.getElementById('train-overlay').style.display = 'none';

        // Switch back to HDNA view
        this.switchToModel('HDNA (primary)');
    }

    // --- Graph Panel ---

    switchGraphTab(tabName) {
        document.querySelectorAll('#graphs-panel .tab-bar button').forEach(b => b.classList.remove('active'));
        document.querySelectorAll('.graph-pane').forEach(p => { p.style.display = 'none'; p.classList.remove('active'); });
        event.target.classList.add('active');
        const pane = document.getElementById('graph-' + tabName);
        if (pane) { pane.style.display = 'block'; pane.classList.add('active'); }
    }

    updateGraphs(data) {
        // Called during training with each batch of steps
        const stats = data.stats || {};
        const steps = data.steps || [];
        const neuronStates = data.neuron_states || {};

        // Collect graph data
        if (!this.graphHistory) {
            this.graphHistory = {
                accuracy: [], dead_pct: [], drift: [], confidence: [],
                active_pct: [], scaffold: [], maxPoints: 150,
            };
        }
        const gh = this.graphHistory;

        // Accuracy
        gh.accuracy.push(stats.accuracy_50 || 0);

        // Dead neuron %
        const totalNeurons = Object.keys(neuronStates).length;
        const deadNeurons = Object.values(neuronStates).filter(s => s.is_dead).length;
        gh.dead_pct.push(totalNeurons > 0 ? deadNeurons / totalNeurons : 0);

        // Active neuron %
        const activeNeurons = Object.values(neuronStates).filter(s => s.activation > 0.01).length;
        gh.active_pct.push(totalNeurons > 0 ? activeNeurons / totalNeurons : 0);

        // Average confidence from recent steps
        const avgConf = steps.length > 0 ?
            steps.reduce((s, st) => s + (st.correct ? 1 : 0), 0) / steps.length : 0;
        gh.confidence.push(avgConf);

        // Scaffold (from last step)
        const lastStep = steps[steps.length - 1] || {};
        gh.scaffold.push(lastStep.epsilon || 0);

        // Trim
        Object.keys(gh).forEach(k => {
            if (Array.isArray(gh[k]) && gh[k].length > gh.maxPoints) gh[k].shift();
        });

        // Draw all visible graphs
        this.drawMiniGraph('graph-accuracy', gh.accuracy, 'var(--green)', 0, 1);
        this.drawMiniGraph('graph-activity', gh.active_pct, 'var(--accent)', 0, 1);
        this.drawMiniGraph('graph-confidence', gh.confidence, 'var(--purple)', 0, 1);
        this.drawMiniGraph('graph-dead', gh.dead_pct, 'var(--red)', 0, 0.5);
        this.drawMiniGraph('graph-scaffold', gh.scaffold, 'var(--orange)', 0, 1);
        this.drawMiniGraph('graph-spread', gh.active_pct, 'var(--accent)', 0, 1);

        // Contextual insights
        this.updateInsights(gh, stats);

        // Curriculum progress bars
        this.updateCurriculumBars(stats);
    }

    updateTransformerGraphs(res) {
        // Bottom graphs for transformer training
        if (!this.graphHistory) {
            this.graphHistory = {
                accuracy: [], dead_pct: [], drift: [], confidence: [],
                active_pct: [], scaffold: [], maxPoints: 150,
            };
        }
        const gh = this.graphHistory;
        const headReport = res.head_report || [];

        // Accuracy
        gh.accuracy.push(res.accuracy || 0);

        // Dead heads as "dead_pct"
        const totalHeads = headReport.length;
        const deadHeads = headReport.filter(h => h.is_dead).length;
        gh.dead_pct.push(totalHeads > 0 ? deadHeads / totalHeads : 0);

        // Specialized heads as "active_pct" (heads that have earned a real tag)
        const specialized = headReport.filter(h => h.tag !== 'balanced' && !h.tag.startsWith('head_')).length;
        gh.active_pct.push(totalHeads > 0 ? specialized / totalHeads : 0);

        // Loss as "confidence" (inverted and normalized so lower loss = higher)
        const lossNorm = Math.max(0, 1 - (res.avg_loss || 7) / 7);
        gh.confidence.push(lossNorm);

        // Entropy variance as "scaffold" (head differentiation)
        const entropies = headReport.map(h => h.avg_entropy);
        const mean = entropies.length > 0 ? entropies.reduce((a,b) => a+b, 0) / entropies.length : 0;
        const variance = entropies.length > 1 ? entropies.reduce((s,e) => s + (e-mean)**2, 0) / entropies.length : 0;
        gh.scaffold.push(Math.min(1, variance * 5));  // scale for visibility

        // Trim
        Object.keys(gh).forEach(k => {
            if (Array.isArray(gh[k]) && gh[k].length > gh.maxPoints) gh[k].shift();
        });

        // Draw graphs
        this.drawMiniGraph('graph-accuracy', gh.accuracy, 'var(--green)', 0, 1);
        this.drawMiniGraph('graph-activity', gh.active_pct, 'var(--accent)', 0, 1);
        this.drawMiniGraph('graph-confidence', gh.confidence, 'var(--purple)', 0, 1);
        this.drawMiniGraph('graph-dead', gh.dead_pct, 'var(--red)', 0, 0.5);
        this.drawMiniGraph('graph-scaffold', gh.scaffold, 'var(--orange)', 0, 1);
        this.drawMiniGraph('graph-spread', gh.active_pct, 'var(--accent)', 0, 1);

        // Update insights for transformer context
        const accInsight = document.getElementById('graph-acc-insight');
        if (accInsight) {
            const a = res.accuracy || 0;
            if (a > 0.7) { accInsight.textContent = 'Patterns learned'; accInsight.className = 'graph-insight good'; }
            else if (a > 0.3) { accInsight.textContent = 'Learning'; accInsight.className = 'graph-insight warn'; }
            else { accInsight.textContent = 'Early training'; accInsight.className = 'graph-insight neutral'; }
        }
        const actInsight = document.getElementById('graph-activity-insight');
        if (actInsight) {
            if (specialized > totalHeads * 0.5) { actInsight.textContent = 'Heads specializing'; actInsight.className = 'graph-insight good'; }
            else if (specialized > 0) { actInsight.textContent = 'Some specialization'; actInsight.className = 'graph-insight warn'; }
            else { actInsight.textContent = 'Heads uniform'; actInsight.className = 'graph-insight neutral'; }
        }
        const deadInsight = document.getElementById('graph-dead-insight');
        if (deadInsight) {
            if (deadHeads === 0) { deadInsight.textContent = 'All heads active'; deadInsight.className = 'graph-insight good'; }
            else { deadInsight.textContent = `${deadHeads} dead`; deadInsight.className = 'graph-insight bad'; }
        }
        const confInsight = document.getElementById('graph-conf-insight');
        if (confInsight) {
            if (lossNorm > 0.5) { confInsight.textContent = 'Loss decreasing'; confInsight.className = 'graph-insight good'; }
            else if (lossNorm > 0.2) { confInsight.textContent = 'Converging'; confInsight.className = 'graph-insight warn'; }
            else { confInsight.textContent = 'High loss'; confInsight.className = 'graph-insight neutral'; }
        }
        const scaffInsight = document.getElementById('graph-scaffold-insight');
        if (scaffInsight) {
            if (variance > 0.1) { scaffInsight.textContent = 'Heads differentiating'; scaffInsight.className = 'graph-insight good'; }
            else { scaffInsight.textContent = 'Heads similar'; scaffInsight.className = 'graph-insight neutral'; }
        }
    }

    drawMiniGraph(canvasId, data, color, min, max) {
        const canvas = document.getElementById(canvasId);
        if (!canvas || data.length < 2) return;
        const ctx = canvas.getContext('2d');
        const w = canvas.width;
        const h = canvas.height;

        ctx.fillStyle = '#1a1a2e';
        ctx.fillRect(0, 0, w, h);

        // Grid
        ctx.strokeStyle = '#2a3a5e';
        ctx.lineWidth = 0.5;
        const midY = h / 2;
        ctx.beginPath(); ctx.moveTo(0, midY); ctx.lineTo(w, midY); ctx.stroke();

        // Parse CSS color
        ctx.strokeStyle = color;
        ctx.lineWidth = 1.5;
        ctx.beginPath();

        const n = data.length;
        const xStep = w / (n - 1);
        const range = max - min || 1;

        for (let i = 0; i < n; i++) {
            const x = i * xStep;
            const y = h - ((data[i] - min) / range) * h;
            if (i === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        }
        ctx.stroke();

        // Current value
        const last = data[n - 1];
        ctx.fillStyle = color;
        ctx.beginPath();
        ctx.arc((n - 1) * xStep, h - ((last - min) / range) * h, 2.5, 0, Math.PI * 2);
        ctx.fill();

        // Value label
        ctx.fillStyle = '#e0e0e0';
        ctx.font = '9px sans-serif';
        ctx.textAlign = 'right';
        const label = (last * 100).toFixed(0) + '%';
        ctx.fillText(label, w - 2, 10);
    }

    updateInsights(gh, stats) {
        // Accuracy insight
        const acc = gh.accuracy;
        const accInsight = document.getElementById('graph-acc-insight');
        if (accInsight && acc.length > 10) {
            const recent = acc.slice(-10);
            const trend = recent[recent.length - 1] - recent[0];
            if (acc[acc.length - 1] > 0.8) {
                accInsight.textContent = 'Excellent';
                accInsight.className = 'graph-insight good';
            } else if (trend > 0.05) {
                accInsight.textContent = 'Improving';
                accInsight.className = 'graph-insight good';
            } else if (trend < -0.05) {
                accInsight.textContent = 'Declining';
                accInsight.className = 'graph-insight bad';
            } else if (acc[acc.length - 1] > 0.4) {
                accInsight.textContent = 'Stable';
                accInsight.className = 'graph-insight neutral';
            } else {
                accInsight.textContent = 'Learning';
                accInsight.className = 'graph-insight warn';
            }
        }

        // Activity insight
        const actInsight = document.getElementById('graph-activity-insight');
        if (actInsight && gh.active_pct.length > 0) {
            const active = gh.active_pct[gh.active_pct.length - 1];
            if (active > 0.7) {
                actInsight.textContent = 'High utilization';
                actInsight.className = 'graph-insight good';
            } else if (active > 0.4) {
                actInsight.textContent = 'Normal';
                actInsight.className = 'graph-insight neutral';
            } else if (active > 0.2) {
                actInsight.textContent = 'Low - may need more training';
                actInsight.className = 'graph-insight warn';
            } else {
                actInsight.textContent = 'Very low - check network size';
                actInsight.className = 'graph-insight bad';
            }
        }

        // Dead neurons insight
        const deadInsight = document.getElementById('graph-dead-insight');
        if (deadInsight && gh.dead_pct.length > 0) {
            const dead = gh.dead_pct[gh.dead_pct.length - 1];
            if (dead < 0.05) {
                deadInsight.textContent = 'Healthy';
                deadInsight.className = 'graph-insight good';
            } else if (dead < 0.2) {
                deadInsight.textContent = 'Some inactive - normal';
                deadInsight.className = 'graph-insight neutral';
            } else if (dead < 0.4) {
                deadInsight.textContent = 'High - homeostasis active';
                deadInsight.className = 'graph-insight warn';
            } else {
                deadInsight.textContent = 'Critical - network may need rebuild';
                deadInsight.className = 'graph-insight bad';
            }
        }

        // Confidence insight
        const confInsight = document.getElementById('graph-conf-insight');
        if (confInsight && gh.confidence.length > 0) {
            const conf = gh.confidence[gh.confidence.length - 1];
            if (conf > 0.8) {
                confInsight.textContent = 'High agreement';
                confInsight.className = 'graph-insight good';
            } else if (conf > 0.5) {
                confInsight.textContent = 'Moderate';
                confInsight.className = 'graph-insight neutral';
            } else {
                confInsight.textContent = 'Low - exploring';
                confInsight.className = 'graph-insight warn';
            }
        }

        // Scaffold insight
        const scaffInsight = document.getElementById('graph-scaffold-insight');
        if (scaffInsight && gh.scaffold.length > 0) {
            const eps = gh.scaffold[gh.scaffold.length - 1];
            if (eps > 0.3) {
                scaffInsight.textContent = 'High exploration';
                scaffInsight.className = 'graph-insight warn';
            } else if (eps > 0.1) {
                scaffInsight.textContent = 'Balanced';
                scaffInsight.className = 'graph-insight neutral';
            } else {
                scaffInsight.textContent = 'Exploiting learned patterns';
                scaffInsight.className = 'graph-insight good';
            }
        }
    }

    updateCurriculumBars(stats) {
        const container = document.getElementById('curriculum-bars');
        if (!container || !stats.curriculum) return;

        const levels = stats.curriculum.levels || [];
        if (!Array.isArray(levels)) return;

        let html = '';
        levels.forEach(l => {
            const acc = l.accuracy || 0;
            const height = Math.max(4, acc * 70);
            const color = l.passed ? 'var(--green)' :
                          acc > 0.5 ? 'var(--accent)' :
                          acc > 0 ? 'var(--orange)' : 'var(--border)';
            const name = (l.name || '').substring(0, 8);
            html += `<div style="display:flex;flex-direction:column;align-items:center;gap:2px;min-width:20px" title="${l.name}: ${(acc * 100).toFixed(0)}%">
                <div style="font-size:8px;color:var(--text-dim)">${(acc * 100).toFixed(0)}</div>
                <div style="width:14px;height:${height}px;background:${color};border-radius:2px"></div>
                <div style="font-size:7px;color:var(--text-dim);overflow:hidden;text-overflow:ellipsis;max-width:30px">${name}</div>
            </div>`;
        });
        container.innerHTML = html;
    }

    // --- Daemon Builder ---

    updateDaemonForm() {
        const template = document.getElementById('daemon-template').value;
        const paramsDiv = document.getElementById('daemon-params');
        const customEditor = document.getElementById('daemon-custom-editor');

        customEditor.style.display = template === 'custom' ? 'block' : 'none';

        const descriptions = {
            'pattern': 'Learns feature-action profiles via cosine similarity. Improves automatically from correct answers.',
            'math': 'Separate profiles per operator type. Best for math curricula where features encode operators.',
            'feature_group': 'Divides features into groups (one per action) and picks the group with highest total.',
            'threshold': 'Fires when a specific feature exceeds a threshold. Good for known decision boundaries.',
            'argmax': 'Simply picks the action matching the highest feature index. Fast baseline.',
            'random': 'Random selection. Use to measure how much better your other daemons are.',
            'custom': 'Write your own reasoning logic. Full control.',
        };

        let html = `<div style="font-size:10px;color:var(--text-dim);margin-bottom:6px">${descriptions[template] || ''}</div>`;

        if (template === 'threshold') {
            html += `
                <div style="display:flex;gap:4px;margin-bottom:4px">
                    <div style="flex:1">
                        <div style="font-size:10px;color:var(--text-dim)">Feature Index</div>
                        <input id="dp-target-feature" type="number" value="0" min="0" style="width:100%;padding:3px;background:var(--bg-dark);border:1px solid var(--border);color:var(--text);border-radius:3px;font-size:12px">
                    </div>
                    <div style="flex:1">
                        <div style="font-size:10px;color:var(--text-dim)">Threshold</div>
                        <input id="dp-threshold" type="number" value="0.5" step="0.1" style="width:100%;padding:3px;background:var(--bg-dark);border:1px solid var(--border);color:var(--text);border-radius:3px;font-size:12px">
                    </div>
                    <div style="flex:1">
                        <div style="font-size:10px;color:var(--text-dim)">Action</div>
                        <input id="dp-action" type="number" value="0" min="0" style="width:100%;padding:3px;background:var(--bg-dark);border:1px solid var(--border);color:var(--text);border-radius:3px;font-size:12px">
                    </div>
                </div>`;
        }

        paramsDiv.innerHTML = html;
    }

    async refreshDaemonList() {
        const container = document.getElementById('build-daemon-list');
        if (!container) return;

        try {
            const data = await fetch('/api/daemons/active').then(r => r.json());
            const daemons = data.daemons || [];

            if (daemons.length === 0) {
                container.innerHTML = '<div style="color:var(--text-dim);font-size:11px">No daemons active</div>';
                return;
            }

            let html = '';
            daemons.forEach(d => {
                const phaseColors = {
                    'APPRENTICE': 'var(--text-dim)', 'JOURNEYMAN': 'var(--orange)',
                    'COMPETENT': 'var(--accent)', 'EXPERT': 'var(--green)',
                    'INDEPENDENT': 'var(--purple)',
                };
                const pc = phaseColors[d.phase] || 'var(--text-dim)';
                const rate = (d.acceptance_rate * 100).toFixed(0);
                const escapedName = d.name.replace(/'/g, "\\'");

                html += `<div style="display:flex;justify-content:space-between;align-items:center;padding:4px 0;border-bottom:1px solid rgba(255,255,255,0.05);font-size:11px">
                    <div>
                        <span style="color:var(--accent);font-weight:600">${d.name}</span>
                        <span style="color:var(--text-dim);margin-left:4px">${d.type}</span>
                    </div>
                    <div style="display:flex;align-items:center;gap:6px">
                        <span style="color:${pc};font-size:10px">${d.phase}</span>
                        <span style="color:var(--text-dim);font-size:10px">${rate}%</span>
                        <span onclick="app.removeDaemon('${escapedName}')" style="color:var(--red);cursor:pointer;font-size:13px" title="Remove">&times;</span>
                    </div>
                </div>`;
            });

            container.innerHTML = html;
        } catch (err) {
            container.innerHTML = `<div style="color:var(--red);font-size:11px">Error: ${err}</div>`;
        }
    }

    async addDaemon() {
        const template = document.getElementById('daemon-template').value;
        const name = document.getElementById('daemon-name').value.trim();

        if (!name) {
            alert('Enter a daemon name');
            return;
        }

        const params = {
            template: template,
            name: name,
            num_actions: parseInt(document.getElementById('build-output')?.value) || 5,
        };

        // Template-specific params
        if (template === 'threshold') {
            params.target_feature = parseInt(document.getElementById('dp-target-feature')?.value) || 0;
            params.threshold = parseFloat(document.getElementById('dp-threshold')?.value) || 0.5;
            params.action = parseInt(document.getElementById('dp-action')?.value) || 0;
        }

        if (template === 'custom') {
            params.custom_code = document.getElementById('daemon-custom-code').value;
        }

        try {
            const res = await fetch('/api/daemons/add', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(params),
            }).then(r => r.json());

            if (res.error) {
                alert('Failed: ' + res.error);
            } else {
                document.getElementById('daemon-name').value = '';
                this.refreshDaemonList();
            }
        } catch (err) {
            alert('Failed: ' + err);
        }
    }

    async removeDaemon(name) {
        try {
            const res = await fetch('/api/daemons/remove', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ name: name }),
            }).then(r => r.json());

            if (res.error) {
                alert(res.error);
            } else {
                this.refreshDaemonList();
            }
        } catch (err) {
            alert('Failed: ' + err);
        }
    }

    // --- Build Tab ---

    addBuildLayer() {
        const container = document.getElementById('build-layers');
        const row = document.createElement('div');
        row.className = 'build-layer-row';
        row.innerHTML = '<input type="number" value="16" class="build-layer-input"><span style="font-size:10px;color:var(--text-dim)">neurons</span>';
        container.appendChild(row);
    }

    removeBuildLayer() {
        const container = document.getElementById('build-layers');
        if (container.children.length > 1) {
            container.removeChild(container.lastChild);
        }
    }

    async buildAndDeploy() {
        const inputDim = parseInt(document.getElementById('build-input').value) || 25;
        const outputDim = parseInt(document.getElementById('build-output').value) || 5;
        const lr = parseFloat(document.getElementById('build-lr').value) || 0.01;
        const epsilon = parseFloat(document.getElementById('build-eps').value) || 0.3;

        // Gather hidden dims
        const inputs = document.querySelectorAll('.build-layer-input');
        const hiddenDims = [];
        inputs.forEach(inp => {
            const v = parseInt(inp.value);
            if (v > 0) hiddenDims.push(v);
        });

        if (hiddenDims.length === 0) {
            alert('Add at least one hidden layer');
            return;
        }

        try {
            const res = await fetch('/api/network/rebuild', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    input_dim: inputDim,
                    output_dim: outputDim,
                    hidden_dims: hiddenDims,
                    learning_rate: lr,
                    epsilon: epsilon,
                }),
            }).then(r => r.json());

            if (res.error) {
                alert('Build failed: ' + res.error);
                return;
            }

            // Reset graph history
            this.graphHistory = null;

            alert(`Model built!\n${res.neurons} neurons, ${res.connections} connections\nLayers: [${inputDim}, ${hiddenDims.join(', ')}, ${outputDim}]`);
            this.switchToModel('HDNA (primary)');
        } catch (err) {
            alert('Build failed: ' + err);
        }
    }

    // --- Network Configuration ---

    applyNetworkPreset() {
        const preset = document.getElementById('net-preset').value;
        const hiddenInput = document.getElementById('net-hidden');
        const presets = {
            'small': '16, 8',
            'medium': '32, 16',
            'large': '48, 24, 12',
            'deep': '32, 24, 16, 8',
            'wide': '64, 32',
        };
        if (presets[preset]) {
            hiddenInput.value = presets[preset];
        }
    }

    async rebuildNetwork() {
        const inputDim = parseInt(document.getElementById('net-input').value) || 25;
        const outputDim = parseInt(document.getElementById('net-output').value) || 5;
        const hiddenStr = document.getElementById('net-hidden').value;
        const hiddenDims = hiddenStr.split(',').map(s => parseInt(s.trim())).filter(n => n > 0);

        if (hiddenDims.length === 0) {
            alert('Enter at least one hidden dimension');
            return;
        }

        try {
            const res = await fetch('/api/network/rebuild', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    input_dim: inputDim,
                    output_dim: outputDim,
                    hidden_dims: hiddenDims,
                }),
            }).then(r => r.json());

            if (res.error) {
                alert('Rebuild failed: ' + res.error);
            } else {
                alert(`Network rebuilt: ${res.neurons} neurons, ${res.connections} connections\nHidden: [${hiddenDims.join(', ')}]`);
                // Reload the 3D view
                this.switchToModel('HDNA (primary)');
            }
        } catch (err) {
            alert('Rebuild failed: ' + err);
        }
    }

    // --- Curriculum Management ---

    async refreshCurriculaDropdown() {
        try {
            const data = await fetch('/api/curricula').then(r => r.json());
            const select = document.getElementById('train-curriculum');
            select.innerHTML = '';

            const curricula = data.curricula || {};
            // Sort: demo first, then by name
            const sorted = Object.entries(curricula).sort((a, b) => {
                const aDemo = a[1].tags && a[1].tags.includes('demo');
                const bDemo = b[1].tags && b[1].tags.includes('demo');
                if (aDemo && !bDemo) return -1;
                if (!aDemo && bDemo) return 1;
                return a[0].localeCompare(b[0]);
            });

            sorted.forEach(([name, info]) => {
                const opt = document.createElement('option');
                opt.value = name;
                opt.textContent = `${name} — ${info.description || ''}`.substring(0, 60);
                if (info.tags && info.tags.includes('custom')) {
                    opt.style.color = 'var(--orange)';
                }
                select.appendChild(opt);
            });
        } catch (err) {
            console.error('Failed to load curricula:', err);
        }
    }

    async loadCurriculumFile() {
        const pathInput = document.getElementById('load-cur-path');
        const nameInput = document.getElementById('load-cur-name');
        const filePath = pathInput.value.trim();
        const name = nameInput.value.trim();

        if (!filePath) {
            alert('Enter a path to a .json or .csv file');
            return;
        }

        try {
            const res = await fetch('/api/curricula/load_file', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ path: filePath, name: name }),
            }).then(r => r.json());

            if (res.error) {
                alert('Load failed: ' + res.error);
            } else {
                alert(`Loaded "${res.name}": ${res.levels} levels, ${res.total_tasks} tasks`);
                pathInput.value = '';
                nameInput.value = '';
                this.refreshCurriculaDropdown();
            }
        } catch (err) {
            alert('Load failed: ' + err);
        }
    }

    // --- Governance Mode ---

    toggleMode() {
        this.governanceMode = !this.governanceMode;
        const btn = document.getElementById('btn-mode');
        const overlay = document.getElementById('governance-overlay');
        const sidepanel = document.getElementById('sidepanel');

        if (this.governanceMode) {
            btn.textContent = 'Research View';
            btn.style.borderColor = 'var(--green)';
            btn.style.color = 'var(--green)';
            overlay.style.display = 'block';
            sidepanel.style.display = 'none';
            document.querySelector('#header h1').textContent = 'AI Governance Dashboard';
            this.updateGovernancePanel();
            // Auto-refresh while in governance mode
            this._govRefreshInterval = setInterval(() => this.updateGovernancePanel(), 2000);
        } else {
            btn.textContent = 'Governance View';
            btn.style.borderColor = 'var(--border)';
            btn.style.color = 'var(--accent)';
            overlay.style.display = 'none';
            sidepanel.style.display = 'flex';
            document.querySelector('#header h1').textContent = 'HDNA Workbench';
            if (this._govRefreshInterval) {
                clearInterval(this._govRefreshInterval);
                this._govRefreshInterval = null;
            }
        }
    }

    async updateGovernancePanel() {
        try {
            // Check if transformer is active
            const txRes = await fetch('/api/transformer').then(r => r.json());
            const isTransformer = txRes.exists && this._transformerRunning;

            const [modelRes, stressRes, auditRes] = await Promise.all([
                fetch('/api/model').then(r => r.json()),
                fetch('/api/stress').then(r => r.json()),
                fetch('/api/audit?count=200').then(r => r.json()),
            ]);

            const stats = auditRes.stats || {};

            if (isTransformer) {
                // --- Transformer Governance ---
                const snap = txRes.snapshot || {};
                const headReport = txRes.head_report || [];
                const expertReport = txRes.expert_report || [];

                document.getElementById('gov-model-name').textContent =
                    `Inspectable Transformer (${(snap.parameters || 0).toLocaleString()} params)`;

                document.getElementById('gov-total-decisions').textContent =
                    (snap.total_forwards || 0).toLocaleString() + ' forward passes';

                // Accuracy from training
                const acc = this.chartData.bestAccuracy || 0;
                const accEl = document.getElementById('gov-accuracy');
                accEl.textContent = (acc * 100).toFixed(1) + '%';
                accEl.className = 'value ' + (acc > 0.7 ? 'good' : acc > 0.3 ? 'warn' : 'bad');

                // Components
                const totalHeads = headReport.length;
                const activeHeads = headReport.filter(h => !h.is_dead).length;
                const specializedHeads = headReport.filter(h => h.tag !== 'balanced' && !h.tag.startsWith('head_')).length;
                document.getElementById('gov-components').textContent =
                    `${totalHeads} heads, ${snap.n_experts * snap.n_layers} experts, ${snap.n_layers} layers`;

                // Risk indicators — transformer specific
                const deadHeads = headReport.filter(h => h.is_dead).length;
                const deadEl = document.getElementById('gov-dead');
                deadEl.textContent = `${deadHeads} dead heads (${totalHeads > 0 ? (deadHeads/totalHeads*100).toFixed(0) : 0}%)`;
                deadEl.className = 'value ' + (deadHeads === 0 ? 'good' : deadHeads <= 1 ? 'warn' : 'bad');

                // Stability: head entropy variance (low = stable specialization)
                const entropies = headReport.map(h => h.avg_entropy);
                const entropyVar = entropies.length > 1 ?
                    entropies.reduce((s, e) => s + (e - entropies.reduce((a,b)=>a+b,0)/entropies.length)**2, 0) / entropies.length : 0;
                const stabEl = document.getElementById('gov-stability');
                stabEl.textContent = entropyVar > 0.3 ? 'Heads Specializing' : entropyVar > 0.1 ? 'Moderate Differentiation' : 'Uniform (early training)';
                stabEl.className = 'value ' + (entropyVar > 0.3 ? 'good' : entropyVar > 0.1 ? 'warn' : 'neutral');

                // Drift: expert load balance
                const allUsage = expertReport.flatMap(e => e.usage_pct);
                const maxUsage = Math.max(...allUsage);
                const minUsage = Math.min(...allUsage);
                const imbalance = maxUsage - minUsage;
                const driftEl = document.getElementById('gov-drift');
                driftEl.textContent = imbalance < 20 ? 'Balanced' : imbalance < 40 ? 'Minor Imbalance' : 'Significant Imbalance';
                driftEl.className = 'value ' + (imbalance < 20 ? 'good' : imbalance < 40 ? 'warn' : 'bad');

                // Anomalies
                const anomEl = document.getElementById('gov-anomalies');
                const anomalies = [];
                if (deadHeads > 0) anomalies.push(`${deadHeads} dead head(s)`);
                if (imbalance > 40) anomalies.push('expert load imbalance');
                anomEl.textContent = anomalies.length === 0 ? 'None' : anomalies.join(', ');
                anomEl.className = 'value ' + (anomalies.length === 0 ? 'good' : 'warn');

                // Overall status
                const healthy = deadHeads === 0 && imbalance < 40;
                this._setGovStatus(healthy, anomalies);

                // Compliance
                document.getElementById('gov-art12').textContent = snap.total_forwards > 0 ? 'Compliant' : 'No Data';
                document.getElementById('gov-art12').className = 'value ' + (snap.total_forwards > 0 ? 'good' : 'warn');
                document.getElementById('gov-art13').textContent = 'Compliant';
                document.getElementById('gov-art13').className = 'value good';
                document.getElementById('gov-art15').textContent = healthy ? 'Compliant' : 'Review Needed';
                document.getElementById('gov-art15').className = 'value ' + (healthy ? 'good' : 'warn');

                // Recent activity — show head specialization timeline
                const activityEl = document.getElementById('gov-activity');
                let actHtml = '';
                headReport.forEach(h => {
                    const tagColor = h.is_dead ? 'var(--red)' :
                        h.tag === 'sharp_selector' ? 'var(--green)' :
                        h.tag === 'local_focus' ? 'var(--accent)' :
                        h.tag === 'position_tracker' ? 'var(--purple)' :
                        h.tag === 'global_mixer' ? 'var(--orange)' :
                        'var(--text-dim)';
                    actHtml += `<div style="display:flex;justify-content:space-between;padding:2px 0;font-size:11px;border-bottom:1px solid rgba(255,255,255,0.05)">
                        <span>L${h.layer} Head ${h.head}</span>
                        <span style="color:${tagColor};font-weight:600">${h.tag}</span>
                        <span style="color:var(--text-dim)">ent:${h.avg_entropy.toFixed(2)}</span>
                    </div>`;
                });
                // Expert usage
                expertReport.forEach(e => {
                    actHtml += `<div style="display:flex;justify-content:space-between;padding:2px 0;font-size:11px;border-bottom:1px solid rgba(255,255,255,0.05)">
                        <span style="color:var(--orange)">Layer ${e.layer} Experts</span>
                        <span style="color:var(--text-dim)">${e.usage_pct.map(p => p.toFixed(0) + '%').join(' | ')}</span>
                    </div>`;
                });
                activityEl.innerHTML = actHtml;

            } else {
                // --- HDNA Governance ---
                // Fetch real governance metrics
                const govRes = await fetch('/api/governance').then(r => r.json());
                const util = govRes.utilization || {};
                const stab = govRes.stability || {};
                const drift = govRes.drift || {};

                document.getElementById('gov-model-name').textContent = modelRes.name || 'Unknown';
                document.getElementById('gov-total-decisions').textContent =
                    (stats.total_predictions || 0).toLocaleString();

                const acc = stats.accuracy_100 || 0;
                const accEl = document.getElementById('gov-accuracy');
                accEl.textContent = (acc * 100).toFixed(1) + '%';
                accEl.className = 'value ' + (acc > 0.7 ? 'good' : acc > 0.4 ? 'warn' : 'bad');

                const extra = modelRes.extra || {};
                document.getElementById('gov-components').textContent =
                    `${util.active || 0} active, ${util.inactive || 0} available capacity (${util.total || 0} total)`;

                // Component Utilization (not "inactive = bad", but "available capacity")
                const deadEl = document.getElementById('gov-dead');
                deadEl.textContent = `${util.utilization_pct || 0}% utilized, ${util.available_capacity_pct || 0}% available`;
                deadEl.className = 'value ' + (util.utilization_pct > 30 ? 'good' : 'warn');

                // Decision Stability (real metric from daemon cluster tightness)
                const stabEl = document.getElementById('gov-stability');
                stabEl.textContent = `${stab.detail || 'Unknown'} (${stab.score || 0}%)`;
                stabEl.className = 'value ' + (stab.score > 70 ? 'good' : stab.score > 40 ? 'warn' : 'neutral');

                // Model Drift (real metric from centroid movement)
                const driftEl = document.getElementById('gov-drift');
                driftEl.textContent = drift.detail || 'Unknown';
                driftEl.className = 'value ' + (
                    drift.detail && drift.detail.includes('Stable') ? 'good' :
                    drift.detail && drift.detail.includes('Minimal') ? 'good' :
                    drift.detail && drift.detail.includes('Moderate') ? 'warn' : 'neutral'
                );

                // Anomalies
                const anomalies = [];
                if (util.utilization_pct < 10 && stats.total_predictions > 50) anomalies.push('Very low utilization');
                if (stab.score > 0 && stab.score < 30 && stats.total_predictions > 100) anomalies.push('Decision inconsistency');
                if (drift.score > 0.5) anomalies.push('Significant drift');
                const anomEl = document.getElementById('gov-anomalies');
                anomEl.textContent = anomalies.length === 0 ? 'None' : anomalies.join(', ');
                anomEl.className = 'value ' + (anomalies.length === 0 ? 'good' : 'warn');

                const healthy = anomalies.length === 0 && (acc > 0.5 || stats.total_predictions < 50);
                this._setGovStatus(healthy, anomalies);

                // Compliance
                document.getElementById('gov-art12').textContent = stats.total_predictions > 0 ? 'Compliant' : 'No Data';
                document.getElementById('gov-art12').className = 'value ' + (stats.total_predictions > 0 ? 'good' : 'warn');
                const art15El = document.getElementById('gov-art15');
                art15El.textContent = healthy ? 'Compliant' : 'Review Needed';
                art15El.className = 'value ' + (healthy ? 'good' : 'warn');

                // Recent activity
                const activityEl = document.getElementById('gov-activity');
                const records = (auditRes.records || []).slice(-8).reverse();
                let actHtml = '';
                if (records.length > 0) {
                    records.forEach(r => {
                        const icon = r.correct ? '<span style="color:var(--green)">&#10003;</span>' :
                                                 '<span style="color:var(--red)">&#10007;</span>';
                        actHtml += `<div style="display:flex;justify-content:space-between;padding:2px 0;font-size:11px;border-bottom:1px solid rgba(255,255,255,0.05)">
                            <span>${icon} Decision #${r.step}</span>
                            <span style="color:var(--text-dim)">Confidence: ${(Math.min(1, Math.max(0, r.confidence)) * 100).toFixed(0)}%</span>
                            <span style="color:var(--text-dim)">${r.source}</span>
                        </div>`;
                    });
                }
                // Add utilization context
                actHtml += `<div style="margin-top:6px;padding-top:4px;border-top:1px solid var(--border);font-size:10px;color:var(--text-dim)">
                    ${util.active} components active for current task. ${util.inactive} available for additional tasks or complexity.
                </div>`;
                activityEl.innerHTML = actHtml;
            }

        } catch (err) {
            console.error('Failed to update governance panel:', err);
        }
    }

    _setGovStatus(healthy, issues) {
        const statusLight = document.getElementById('gov-status-light');
        const statusText = document.getElementById('gov-status-text');
        const statusSub = document.getElementById('gov-status-sub');
        const statusCard = document.getElementById('gov-status-card');

        if (healthy) {
            statusLight.style.background = 'var(--green)';
            statusText.textContent = 'System Healthy';
            statusText.style.color = 'var(--green)';
            statusSub.textContent = 'All components operating normally';
            statusCard.style.borderColor = 'var(--green)';
        } else if (issues && issues.length > 0) {
            statusLight.style.background = 'var(--orange)';
            statusText.textContent = 'Review Recommended';
            statusText.style.color = 'var(--orange)';
            statusSub.textContent = Array.isArray(issues) ? issues.join(', ') : String(issues);
            statusCard.style.borderColor = 'var(--orange)';
        } else {
            statusLight.style.background = 'var(--orange)';
            statusText.textContent = 'Review Recommended';
            statusText.style.color = 'var(--orange)';
            statusSub.textContent = 'Some metrics outside optimal range';
            statusCard.style.borderColor = 'var(--orange)';
        }
    }

    async generateReport() {
        const timestamp = new Date().toISOString().split('T')[0];
        try {
            const [modelRes, stressRes, auditRes] = await Promise.all([
                fetch('/api/model').then(r => r.json()),
                fetch('/api/stress').then(r => r.json()),
                fetch('/api/audit?count=1000').then(r => r.json()),
            ]);

            const stats = auditRes.stats || {};
            const dead = stressRes.dead_pct || 0;
            const healthy = stressRes.is_healthy !== false;
            const warnings = stressRes.warnings || [];

            let report = `AI GOVERNANCE REPORT
Generated: ${new Date().toLocaleString()}
${'='.repeat(50)}

MODEL INFORMATION
  Name: ${modelRes.name || 'Unknown'}
  Framework: ${modelRes.framework || 'Unknown'}
  Architecture: ${modelRes.architecture || 'Unknown'}
  Parameters: ${(modelRes.parameter_count || 0).toLocaleString()}
  Layers: ${modelRes.layer_count || 0}

SYSTEM STATUS: ${healthy ? 'HEALTHY' : 'REVIEW REQUIRED'}
${'='.repeat(50)}

KEY METRICS
  Total Decisions Logged: ${(stats.total_predictions || 0).toLocaleString()}
  Decision Accuracy (last 100): ${((stats.accuracy_100 || 0) * 100).toFixed(1)}%
  Overall Accuracy: ${((stats.accuracy_all || 0) * 100).toFixed(1)}%
  Novelty Rate: ${((stats.novelty_rate || 0) * 100).toFixed(1)}%

RISK INDICATORS
  Inactive Components: ${dead.toFixed(1)}% ${dead < 10 ? '(Low Risk)' : dead < 30 ? '(Medium Risk)' : '(High Risk)'}
  Decision Stability: ${(stressRes.avg_jitter || 0) < 0.01 ? 'Stable' : 'Fluctuating'}
  Model Drift: ${(stressRes.avg_weight_drift || 0) < 0.001 ? 'None Detected' : 'Detected'}
  Active Warnings: ${warnings.length === 0 ? 'None' : warnings.join(', ')}

COMPLIANCE STATUS
  EU AI Act Article 12 (Record-Keeping): ${stats.total_predictions > 0 ? 'COMPLIANT - All decisions logged with full audit trail' : 'NO DATA'}
  EU AI Act Article 13 (Transparency): COMPLIANT - Decision replay and inspection available
  EU AI Act Article 14 (Human Oversight): COMPLIANT - Breakpoints and intervention capabilities active
  EU AI Act Article 15 (Robustness): ${healthy ? 'COMPLIANT - Health monitoring active, no issues' : 'REVIEW - ' + warnings.join(', ')}
  NIST AI RMF: ALIGNED - Govern, Map, Measure, Manage functions covered

AUDIT TRAIL
  Logging: Active (append-only)
  Records Retained: ${(stats.total_predictions || 0).toLocaleString()}
  Replay Capability: Available for all logged decisions
  Export Format: JSON (machine-readable)

${'='.repeat(50)}
Report generated by HDNA Workbench AI Governance Dashboard
https://github.com/staffman76/HDNA-Workbench
`;

            // Download as text file
            const blob = new Blob([report], { type: 'text/plain' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `ai_governance_report_${timestamp}.txt`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);

        } catch (err) {
            alert('Failed to generate report: ' + err);
        }
    }

    async exportAuditLog() {
        try {
            const data = await fetch('/api/audit?count=10000').then(r => r.json());
            const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `audit_log_${new Date().toISOString().split('T')[0]}.json`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
        } catch (err) {
            alert('Failed to export: ' + err);
        }
    }

    // --- External Model Management ---

    async loadExternalModel() {
        const typeSelect = document.getElementById('load-model-type');
        const pathInput = document.getElementById('load-model-path');
        const nameInput = document.getElementById('load-model-name');

        let modelType = typeSelect.value;
        let modelPath = pathInput.value.trim();
        let modelName = nameInput.value.trim();

        // Map UI options to API params
        if (modelType === 'pytorch') {
            modelPath = '';  // demo model
            modelName = modelName || 'PyTorch Demo MLP';
        } else if (modelType === 'pytorch_file') {
            modelType = 'pytorch';
            if (!modelPath) { alert('Enter a .pt file path'); return; }
        } else if (modelType === 'huggingface') {
            if (!modelPath) { alert('Enter a HuggingFace model name (e.g. distilgpt2)'); return; }
            modelName = modelName || modelPath;
        } else if (modelType === 'onnx') {
            if (!modelPath) { alert('Enter an .onnx file path'); return; }
        }

        const btn = event.target;
        btn.textContent = 'Loading...';
        btn.disabled = true;

        try {
            const res = await fetch('/api/models/load', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    type: modelType,
                    path: modelPath,
                    name: modelName,
                }),
            }).then(r => r.json());

            if (res.error) {
                alert('Load failed: ' + res.error);
            } else {
                pathInput.value = '';
                nameInput.value = '';
                alert('Loaded: ' + res.name + ' (' + res.info.parameter_count.toLocaleString() + ' params, ' + res.capabilities + ')');
                await this.refreshModelsList();
            }
        } catch (err) {
            alert('Load failed: ' + err);
        }

        btn.textContent = 'Load Model';
        btn.disabled = false;
    }

    async refreshModelsList() {
        const container = document.getElementById('models-list');
        try {
            const data = await fetch('/api/models').then(r => r.json());
            if (!data.models || Object.keys(data.models).length === 0) {
                container.innerHTML = '<div class="loading">No models loaded</div>';
                return;
            }

            let html = '';
            Object.entries(data.models).forEach(([name, model]) => {
                const info = model.info;
                const isPrimary = model.is_primary;
                const borderColor = isPrimary ? 'var(--accent)' : 'var(--border)';

                html += `<div class="card" style="border-color:${borderColor}">
                    <div class="card-header" onclick="this.nextElementSibling.classList.toggle('collapsed')">
                        <h3>${isPrimary ? '&#9733; ' : ''}${name}</h3>
                        <span class="toggle" style="color:var(--text-dim)">${info.framework} &#9660;</span>
                    </div>
                    <div class="card-body">
                        <div class="stat-row"><span class="label">Framework</span><span class="value">${info.framework}</span></div>
                        <div class="stat-row"><span class="label">Architecture</span><span class="value">${info.architecture}</span></div>
                        <div class="stat-row"><span class="label">Parameters</span><span class="value">${(info.parameter_count || 0).toLocaleString()}</span></div>
                        <div class="stat-row"><span class="label">Layers</span><span class="value">${info.layer_count || 0}</span></div>
                        <div class="stat-row"><span class="label">Capabilities</span><span class="value" style="font-size:10px;word-break:break-all">${model.capabilities}</span></div>`;

                {
                    const escapedName = name.replace(/'/g, "\\'");
                    const viewLabel = (isPrimary && this.activeModel !== name && this.activeModel !== null)
                        ? 'View' : (!isPrimary ? 'View' : '');
                    html += `<div style="margin-top:6px;display:flex;gap:4px">
                        <button onclick="app.switchToModel('${escapedName}')" style="flex:1;padding:4px;background:var(--bg-dark);border:1px solid var(--accent);color:var(--accent);border-radius:3px;cursor:pointer;font-size:11px">View</button>`;
                    if (!isPrimary) {
                        html += `<button onclick="app.inspectModel('${escapedName}')" style="flex:1;padding:4px;background:var(--bg-dark);border:1px solid var(--border);color:var(--text);border-radius:3px;cursor:pointer;font-size:11px">Inspect</button>
                        <button onclick="app.compareWithPrimary('${escapedName}')" style="flex:1;padding:4px;background:var(--bg-dark);border:1px solid var(--border);color:var(--orange);border-radius:3px;cursor:pointer;font-size:11px">Compare</button>`;
                    }
                    html += `</div>`;
                }

                html += `</div></div>`;
            });

            container.innerHTML = html;
        } catch (err) {
            container.innerHTML = `<div class="loading">Error: ${err}</div>`;
        }
    }

    async inspectModel(name) {
        const container = document.getElementById('compare-results');
        container.innerHTML = '<div class="loading">Inspecting...</div>';

        try {
            const res = await fetch('/api/models/inspect', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ name: name }),
            }).then(r => r.json());

            if (res.error) {
                container.innerHTML = `<div class="card"><div class="card-body" style="color:var(--red)">${res.error}</div></div>`;
                return;
            }

            let html = `<div class="card" style="border-color:var(--accent)">
                <div class="card-header"><h3>Inspection: ${name}</h3></div>
                <div class="card-body">`;

            const info = res.info || {};
            html += `<div class="stat-row"><span class="label">Tier</span><span class="value">${res.tier || '?'}</span></div>`;

            // Layers
            if (res.layers) {
                html += `<div style="margin-top:8px;color:var(--accent);font-weight:600;font-size:11px">Layers (${res.layers.length})</div>`;
                res.layers.slice(0, 15).forEach(l => {
                    const inspectable = l.inspectable ? ' <span style="color:var(--green)">[I]</span>' : '';
                    html += `<div class="stat-row">
                        <span class="label">${l.name}${inspectable}</span>
                        <span class="value">${l.type} (${(l.parameter_count || 0).toLocaleString()})</span>
                    </div>`;
                });
                if (res.layers.length > 15) {
                    html += `<div style="color:var(--text-dim);font-size:10px">...and ${res.layers.length - 15} more</div>`;
                }
            }

            html += `</div></div>`;
            container.innerHTML = html;
        } catch (err) {
            container.innerHTML = `<div class="card"><div class="card-body" style="color:var(--red)">Error: ${err}</div></div>`;
        }
    }

    async compareWithPrimary(name) {
        const container = document.getElementById('compare-results');
        container.innerHTML = '<div class="loading">Comparing...</div>';

        try {
            const res = await fetch('/api/models/compare', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ models: [name] }),
            }).then(r => r.json());

            if (res.error) {
                container.innerHTML = `<div class="card"><div class="card-body" style="color:var(--red)">${res.error}</div></div>`;
                return;
            }

            let html = `<div class="card" style="border-color:var(--orange)">
                <div class="card-header"><h3>Comparison: HDNA vs ${name}</h3></div>
                <div class="card-body">`;

            html += `<div class="stat-row"><span class="label">Input Shape</span><span class="value">${JSON.stringify(res.input_shape)}</span></div>`;

            Object.entries(res.results).forEach(([modelName, result]) => {
                const color = modelName.includes('HDNA') ? 'var(--accent)' : 'var(--purple)';
                html += `<div style="margin-top:8px;color:${color};font-weight:600;font-size:11px">${modelName} (${result.framework || '?'})</div>`;

                if (result.error) {
                    html += `<div style="color:var(--red);font-size:11px">${result.error}</div>`;
                } else {
                    const output = Array.isArray(result.output) ? result.output.map(v => v.toFixed(4)).join(', ') : JSON.stringify(result.output);
                    html += `<div class="stat-row"><span class="label">Output</span><span class="value" style="font-size:10px;word-break:break-all">[${output}]</span></div>`;

                    if (result.activations) {
                        html += `<div style="margin-top:4px;font-size:10px;color:var(--text-dim)">Activations (${result.activations.length} layers):</div>`;
                        result.activations.slice(0, 8).forEach(a => {
                            html += `<div class="stat-row" style="font-size:10px">
                                <span class="label">${a.layer}</span>
                                <span class="value">mean=${a.mean.toFixed(4)}, std=${a.std.toFixed(4)}</span>
                            </div>`;
                        });
                    }
                }
            });

            html += `</div></div>`;
            container.innerHTML = html;
        } catch (err) {
            container.innerHTML = `<div class="card"><div class="card-body" style="color:var(--red)">Error: ${err}</div></div>`;
        }
    }

    // --- Accuracy Chart ---

    drawChart() {
        const canvas = document.getElementById('accuracy-chart');
        if (!canvas) return;
        const ctx = canvas.getContext('2d');
        const w = canvas.width;
        const h = canvas.height;

        ctx.fillStyle = '#1a1a2e';
        ctx.fillRect(0, 0, w, h);

        const accData = this.chartData.accuracy;
        const smoothData = this.chartData.smoothed;
        const epsData = this.chartData.epsilon;
        if (accData.length < 2) return;

        const n = accData.length;
        const xStep = w / (n - 1);

        // Grid lines
        ctx.strokeStyle = '#2a3a5e';
        ctx.lineWidth = 0.5;
        [0.25, 0.50, 0.75].forEach(pct => {
            const y = h - pct * h;
            ctx.beginPath();
            ctx.moveTo(0, y);
            ctx.lineTo(w, y);
            ctx.stroke();
        });

        // Y-axis labels
        ctx.fillStyle = '#556677';
        ctx.font = '8px sans-serif';
        ctx.textAlign = 'left';
        ctx.fillText('25%', 2, h - 0.25 * h - 2);
        ctx.fillText('50%', 2, h - 0.50 * h - 2);
        ctx.fillText('75%', 2, h - 0.75 * h - 2);

        // Lifetime average line (horizontal, white dashed)
        const lifetimeAcc = this.chartData.allTotal > 0 ?
            this.chartData.allCorrect / this.chartData.allTotal : 0;
        if (lifetimeAcc > 0) {
            const avgY = h - lifetimeAcc * h;
            ctx.strokeStyle = '#ffffff';
            ctx.lineWidth = 0.5;
            ctx.setLineDash([4, 4]);
            ctx.beginPath();
            ctx.moveTo(0, avgY);
            ctx.lineTo(w, avgY);
            ctx.stroke();
            ctx.setLineDash([]);
            ctx.fillStyle = '#ffffff';
            ctx.font = '8px sans-serif';
            ctx.fillText('avg ' + (lifetimeAcc * 100).toFixed(0) + '%', 2, avgY - 3);
        }

        // Raw accuracy (dim green, thin — the "stock market" line)
        ctx.beginPath();
        ctx.strokeStyle = '#00e676';
        ctx.lineWidth = 0.5;
        ctx.globalAlpha = 0.3;
        for (let i = 0; i < n; i++) {
            const x = i * xStep;
            const y = h - accData[i] * h;
            if (i === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        }
        ctx.stroke();
        ctx.globalAlpha = 1.0;

        // Smoothed accuracy (bright green, thick — the real trend)
        if (smoothData.length >= 2) {
            ctx.beginPath();
            ctx.strokeStyle = '#00e676';
            ctx.lineWidth = 2.5;
            for (let i = 0; i < smoothData.length; i++) {
                const x = i * xStep;
                const y = h - smoothData[i] * h;
                if (i === 0) ctx.moveTo(x, y);
                else ctx.lineTo(x, y);
            }
            ctx.stroke();
        }

        // Current smoothed value dot
        const lastSmooth = smoothData.length > 0 ? smoothData[smoothData.length - 1] : accData[n - 1];
        const dotX = (n - 1) * xStep;
        const dotY = h - lastSmooth * h;
        ctx.beginPath();
        ctx.fillStyle = lastSmooth > 0.7 ? '#00e676' : lastSmooth > 0.4 ? '#ffab40' : '#ff5252';
        ctx.arc(dotX, dotY, 3.5, 0, Math.PI * 2);
        ctx.fill();

        // Current smoothed value label
        ctx.fillStyle = '#e0e0e0';
        ctx.font = 'bold 10px sans-serif';
        ctx.textAlign = 'right';
        ctx.fillText((lastSmooth * 100).toFixed(0) + '%', w - 4, Math.max(12, dotY - 5));
        ctx.textAlign = 'left';
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

        // Stop any existing training first
        if (this.trainInterval) {
            clearTimeout(this.trainInterval);
            this.trainInterval = null;
        }
        try {
            await fetch('/api/train/stop', { method: 'POST' });
        } catch (e) {}

        try {
            const res = await fetch('/api/train/start', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ curriculum: curriculum, phases: 5 }),
            }).then(r => r.json());

            if (res.error) {
                alert('Training failed to start: ' + res.error);
                return;
            }
            console.log('Training started:', JSON.stringify(res, null, 2));
        } catch (err) {
            console.error('Failed to start training:', err);
            alert('Failed to start training: ' + err);
            return;
        }

        this.isTraining = true;
        this.isReplaying = false;
        this.chartData.accuracy = [];
        this.chartData.smoothed = [];
        this.chartData.epsilon = [];
        this.chartData.bestAccuracy = 0;
        this.chartData.allCorrect = 0;
        this.chartData.allTotal = 0;
        this.graphHistory = null;

        // Clear the chart canvases immediately
        ['accuracy-chart', 'graph-accuracy', 'graph-activity', 'graph-confidence',
         'graph-dead', 'graph-drift', 'graph-spread', 'graph-scaffold'].forEach(id => {
            const c = document.getElementById(id);
            if (c && c.getContext) {
                const ctx = c.getContext('2d');
                ctx.fillStyle = '#1a1a2e';
                ctx.fillRect(0, 0, c.width, c.height);
            }
        });

        // Reset display values
        document.getElementById('train-ep').textContent = '0';
        document.getElementById('train-acc').textContent = '0%';
        document.getElementById('train-eps').textContent = '0';
        document.getElementById('train-best').textContent = '0%';
        document.getElementById('train-last').innerHTML = '';
        document.getElementById('chart-episodes').textContent = '0';
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
            const episode = stats.episode || 0;

            // Track lifetime totals (must be before any display code)
            data.steps.forEach(s => {
                this.chartData.allTotal++;
                if (s.correct) this.chartData.allCorrect++;
            });
            const lifetimeAcc = this.chartData.allTotal > 0 ?
                this.chartData.allCorrect / this.chartData.allTotal : 0;

            // Record chart data (skip first 10 episodes)
            if (episode >= 10) {
                this.chartData.accuracy.push(acc50);
                this.chartData.epsilon.push(lastStep.epsilon || 0);

                const raw = this.chartData.accuracy;
                const smoothWin = Math.min(10, raw.length);
                const smoothed = raw.slice(-smoothWin).reduce((a, b) => a + b, 0) / smoothWin;
                this.chartData.smoothed.push(smoothed);

                if (this.chartData.accuracy.length > this.chartData.maxPoints) {
                    this.chartData.accuracy.shift();
                    this.chartData.epsilon.shift();
                    this.chartData.smoothed.shift();
                }
                this.drawChart();
            }

            // Smoothed accuracy for display
            const smoothArr = this.chartData.smoothed;
            const displayAcc = smoothArr.length > 0 ? smoothArr[smoothArr.length - 1] : acc50;

            // Update all display elements
            const accEl = document.getElementById('train-acc');
            accEl.textContent = (displayAcc * 100).toFixed(1) + '%';
            accEl.style.color = displayAcc > 0.5 ? 'var(--green)' : displayAcc > 0.25 ? 'var(--orange)' : 'var(--red)';

            document.getElementById('train-lifetime').textContent =
                (lifetimeAcc * 100).toFixed(1) + '%';
            document.getElementById('train-eps').textContent = lastStep.epsilon || 0;

            const curProgress = stats.curriculum || {};
            document.getElementById('train-level').textContent =
                curProgress.current_level || lastStep.level || '-';

            const lr = lastStep.lr || 0;
            document.getElementById('train-lr').textContent = lr.toFixed(4);
            document.getElementById('train-neurons').textContent =
                Object.keys(data.neuron_states).length;

            if (displayAcc > this.chartData.bestAccuracy) {
                this.chartData.bestAccuracy = displayAcc;
            }
            document.getElementById('train-best').textContent =
                (this.chartData.bestAccuracy * 100).toFixed(1) + '%';

            document.getElementById('stat-neurons').textContent =
                Object.keys(data.neuron_states).length;
            document.getElementById('chart-episodes').textContent = episode;

            // Show last few results
            const lastN = data.steps.slice(-5).reverse();
            document.getElementById('train-last').innerHTML = lastN.map(s =>
                `<span style="color:${s.correct ? 'var(--green)' : 'var(--red)'}">` +
                `${s.correct ? 'OK' : 'X'}</span>`
            ).join(' ');

            // Show interventions
            data.steps.forEach(s => {
                if (s.intervention) {
                    const info = s.intervention;
                    document.getElementById('train-last').innerHTML +=
                        `<div style="color:var(--orange);margin-top:2px">` +
                        `Homeostasis: pruned ${info.pruned}, spawned ${info.spawned}</div>`;
                }
            });

            // Pulse the training indicator
            const pulse = document.getElementById('train-pulse');
            pulse.style.opacity = pulse.style.opacity === '0.3' ? '1' : '0.3';
            document.getElementById('chart-episodes').textContent = episode;

            // Update graphs
            this.updateGraphs(data);

            // Refresh neuron detail if one is selected
            if (this.selectedNeuron !== null) {
                this.refreshSelectedNeuronLive(data.neuron_states);
            }

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

        // Update edges: brighten connections from active neurons
        const activeNeuronSet = new Set();
        Object.entries(states).forEach(([nid, state]) => {
            if (state.normalized > 0.3) activeNeuronSet.add(parseInt(nid));
        });

        this.edgeLines.forEach(eg => {
            if (!eg.edge || !eg.line) return;
            const srcActive = activeNeuronSet.has(eg.edge.source);
            const tgtActive = activeNeuronSet.has(eg.edge.target);

            if (srcActive && tgtActive) {
                // Both endpoints active — this edge is carrying signal
                eg.line.material.color.setHex(0x44aacc);
                eg.line.material.opacity = 0.5;
            } else if (srcActive || tgtActive) {
                // One endpoint active
                eg.line.material.color.setHex(eg.color);
                eg.line.material.opacity = eg.baseOpacity * 0.8;
            } else {
                // Neither active — dim
                eg.line.material.color.setHex(eg.color);
                eg.line.material.opacity = eg.baseOpacity * 0.3;
            }
        });
    }

    // --- Animation loop ---

    animate() {
        requestAnimationFrame(() => this.animate());
        this.updateCamera();

        const time = Date.now() * 0.001;

        if (!this.isReplaying && !this.isTraining) {
            // LIVE VIEW: gentle steady glow, no pulsing
            Object.values(this.neuronMeshes).forEach(m => {
                if (!m.userData.node.is_dead && m.userData.neuronId !== this.selectedNeuron) {
                    const act = m.userData.node.avg_activation || 0;
                    m.material.emissiveIntensity = 0.15 + act * 0.35;
                }
            });
        } else if (this.isReplaying) {
            // REPLAY VIEW: chosen output gently pulses
            Object.values(this.neuronMeshes).forEach(m => {
                if (m.scale.x > 1.3) {
                    m.material.emissiveIntensity = 0.5 + Math.sin(time * 1.5) * 0.15;
                }
            });
        }
        // Training view: updateNetworkLive handles everything, no extra animation

        // Animate dot trains on edges
        this.animateEdges(time);

        this.renderer.render(this.scene, this.camera);
    }
}

// Launch
const app = new HDNAViewer();
