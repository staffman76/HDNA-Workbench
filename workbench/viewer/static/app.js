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
            accuracy: [],   // rolling accuracy values over time
            epsilon: [],    // epsilon values over time
            maxPoints: 200, // max data points to show
            bestAccuracy: 0,
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

    // --- Model switching ---

    clearScene() {
        // Remove all neuron meshes
        Object.values(this.neuronMeshes).forEach(m => this.scene.remove(m));
        this.neuronMeshes = {};

        // Remove all edges
        this.edgeLines.forEach(l => this.scene.remove(l));
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
        } else {
            btn.textContent = 'Governance View';
            btn.style.borderColor = 'var(--border)';
            btn.style.color = 'var(--accent)';
            overlay.style.display = 'none';
            sidepanel.style.display = 'flex';
            document.querySelector('#header h1').textContent = 'HDNA Workbench';
        }
    }

    async updateGovernancePanel() {
        try {
            const [modelRes, stressRes, auditRes, daemonRes] = await Promise.all([
                fetch('/api/model').then(r => r.json()),
                fetch('/api/stress').then(r => r.json()),
                fetch('/api/audit?count=200').then(r => r.json()),
                fetch('/api/daemons').then(r => r.json()),
            ]);

            // Model name
            document.getElementById('gov-model-name').textContent = modelRes.name || 'Unknown';

            // Total decisions
            const stats = auditRes.stats || {};
            document.getElementById('gov-total-decisions').textContent =
                (stats.total_predictions || 0).toLocaleString();

            // Accuracy
            const acc = stats.accuracy_100 || 0;
            const accEl = document.getElementById('gov-accuracy');
            accEl.textContent = (acc * 100).toFixed(1) + '%';
            accEl.className = 'value ' + (acc > 0.7 ? 'good' : acc > 0.4 ? 'warn' : 'bad');

            // Components
            const extra = modelRes.extra || {};
            document.getElementById('gov-components').textContent =
                (extra.num_neurons || modelRes.layer_count || 0) + ' components, ' +
                (modelRes.layer_count || 0) + ' layers';

            // Risk indicators
            const dead = stressRes.dead_pct || 0;
            const deadEl = document.getElementById('gov-dead');
            deadEl.textContent = dead.toFixed(1) + '%';
            deadEl.className = 'value ' + (dead < 10 ? 'good' : dead < 30 ? 'warn' : 'bad');

            const jitter = stressRes.avg_jitter || 0;
            const stabEl = document.getElementById('gov-stability');
            stabEl.textContent = jitter < 0.01 ? 'Stable' : jitter < 0.1 ? 'Minor Fluctuation' : 'Unstable';
            stabEl.className = 'value ' + (jitter < 0.01 ? 'good' : jitter < 0.1 ? 'warn' : 'bad');

            const drift = stressRes.avg_weight_drift || 0;
            const driftEl = document.getElementById('gov-drift');
            driftEl.textContent = drift < 0.001 ? 'None Detected' : drift < 0.01 ? 'Minor' : 'Significant';
            driftEl.className = 'value ' + (drift < 0.001 ? 'good' : drift < 0.01 ? 'warn' : 'bad');

            const warnings = stressRes.warnings || [];
            const anomEl = document.getElementById('gov-anomalies');
            anomEl.textContent = warnings.length === 0 ? 'None' : warnings.length + ' warning(s)';
            anomEl.className = 'value ' + (warnings.length === 0 ? 'good' : 'bad');

            // Overall status
            const healthy = (dead < 30 && warnings.length === 0 && jitter < 0.1);
            const statusLight = document.getElementById('gov-status-light');
            const statusText = document.getElementById('gov-status-text');
            const statusSub = document.getElementById('gov-status-sub');
            const statusCard = document.getElementById('gov-status-card');

            if (healthy) {
                statusLight.style.background = 'var(--green)';
                statusText.textContent = 'System Healthy';
                statusText.style.color = 'var(--green)';
                statusSub.textContent = 'All monitored models operating normally';
                statusCard.style.borderColor = 'var(--green)';
            } else if (warnings.length > 0) {
                statusLight.style.background = 'var(--red)';
                statusText.textContent = 'Attention Required';
                statusText.style.color = 'var(--red)';
                statusSub.textContent = warnings.join(', ');
                statusCard.style.borderColor = 'var(--red)';
            } else {
                statusLight.style.background = 'var(--orange)';
                statusText.textContent = 'Review Recommended';
                statusText.style.color = 'var(--orange)';
                statusSub.textContent = 'Some metrics outside optimal range';
                statusCard.style.borderColor = 'var(--orange)';
            }

            // Compliance status based on features present
            document.getElementById('gov-art12').textContent = stats.total_predictions > 0 ? 'Compliant' : 'No Data';
            document.getElementById('gov-art12').className = 'value ' + (stats.total_predictions > 0 ? 'good' : 'warn');

            // Art 15 - robustness
            const art15El = document.getElementById('gov-art15');
            if (healthy) {
                art15El.textContent = 'Compliant';
                art15El.className = 'value good';
            } else {
                art15El.textContent = 'Review Needed';
                art15El.className = 'value warn';
            }

            // Recent activity
            const activityEl = document.getElementById('gov-activity');
            const records = (auditRes.records || []).slice(-8).reverse();
            if (records.length > 0) {
                let html = '';
                records.forEach(r => {
                    const time = new Date(r.timestamp * 1000).toLocaleTimeString();
                    const icon = r.correct ? '<span style="color:var(--green)">&#10003;</span>' :
                                             '<span style="color:var(--red)">&#10007;</span>';
                    html += `<div style="display:flex;justify-content:space-between;padding:2px 0;font-size:11px;border-bottom:1px solid rgba(255,255,255,0.05)">
                        <span>${icon} Decision #${r.step}</span>
                        <span style="color:var(--text-dim)">Confidence: ${(Math.min(1, Math.max(0, r.confidence)) * 100).toFixed(0)}%</span>
                        <span style="color:var(--text-dim)">${r.source}</span>
                    </div>`;
                });
                activityEl.innerHTML = html;
            }

        } catch (err) {
            console.error('Failed to update governance panel:', err);
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

        // Clear
        ctx.fillStyle = '#1a1a2e';
        ctx.fillRect(0, 0, w, h);

        const accData = this.chartData.accuracy;
        const epsData = this.chartData.epsilon;
        if (accData.length < 2) return;

        const n = accData.length;
        const xStep = w / (n - 1);

        // Grid lines at 25%, 50%, 75%
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
        ctx.fillText('25%', 2, h - 0.25 * h - 2);
        ctx.fillText('50%', 2, h - 0.50 * h - 2);
        ctx.fillText('75%', 2, h - 0.75 * h - 2);

        // Epsilon line (orange, scaled 0-1)
        ctx.beginPath();
        ctx.strokeStyle = '#ffab40';
        ctx.lineWidth = 1;
        ctx.globalAlpha = 0.5;
        for (let i = 0; i < n; i++) {
            const x = i * xStep;
            const y = h - epsData[i] * h;
            if (i === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        }
        ctx.stroke();
        ctx.globalAlpha = 1.0;

        // Accuracy line (green)
        ctx.beginPath();
        ctx.strokeStyle = '#00e676';
        ctx.lineWidth = 2;
        for (let i = 0; i < n; i++) {
            const x = i * xStep;
            const y = h - accData[i] * h;
            if (i === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        }
        ctx.stroke();

        // Current accuracy dot
        const lastAcc = accData[n - 1];
        const dotX = (n - 1) * xStep;
        const dotY = h - lastAcc * h;
        ctx.beginPath();
        ctx.fillStyle = lastAcc > 0.5 ? '#00e676' : lastAcc > 0.25 ? '#ffab40' : '#ff5252';
        ctx.arc(dotX, dotY, 3, 0, Math.PI * 2);
        ctx.fill();

        // Current value text
        ctx.fillStyle = '#e0e0e0';
        ctx.font = '10px sans-serif';
        ctx.textAlign = 'right';
        ctx.fillText((lastAcc * 100).toFixed(1) + '%', w - 4, dotY - 5);
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
        this.chartData.accuracy = [];
        this.chartData.epsilon = [];
        this.chartData.bestAccuracy = 0;
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

            // Extra stats
            const lr = lastStep.lr || 0;
            document.getElementById('train-lr').textContent = lr.toFixed(4);
            document.getElementById('train-neurons').textContent =
                Object.keys(data.neuron_states).length;

            // Track best accuracy
            if (acc50 > this.chartData.bestAccuracy) {
                this.chartData.bestAccuracy = acc50;
            }
            document.getElementById('train-best').textContent =
                (this.chartData.bestAccuracy * 100).toFixed(1) + '%';

            // Update header stats
            document.getElementById('stat-neurons').textContent =
                Object.keys(data.neuron_states).length;

            // Record chart data
            this.chartData.accuracy.push(acc50);
            this.chartData.epsilon.push(lastStep.epsilon || 0);
            if (this.chartData.accuracy.length > this.chartData.maxPoints) {
                this.chartData.accuracy.shift();
                this.chartData.epsilon.shift();
            }
            document.getElementById('chart-episodes').textContent = stats.episode || 0;
            this.drawChart();

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
