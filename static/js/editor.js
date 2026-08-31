/**
 * PFD Visual Editor
 * Canvas-based process flow diagram editor
 */

// ============================================================================
// State Management
// ============================================================================

const state = {
    pfd: {
        metadata: {
            process_name: 'New Process',
            version: '1.0',
            description: '',
            author: '',
            date: '',
            thermo_method: 'IDEAL'
        },
        components: [],
        streams: [],
        units: []
    },
    filename: 'untitled.pfd',
    selection: null, // { type: 'unit'|'stream', id: string }
    dragging: null,  // { unitId: string, offsetX: number, offsetY: number }
    connecting: null, // { unitId: string, portId: string, startX: number, startY: number }
    pan: { x: 0, y: 0 },
    zoom: 1,
    unitTemplates: {},
    isDirty: false
};

// ============================================================================
// Canvas Setup
// ============================================================================

const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const container = document.getElementById('canvas-container');

function resizeCanvas() {
    canvas.width = container.clientWidth;
    canvas.height = container.clientHeight;
    render();
}

window.addEventListener('resize', resizeCanvas);
resizeCanvas();

// ============================================================================
// Unit Type Colors and Icons
// ============================================================================

const unitColors = {
    'Mixer': '#3b82f6',
    'Splitter': '#8b5cf6',
    'Pump': '#8b5cf6',
    'Compressor': '#8b5cf6',
    'Valve': '#6b7280',
    'Pipe': '#6b7280',
    'Heater': '#ef4444',
    'Cooler': '#22d3ee',
    'HeatExchanger': '#f59e0b',
    'Flash': '#10b981',
    'Distillation': '#10b981',
    'Reactor': '#ef4444',
    'EquilibriumReactor': '#ef4444',
    'PFR': '#ef4444',
    'CSTR': '#ef4444'
};

const unitIcons = {
    'Mixer': 'M',
    'Splitter': 'S',
    'Pump': 'P',
    'Compressor': 'C',
    'Valve': 'V',
    'Pipe': 'PI',
    'Heater': 'H',
    'Cooler': 'CL',
    'HeatExchanger': 'HX',
    'Flash': 'F',
    'Distillation': 'D',
    'Reactor': 'R',
    'EquilibriumReactor': 'ER',
    'PFR': 'PFR',
    'CSTR': 'CSTR'
};

// ============================================================================
// Rendering
// ============================================================================

function render() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    ctx.save();
    ctx.translate(state.pan.x, state.pan.y);
    ctx.scale(state.zoom, state.zoom);
    
    // Draw streams first (behind units)
    for (const stream of state.pfd.streams) {
        drawStream(stream);
    }
    
    // Draw connecting line if in progress
    if (state.connecting) {
        ctx.beginPath();
        ctx.strokeStyle = '#4a6cf7';
        ctx.lineWidth = 2;
        ctx.setLineDash([5, 5]);
        ctx.moveTo(state.connecting.startX, state.connecting.startY);
        ctx.lineTo(state.connecting.currentX, state.connecting.currentY);
        ctx.stroke();
        ctx.setLineDash([]);
    }
    
    // Draw FEED/PRODUCT nodes
    drawBoundaryNodes();
    
    // Draw units
    for (const unit of state.pfd.units) {
        drawUnit(unit);
    }
    
    ctx.restore();
}

function drawUnit(unit) {
    const x = unit.x || 0;
    const y = unit.y || 0;
    const width = 120;
    const height = 80;
    const color = unitColors[unit.unit_type] || '#6b7280';
    const isSelected = state.selection?.type === 'unit' && state.selection?.id === unit.id;
    
    // Shadow
    ctx.fillStyle = 'rgba(0, 0, 0, 0.3)';
    ctx.beginPath();
    ctx.roundRect(x + 3, y + 3, width, height, 8);
    ctx.fill();
    
    // Background
    ctx.fillStyle = isSelected ? '#2a2a3a' : '#1a1a25';
    ctx.beginPath();
    ctx.roundRect(x, y, width, height, 8);
    ctx.fill();
    
    // Border
    ctx.strokeStyle = isSelected ? '#4a6cf7' : color;
    ctx.lineWidth = isSelected ? 2 : 1;
    ctx.stroke();
    
    // Type indicator bar
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.roundRect(x, y, width, 24, [8, 8, 0, 0]);
    ctx.fill();
    
    // Unit type text
    ctx.fillStyle = '#ffffff';
    ctx.font = '11px "Plus Jakarta Sans", sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(unit.unit_type || 'Unit', x + width / 2, y + 12);
    
    // Unit ID
    ctx.fillStyle = '#e8e8ef';
    ctx.font = 'bold 14px "Plus Jakarta Sans", sans-serif';
    ctx.fillText(unit.id, x + width / 2, y + 50);
    
    // Draw ports
    drawPorts(unit, x, y, width, height);
}

function drawPorts(unit, x, y, width, height) {
    const ports = unit.ports || [];
    const inlets = ports.filter(p => p.port_type === 'inlet');
    const outlets = ports.filter(p => p.port_type !== 'inlet');
    
    // Left side (inlets)
    const inletSpacing = height / (inlets.length + 1);
    inlets.forEach((port, i) => {
        const py = y + inletSpacing * (i + 1);
        drawPort(x, py, port, 'inlet');
    });
    
    // Right side (outlets)
    const outletSpacing = height / (outlets.length + 1);
    outlets.forEach((port, i) => {
        const py = y + outletSpacing * (i + 1);
        drawPort(x + width, py, port, 'outlet');
    });
}

function drawPort(x, y, port, side) {
    const radius = 6;
    
    ctx.beginPath();
    ctx.arc(x, y, radius, 0, Math.PI * 2);
    ctx.fillStyle = '#12121a';
    ctx.fill();
    ctx.strokeStyle = side === 'inlet' ? '#22d3ee' : '#10b981';
    ctx.lineWidth = 2;
    ctx.stroke();
    
    // Port label
    ctx.fillStyle = '#9898a8';
    ctx.font = '9px "JetBrains Mono", monospace';
    ctx.textAlign = side === 'inlet' ? 'right' : 'left';
    ctx.textBaseline = 'middle';
    const labelX = side === 'inlet' ? x - 10 : x + 10;
    ctx.fillText(port.id, labelX, y);
}

function drawStream(stream) {
    const sourcePos = getPortPosition(stream.source);
    const destPos = getPortPosition(stream.destination);
    
    if (!sourcePos || !destPos) return;
    
    const isSelected = state.selection?.type === 'stream' && state.selection?.id === stream.id;
    
    // Draw curved path
    ctx.beginPath();
    ctx.strokeStyle = isSelected ? '#4a6cf7' : '#5a5a6a';
    ctx.lineWidth = isSelected ? 3 : 2;
    
    const midX = (sourcePos.x + destPos.x) / 2;
    
    ctx.moveTo(sourcePos.x, sourcePos.y);
    ctx.bezierCurveTo(
        midX, sourcePos.y,
        midX, destPos.y,
        destPos.x, destPos.y
    );
    ctx.stroke();
    
    // Draw arrow at destination
    drawArrow(destPos.x, destPos.y, destPos.x - 20, destPos.y, isSelected);
    
    // Stream label
    const labelX = midX;
    const labelY = (sourcePos.y + destPos.y) / 2 - 10;
    
    ctx.fillStyle = isSelected ? '#4a6cf7' : '#9898a8';
    ctx.font = '11px "JetBrains Mono", monospace';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(stream.id, labelX, labelY);
}

function drawArrow(toX, toY, fromX, fromY, isSelected) {
    const headLength = 10;
    const angle = Math.atan2(toY - fromY, toX - fromX);
    
    ctx.beginPath();
    ctx.fillStyle = isSelected ? '#4a6cf7' : '#5a5a6a';
    ctx.moveTo(toX, toY);
    ctx.lineTo(
        toX - headLength * Math.cos(angle - Math.PI / 6),
        toY - headLength * Math.sin(angle - Math.PI / 6)
    );
    ctx.lineTo(
        toX - headLength * Math.cos(angle + Math.PI / 6),
        toY - headLength * Math.sin(angle + Math.PI / 6)
    );
    ctx.closePath();
    ctx.fill();
}

function drawBoundaryNodes() {
    // Find all FEED and PRODUCT connections
    const feeds = [];
    const products = [];
    
    for (const stream of state.pfd.streams) {
        if (stream.source.is_feed) {
            const destPos = getPortPosition(stream.destination);
            if (destPos) {
                feeds.push({ stream, x: destPos.x - 80, y: destPos.y });
            }
        }
        if (stream.destination.is_product) {
            const srcPos = getPortPosition(stream.source);
            if (srcPos) {
                products.push({ stream, x: srcPos.x + 80, y: srcPos.y });
            }
        }
    }
    
    // Draw FEED nodes
    for (const feed of feeds) {
        ctx.fillStyle = '#22d3ee';
        ctx.beginPath();
        ctx.roundRect(feed.x - 30, feed.y - 12, 60, 24, 4);
        ctx.fill();
        
        ctx.fillStyle = '#000';
        ctx.font = 'bold 11px "Plus Jakarta Sans", sans-serif';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText('FEED', feed.x, feed.y);
    }
    
    // Draw PRODUCT nodes
    for (const prod of products) {
        ctx.fillStyle = '#10b981';
        ctx.beginPath();
        ctx.roundRect(prod.x - 35, prod.y - 12, 70, 24, 4);
        ctx.fill();
        
        ctx.fillStyle = '#000';
        ctx.font = 'bold 11px "Plus Jakarta Sans", sans-serif';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText('PRODUCT', prod.x, prod.y);
    }
}

function getPortPosition(portRef) {
    if (portRef.is_feed) {
        // Find where this feed connects to
        return null; // Handled separately
    }
    
    if (portRef.is_product) {
        return null; // Handled separately
    }
    
    const unit = state.pfd.units.find(u => u.id === portRef.unit_id);
    if (!unit) return null;
    
    const port = unit.ports?.find(p => p.id === portRef.port_id);
    if (!port) return null;
    
    const x = unit.x || 0;
    const y = unit.y || 0;
    const width = 120;
    const height = 80;
    
    const inlets = unit.ports.filter(p => p.port_type === 'inlet');
    const outlets = unit.ports.filter(p => p.port_type !== 'inlet');
    
    if (port.port_type === 'inlet') {
        const idx = inlets.indexOf(port);
        const spacing = height / (inlets.length + 1);
        return { x: x, y: y + spacing * (idx + 1) };
    } else {
        const idx = outlets.indexOf(port);
        const spacing = height / (outlets.length + 1);
        return { x: x + width, y: y + spacing * (idx + 1) };
    }
}

// ============================================================================
// Interaction Handling
// ============================================================================

canvas.addEventListener('mousedown', handleMouseDown);
canvas.addEventListener('mousemove', handleMouseMove);
canvas.addEventListener('mouseup', handleMouseUp);
canvas.addEventListener('wheel', handleWheel);
canvas.addEventListener('dblclick', handleDoubleClick);

function screenToWorld(screenX, screenY) {
    return {
        x: (screenX - state.pan.x) / state.zoom,
        y: (screenY - state.pan.y) / state.zoom
    };
}

function handleMouseDown(e) {
    const rect = canvas.getBoundingClientRect();
    const screenX = e.clientX - rect.left;
    const screenY = e.clientY - rect.top;
    const world = screenToWorld(screenX, screenY);
    
    // Check for unit click
    const unit = findUnitAtPosition(world.x, world.y);
    
    if (unit) {
        selectUnit(unit.id);
        state.dragging = {
            unitId: unit.id,
            offsetX: world.x - (unit.x || 0),
            offsetY: world.y - (unit.y || 0)
        };
        return;
    }
    
    // Check for stream click
    const stream = findStreamAtPosition(world.x, world.y);
    if (stream) {
        selectStream(stream.id);
        return;
    }
    
    // Start panning
    state.dragging = {
        panning: true,
        startX: screenX,
        startY: screenY,
        startPanX: state.pan.x,
        startPanY: state.pan.y
    };
    
    clearSelection();
}

function handleMouseMove(e) {
    const rect = canvas.getBoundingClientRect();
    const screenX = e.clientX - rect.left;
    const screenY = e.clientY - rect.top;
    const world = screenToWorld(screenX, screenY);
    
    if (state.dragging) {
        if (state.dragging.panning) {
            state.pan.x = state.dragging.startPanX + (screenX - state.dragging.startX);
            state.pan.y = state.dragging.startPanY + (screenY - state.dragging.startY);
        } else if (state.dragging.unitId) {
            const unit = state.pfd.units.find(u => u.id === state.dragging.unitId);
            if (unit) {
                unit.x = Math.round((world.x - state.dragging.offsetX) / 10) * 10;
                unit.y = Math.round((world.y - state.dragging.offsetY) / 10) * 10;
                state.isDirty = true;
            }
        }
        render();
    }
    
    if (state.connecting) {
        state.connecting.currentX = world.x;
        state.connecting.currentY = world.y;
        render();
    }
}

function handleMouseUp(e) {
    state.dragging = null;
    state.connecting = null;
}

function handleWheel(e) {
    e.preventDefault();
    
    const rect = canvas.getBoundingClientRect();
    const mouseX = e.clientX - rect.left;
    const mouseY = e.clientY - rect.top;
    
    const zoomFactor = e.deltaY > 0 ? 0.9 : 1.1;
    const newZoom = Math.min(Math.max(state.zoom * zoomFactor, 0.25), 4);
    
    // Zoom toward mouse position
    state.pan.x = mouseX - (mouseX - state.pan.x) * (newZoom / state.zoom);
    state.pan.y = mouseY - (mouseY - state.pan.y) * (newZoom / state.zoom);
    state.zoom = newZoom;
    
    document.getElementById('zoom-level').textContent = Math.round(state.zoom * 100) + '%';
    render();
}

function handleDoubleClick(e) {
    const rect = canvas.getBoundingClientRect();
    const world = screenToWorld(e.clientX - rect.left, e.clientY - rect.top);
    
    const unit = findUnitAtPosition(world.x, world.y);
    if (unit) {
        // Focus on unit ID input
        document.getElementById('unit-id-input').focus();
        document.getElementById('unit-id-input').select();
    }
}

function findUnitAtPosition(x, y) {
    for (const unit of state.pfd.units) {
        const ux = unit.x || 0;
        const uy = unit.y || 0;
        if (x >= ux && x <= ux + 120 && y >= uy && y <= uy + 80) {
            return unit;
        }
    }
    return null;
}

function findStreamAtPosition(x, y) {
    for (const stream of state.pfd.streams) {
        const sourcePos = getPortPosition(stream.source);
        const destPos = getPortPosition(stream.destination);
        
        if (!sourcePos || !destPos) continue;
        
        // Simple distance check to bezier curve
        const midX = (sourcePos.x + destPos.x) / 2;
        const midY = (sourcePos.y + destPos.y) / 2;
        
        const dist = Math.sqrt((x - midX) ** 2 + (y - midY) ** 2);
        if (dist < 30) return stream;
    }
    return null;
}

// ============================================================================
// Selection Management
// ============================================================================

function selectUnit(unitId) {
    state.selection = { type: 'unit', id: unitId };
    updatePropertiesPanel();
    render();
}

function selectStream(streamId) {
    state.selection = { type: 'stream', id: streamId };
    updatePropertiesPanel();
    render();
}

function clearSelection() {
    state.selection = null;
    updatePropertiesPanel();
    render();
}

function updatePropertiesPanel() {
    const noSelection = document.getElementById('no-selection');
    const unitProps = document.getElementById('unit-properties');
    const streamProps = document.getElementById('stream-properties');
    
    noSelection.style.display = 'none';
    unitProps.style.display = 'none';
    streamProps.style.display = 'none';
    
    if (!state.selection) {
        noSelection.style.display = 'flex';
        return;
    }
    
    if (state.selection.type === 'unit') {
        const unit = state.pfd.units.find(u => u.id === state.selection.id);
        if (unit) {
            unitProps.style.display = 'block';
            document.getElementById('selected-unit-name').textContent = unit.id;
            document.getElementById('selected-unit-type').textContent = unit.unit_type;
            document.getElementById('unit-id-input').value = unit.id;
            
            // Show reactions section for reactor types
            const isReactor = ['Reactor', 'EquilibriumReactor', 'PFR', 'CSTR'].includes(unit.unit_type);
            document.getElementById('reactions-section').style.display = isReactor ? 'block' : 'none';
            
            renderUnitParams(unit);
            renderUnitReactions(unit);
        }
    } else if (state.selection.type === 'stream') {
        const stream = state.pfd.streams.find(s => s.id === state.selection.id);
        if (stream) {
            streamProps.style.display = 'block';
            document.getElementById('selected-stream-name').textContent = stream.id;
            document.getElementById('selected-stream-connection').textContent = 
                `${stream.source.is_feed ? 'FEED' : stream.source.unit_id + '.' + stream.source.port_id} → ${stream.destination.is_product ? 'PRODUCT' : stream.destination.unit_id + '.' + stream.destination.port_id}`;
            document.getElementById('stream-id-input').value = stream.id;
            
            renderStreamProps(stream);
        }
    }
}

function renderUnitParams(unit) {
    const container = document.getElementById('unit-params-list');
    container.innerHTML = '';
    
    for (let i = 0; i < (unit.params || []).length; i++) {
        const param = unit.params[i];
        const div = document.createElement('div');
        div.className = 'property-item';
        div.innerHTML = `
            <input type="text" class="form-input" value="${param.name}" 
                   onchange="updateUnitParam(${i}, 'name', this.value)" placeholder="Name">
            <input type="text" class="form-input" value="${param.value}" 
                   onchange="updateUnitParam(${i}, 'value', this.value)" placeholder="Value">
            <button class="btn btn-icon btn-danger" onclick="removeUnitParam(${i})">
                <svg viewBox="0 0 24 24" stroke="currentColor" fill="none" stroke-width="2">
                    <line x1="18" y1="6" x2="6" y2="18"/>
                    <line x1="6" y1="6" x2="18" y2="18"/>
                </svg>
            </button>
        `;
        container.appendChild(div);
    }
}

function renderUnitReactions(unit) {
    const container = document.getElementById('unit-reactions-list');
    container.innerHTML = '';
    
    for (let i = 0; i < (unit.reactions || []).length; i++) {
        const rxn = unit.reactions[i];
        const div = document.createElement('div');
        div.className = 'property-item';
        div.style.gridTemplateColumns = '1fr auto';
        div.innerHTML = `
            <input type="text" class="form-input" value="${rxn.equation}" 
                   onchange="updateUnitReaction(${i}, this.value)" placeholder="A + B -> C">
            <button class="btn btn-icon btn-danger" onclick="removeUnitReaction(${i})">
                <svg viewBox="0 0 24 24" stroke="currentColor" fill="none" stroke-width="2">
                    <line x1="18" y1="6" x2="6" y2="18"/>
                    <line x1="6" y1="6" x2="18" y2="18"/>
                </svg>
            </button>
        `;
        container.appendChild(div);
    }
}

function renderStreamProps(stream) {
    const container = document.getElementById('stream-props-list');
    container.innerHTML = '';
    
    for (let i = 0; i < (stream.properties || []).length; i++) {
        const prop = stream.properties[i];
        const div = document.createElement('div');
        div.className = 'property-item';
        div.innerHTML = `
            <input type="text" class="form-input" value="${prop.name}" 
                   onchange="updateStreamProp(${i}, 'name', this.value)" placeholder="Name">
            <input type="text" class="form-input" value="${prop.value}" 
                   onchange="updateStreamProp(${i}, 'value', this.value)" placeholder="Value">
            <button class="btn btn-icon btn-danger" onclick="removeStreamProp(${i})">
                <svg viewBox="0 0 24 24" stroke="currentColor" fill="none" stroke-width="2">
                    <line x1="18" y1="6" x2="6" y2="18"/>
                    <line x1="6" y1="6" x2="18" y2="18"/>
                </svg>
            </button>
        `;
        container.appendChild(div);
    }
}

// ============================================================================
// Property Editing
// ============================================================================

function updateUnitParam(index, field, value) {
    if (!state.selection || state.selection.type !== 'unit') return;
    const unit = state.pfd.units.find(u => u.id === state.selection.id);
    if (unit && unit.params && unit.params[index]) {
        unit.params[index][field] = value;
        state.isDirty = true;
        syncToCodeEditor();
    }
}

function removeUnitParam(index) {
    if (!state.selection || state.selection.type !== 'unit') return;
    const unit = state.pfd.units.find(u => u.id === state.selection.id);
    if (unit && unit.params) {
        unit.params.splice(index, 1);
        state.isDirty = true;
        renderUnitParams(unit);
        syncToCodeEditor();
    }
}

function addParameter() {
    if (!state.selection || state.selection.type !== 'unit') return;
    const unit = state.pfd.units.find(u => u.id === state.selection.id);
    if (unit) {
        if (!unit.params) unit.params = [];
        unit.params.push({ name: '', value: '', unit: null });
        state.isDirty = true;
        renderUnitParams(unit);
        syncToCodeEditor();
    }
}

function updateUnitReaction(index, value) {
    if (!state.selection || state.selection.type !== 'unit') return;
    const unit = state.pfd.units.find(u => u.id === state.selection.id);
    if (unit && unit.reactions && unit.reactions[index]) {
        unit.reactions[index].equation = value;
        state.isDirty = true;
        syncToCodeEditor();
    }
}

function removeUnitReaction(index) {
    if (!state.selection || state.selection.type !== 'unit') return;
    const unit = state.pfd.units.find(u => u.id === state.selection.id);
    if (unit && unit.reactions) {
        unit.reactions.splice(index, 1);
        state.isDirty = true;
        renderUnitReactions(unit);
        syncToCodeEditor();
    }
}

function addReaction() {
    if (!state.selection || state.selection.type !== 'unit') return;
    const unit = state.pfd.units.find(u => u.id === state.selection.id);
    if (unit) {
        if (!unit.reactions) unit.reactions = [];
        unit.reactions.push({ equation: '', parameters: {} });
        state.isDirty = true;
        renderUnitReactions(unit);
        syncToCodeEditor();
    }
}

function updateStreamProp(index, field, value) {
    if (!state.selection || state.selection.type !== 'stream') return;
    const stream = state.pfd.streams.find(s => s.id === state.selection.id);
    if (stream && stream.properties && stream.properties[index]) {
        stream.properties[index][field] = value;
        state.isDirty = true;
        syncToCodeEditor();
    }
}

function removeStreamProp(index) {
    if (!state.selection || state.selection.type !== 'stream') return;
    const stream = state.pfd.streams.find(s => s.id === state.selection.id);
    if (stream && stream.properties) {
        stream.properties.splice(index, 1);
        state.isDirty = true;
        renderStreamProps(stream);
        syncToCodeEditor();
    }
}

function addStreamProperty() {
    if (!state.selection || state.selection.type !== 'stream') return;
    const stream = state.pfd.streams.find(s => s.id === state.selection.id);
    if (stream) {
        if (!stream.properties) stream.properties = [];
        stream.properties.push({ name: '', value: '', unit: null });
        state.isDirty = true;
        renderStreamProps(stream);
        syncToCodeEditor();
    }
}

function deleteSelected() {
    if (!state.selection) return;
    
    if (state.selection.type === 'unit') {
        const idx = state.pfd.units.findIndex(u => u.id === state.selection.id);
        if (idx !== -1) {
            // Remove connected streams
            state.pfd.streams = state.pfd.streams.filter(s => 
                s.source.unit_id !== state.selection.id && 
                s.destination.unit_id !== state.selection.id
            );
            state.pfd.units.splice(idx, 1);
        }
    } else if (state.selection.type === 'stream') {
        const idx = state.pfd.streams.findIndex(s => s.id === state.selection.id);
        if (idx !== -1) {
            state.pfd.streams.splice(idx, 1);
        }
    }
    
    state.isDirty = true;
    clearSelection();
    syncToCodeEditor();
}

// Handle ID changes
document.getElementById('unit-id-input').addEventListener('change', function() {
    if (!state.selection || state.selection.type !== 'unit') return;
    const unit = state.pfd.units.find(u => u.id === state.selection.id);
    if (unit) {
        const oldId = unit.id;
        const newId = this.value;
        
        // Update stream references
        for (const stream of state.pfd.streams) {
            if (stream.source.unit_id === oldId) stream.source.unit_id = newId;
            if (stream.destination.unit_id === oldId) stream.destination.unit_id = newId;
        }
        
        unit.id = newId;
        state.selection.id = newId;
        state.isDirty = true;
        
        document.getElementById('selected-unit-name').textContent = newId;
        render();
        syncToCodeEditor();
    }
});

document.getElementById('stream-id-input').addEventListener('change', function() {
    if (!state.selection || state.selection.type !== 'stream') return;
    const stream = state.pfd.streams.find(s => s.id === state.selection.id);
    if (stream) {
        stream.id = this.value;
        state.selection.id = this.value;
        state.isDirty = true;
        
        document.getElementById('selected-stream-name').textContent = this.value;
        render();
        syncToCodeEditor();
    }
});

// ============================================================================
// Unit Palette
// ============================================================================

const paletteUnits = [
    { type: 'Mixer', icon: '⊕' },
    { type: 'Splitter', icon: '⊖' },
    { type: 'Pump', icon: '⟳' },
    { type: 'Compressor', icon: '⟲' },
    { type: 'Heater', icon: '🔥' },
    { type: 'Cooler', icon: '❄' },
    { type: 'HeatExchanger', icon: '⇌' },
    { type: 'Flash', icon: '◇' },
    { type: 'Reactor', icon: '⚗' },
    { type: 'Distillation', icon: '▭' }
];

function initPalette() {
    const palette = document.getElementById('unit-palette');
    palette.innerHTML = '';
    
    for (const item of paletteUnits) {
        const div = document.createElement('div');
        div.className = 'palette-item';
        div.draggable = true;
        div.dataset.type = item.type;
        
        const color = unitColors[item.type] || '#6b7280';
        
        div.innerHTML = `
            <div class="palette-icon" style="background: ${color};">
                <span style="color: white; font-size: 16px;">${unitIcons[item.type] || 'U'}</span>
            </div>
            <div class="palette-label">${item.type}</div>
        `;
        
        div.addEventListener('dragstart', handleDragStart);
        div.addEventListener('click', () => addUnitFromPalette(item.type));
        
        palette.appendChild(div);
    }
}

function handleDragStart(e) {
    e.dataTransfer.setData('text/plain', e.target.dataset.type);
}

canvas.addEventListener('dragover', (e) => {
    e.preventDefault();
});

canvas.addEventListener('drop', (e) => {
    e.preventDefault();
    const type = e.dataTransfer.getData('text/plain');
    if (type) {
        const rect = canvas.getBoundingClientRect();
        const world = screenToWorld(e.clientX - rect.left, e.clientY - rect.top);
        addUnit(type, world.x - 60, world.y - 40);
    }
});

function addUnitFromPalette(type) {
    // Add at center of visible canvas
    const centerX = (canvas.width / 2 - state.pan.x) / state.zoom;
    const centerY = (canvas.height / 2 - state.pan.y) / state.zoom;
    addUnit(type, centerX - 60, centerY - 40);
}

async function addUnit(type, x, y) {
    // Get template for this unit type
    let template = state.unitTemplates[type];
    
    if (!template) {
        try {
            const response = await fetch('/api/unit-templates');
            const templates = await response.json();
            state.unitTemplates = templates;
            template = templates[type];
        } catch (e) {
            console.error('Failed to load unit templates:', e);
        }
    }
    
    // Generate unique ID
    const existingIds = state.pfd.units.map(u => u.id);
    let id = type.substring(0, 3).toUpperCase() + '-1';
    let counter = 1;
    while (existingIds.includes(id)) {
        counter++;
        id = type.substring(0, 3).toUpperCase() + '-' + counter;
    }
    
    const unit = {
        id: id,
        unit_type: type,
        x: Math.round(x / 10) * 10,
        y: Math.round(y / 10) * 10,
        ports: template?.ports || [
            { id: 'in', port_type: 'inlet' },
            { id: 'out', port_type: 'outlet' }
        ],
        params: template?.default_params || [],
        reactions: []
    };
    
    state.pfd.units.push(unit);
    state.isDirty = true;
    selectUnit(unit.id);
    syncToCodeEditor();
}

// ============================================================================
// Code Editor Sync
// ============================================================================

const codeEditor = document.getElementById('code-editor');
let codeEditorTimeout = null;

function syncToCodeEditor() {
    fetch('/api/serialize', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ pfd: state.pfd })
    })
    .then(res => res.json())
    .then(data => {
        if (data.success) {
            codeEditor.value = data.text;
            updateCodeStatus('success', 'Synced');
        }
    })
    .catch(err => {
        updateCodeStatus('error', 'Sync failed');
    });
}

function syncFromCodeEditor() {
    const text = codeEditor.value;
    
    fetch('/api/parse', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: text })
    })
    .then(res => res.json())
    .then(data => {
        if (data.success) {
            state.pfd = data.pfd;
            state.isDirty = true;
            clearSelection();
            updateCodeStatus('success', `${data.errors.length} errors, ${data.warnings.length} warnings`);
        } else {
            updateCodeStatus('error', data.error);
        }
    })
    .catch(err => {
        updateCodeStatus('error', err.message);
    });
}

codeEditor.addEventListener('input', () => {
    clearTimeout(codeEditorTimeout);
    updateCodeStatus('warning', 'Editing...');
    codeEditorTimeout = setTimeout(syncFromCodeEditor, 1000);
});

function updateCodeStatus(status, text) {
    const dot = document.getElementById('code-status-dot');
    const textEl = document.getElementById('code-status-text');
    
    dot.className = 'status-dot ' + status;
    textEl.textContent = text;
}

// ============================================================================
// Tab Switching
// ============================================================================

function switchTab(tab) {
    document.querySelectorAll('.panel-tab').forEach(t => t.classList.remove('active'));
    document.querySelector(`.panel-tab[data-tab="${tab}"]`).classList.add('active');
    
    document.getElementById('visual-panel').style.display = tab === 'visual' ? 'block' : 'none';
    document.getElementById('code-panel').classList.toggle('active', tab === 'code');
    
    if (tab === 'code') {
        syncToCodeEditor();
    }
}

// ============================================================================
// Zoom Controls
// ============================================================================

function zoomIn() {
    state.zoom = Math.min(state.zoom * 1.2, 4);
    document.getElementById('zoom-level').textContent = Math.round(state.zoom * 100) + '%';
    render();
}

function zoomOut() {
    state.zoom = Math.max(state.zoom / 1.2, 0.25);
    document.getElementById('zoom-level').textContent = Math.round(state.zoom * 100) + '%';
    render();
}

function resetView() {
    state.pan = { x: 0, y: 0 };
    state.zoom = 1;
    document.getElementById('zoom-level').textContent = '100%';
    render();
}

function fitToScreen() {
    if (state.pfd.units.length === 0) {
        resetView();
        return;
    }
    
    let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
    
    for (const unit of state.pfd.units) {
        minX = Math.min(minX, unit.x || 0);
        minY = Math.min(minY, unit.y || 0);
        maxX = Math.max(maxX, (unit.x || 0) + 120);
        maxY = Math.max(maxY, (unit.y || 0) + 80);
    }
    
    const contentWidth = maxX - minX + 200;
    const contentHeight = maxY - minY + 200;
    
    const scaleX = canvas.width / contentWidth;
    const scaleY = canvas.height / contentHeight;
    state.zoom = Math.min(scaleX, scaleY, 1.5);
    
    state.pan.x = (canvas.width - contentWidth * state.zoom) / 2 - minX * state.zoom + 100 * state.zoom;
    state.pan.y = (canvas.height - contentHeight * state.zoom) / 2 - minY * state.zoom + 100 * state.zoom;
    
    document.getElementById('zoom-level').textContent = Math.round(state.zoom * 100) + '%';
    render();
}

// ============================================================================
// File Operations
// ============================================================================

function openFile() {
    document.getElementById('file-input').click();
}

async function handleFileUpload(event) {
    const file = event.target.files[0];
    if (!file) return;
    
    const formData = new FormData();
    formData.append('file', file);
    
    try {
        const response = await fetch('/api/upload', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (data.success) {
            state.pfd = data.pfd;
            state.filename = data.filename;
            document.getElementById('current-filename').textContent = data.filename;
            codeEditor.value = data.text;
            clearSelection();
            fitToScreen();
            showToast('success', `Loaded ${data.filename}`);
            
            if (data.errors.length > 0) {
                showValidationResults(data.errors, data.warnings);
            }
        } else {
            showToast('error', data.error);
        }
    } catch (err) {
        showToast('error', 'Failed to upload file');
    }
    
    event.target.value = '';
}

async function downloadFile() {
    try {
        const response = await fetch('/api/download', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                pfd: state.pfd,
                filename: state.filename
            })
        });
        
        const blob = await response.blob();
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = state.filename;
        a.click();
        URL.revokeObjectURL(url);
        
        showToast('success', `Downloaded ${state.filename}`);
    } catch (err) {
        showToast('error', 'Failed to download file');
    }
}

async function loadExample(filename = null) {
    try {
        // If no filename, show example selector
        if (!filename) {
            await showExampleSelector();
            return;
        }
        
        const response = await fetch(`/api/examples/${filename}`);
        const data = await response.json();
        
        if (data.success) {
            state.pfd = data.pfd;
            state.filename = data.filename || filename;
            document.getElementById('current-filename').textContent = state.filename;
            codeEditor.value = data.text;
            clearSelection();
            fitToScreen();
            
            // Extract process name for toast
            let processName = filename.replace('.pfd', '').replace(/_/g, ' ');
            for (const line of data.text.split('\n')) {
                if (line.startsWith('PROCESS:')) {
                    processName = line.split(':')[1].trim();
                    break;
                }
            }
            showToast('success', `Loaded: ${processName}`);
        } else {
            showToast('error', data.error || 'Failed to load example');
        }
    } catch (err) {
        showToast('error', 'Failed to load example');
    }
}

async function showExampleSelector() {
    try {
        const response = await fetch('/api/examples');
        const data = await response.json();
        
        if (!data.success) {
            showToast('error', 'Failed to load examples list');
            return;
        }
        
        // Create modal HTML
        const modal = document.createElement('div');
        modal.className = 'modal-overlay active';
        modal.id = 'examples-modal';
        modal.innerHTML = `
            <div class="modal">
                <div class="modal-header">
                    <h2 class="modal-title">Load Example</h2>
                    <button class="modal-close" onclick="closeModal('examples-modal')">
                        <svg viewBox="0 0 24 24" width="20" height="20" stroke="currentColor" fill="none" stroke-width="2">
                            <line x1="18" y1="6" x2="6" y2="18"/>
                            <line x1="6" y1="6" x2="18" y2="18"/>
                        </svg>
                    </button>
                </div>
                <div class="modal-body" style="max-height: 400px; overflow-y: auto;">
                    <div class="property-list">
                        ${data.examples.map(ex => `
                            <div class="property-row" style="cursor: pointer; padding: 12px; border-radius: 6px; margin-bottom: 8px; background: var(--bg-tertiary);" 
                                 onclick="loadExample('${ex.filename}'); closeModal('examples-modal');"
                                 onmouseover="this.style.background='var(--bg-elevated)'" 
                                 onmouseout="this.style.background='var(--bg-tertiary)'">
                                <div style="font-weight: 500;">${ex.name}</div>
                                <div style="font-size: 12px; color: var(--text-secondary); margin-top: 4px;">
                                    ${ex.filename} • ${ex.thermo_method}
                                </div>
                            </div>
                        `).join('')}
                    </div>
                </div>
            </div>
        `;
        
        document.body.appendChild(modal);
        
        // Close on background click
        modal.addEventListener('click', (e) => {
            if (e.target === modal) {
                closeModal('examples-modal');
            }
        });
        
    } catch (err) {
        showToast('error', 'Failed to load examples');
    }
}

// ============================================================================
// Validation
// ============================================================================

async function validatePFD() {
    try {
        const response = await fetch('/api/validate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ pfd: state.pfd })
        });
        
        const data = await response.json();
        showValidationResults(data.errors || [], data.warnings || []);
        
        if (data.success && data.errors.length === 0) {
            showToast('success', 'Validation passed!');
        }
    } catch (err) {
        showToast('error', 'Validation failed');
    }
}

function showValidationResults(errors, warnings) {
    const container = document.getElementById('validation-results');
    container.innerHTML = '';
    
    for (const error of errors) {
        const div = document.createElement('div');
        div.className = 'validation-item error';
        div.innerHTML = `
            <svg viewBox="0 0 24 24" stroke="currentColor" fill="none" stroke-width="2">
                <circle cx="12" cy="12" r="10"/>
                <line x1="15" y1="9" x2="9" y2="15"/>
                <line x1="9" y1="9" x2="15" y2="15"/>
            </svg>
            <span>${error}</span>
        `;
        container.appendChild(div);
    }
    
    for (const warning of warnings) {
        const div = document.createElement('div');
        div.className = 'validation-item warning';
        div.innerHTML = `
            <svg viewBox="0 0 24 24" stroke="currentColor" fill="none" stroke-width="2">
                <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/>
                <line x1="12" y1="9" x2="12" y2="13"/>
                <line x1="12" y1="17" x2="12.01" y2="17"/>
            </svg>
            <span>${warning}</span>
        `;
        container.appendChild(div);
    }
}

// ============================================================================
// Components Modal
// ============================================================================

function openComponentsModal() {
    renderComponentsList();
    document.getElementById('components-modal').classList.add('active');
}

function closeModal(id) {
    document.getElementById(id).classList.remove('active');
}

function renderComponentsList() {
    const container = document.getElementById('components-list');
    container.innerHTML = '';
    
    for (let i = 0; i < state.pfd.components.length; i++) {
        const comp = state.pfd.components[i];
        const div = document.createElement('div');
        div.className = 'property-item';
        div.innerHTML = `
            <input type="text" class="form-input" value="${comp.symbol}" 
                   onchange="updateComponent(${i}, 'symbol', this.value)" placeholder="Symbol">
            <input type="text" class="form-input" value="${comp.name}" 
                   onchange="updateComponent(${i}, 'name', this.value)" placeholder="Name">
            <button class="btn btn-icon btn-danger" onclick="removeComponent(${i})">
                <svg viewBox="0 0 24 24" stroke="currentColor" fill="none" stroke-width="2">
                    <line x1="18" y1="6" x2="6" y2="18"/>
                    <line x1="6" y1="6" x2="18" y2="18"/>
                </svg>
            </button>
        `;
        container.appendChild(div);
    }
}

function updateComponent(index, field, value) {
    if (state.pfd.components[index]) {
        state.pfd.components[index][field] = value;
        state.isDirty = true;
        syncToCodeEditor();
    }
}

function removeComponent(index) {
    state.pfd.components.splice(index, 1);
    state.isDirty = true;
    renderComponentsList();
    syncToCodeEditor();
}

function addComponent() {
    state.pfd.components.push({
        symbol: '',
        name: '',
        molecular_weight: 0
    });
    state.isDirty = true;
    renderComponentsList();
}

// ============================================================================
// Toast Notifications
// ============================================================================

function showToast(type, message) {
    const container = document.getElementById('toast-container');
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.innerHTML = `
        <svg viewBox="0 0 24 24" width="16" height="16" stroke="${type === 'error' ? '#ef4444' : '#10b981'}" fill="none" stroke-width="2">
            ${type === 'error' 
                ? '<circle cx="12" cy="12" r="10"/><line x1="15" y1="9" x2="9" y2="15"/><line x1="9" y1="9" x2="15" y2="15"/>'
                : '<path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/>'}
        </svg>
        <span>${message}</span>
    `;
    container.appendChild(toast);
    
    setTimeout(() => {
        toast.remove();
    }, 4000);
}

// ============================================================================
// Keyboard Shortcuts
// ============================================================================

document.addEventListener('keydown', (e) => {
    // Delete selected
    if ((e.key === 'Delete' || e.key === 'Backspace') && state.selection) {
        if (document.activeElement.tagName !== 'INPUT' && document.activeElement.tagName !== 'TEXTAREA') {
            e.preventDefault();
            deleteSelected();
        }
    }
    
    // Escape to clear selection
    if (e.key === 'Escape') {
        clearSelection();
        closeModal('components-modal');
    }
    
    // Ctrl+S to save
    if (e.ctrlKey && e.key === 's') {
        e.preventDefault();
        downloadFile();
    }
    
    // Ctrl+O to open
    if (e.ctrlKey && e.key === 'o') {
        e.preventDefault();
        openFile();
    }
});

// ============================================================================
// Simulation
// ============================================================================

let simulationResults = null;
let pfrContent = null;

async function runSimulation() {
    const statusDiv = document.getElementById('simulation-status');
    const downloadBtn = document.getElementById('download-results-btn');
    
    // Show running status
    statusDiv.style.display = 'block';
    statusDiv.className = 'simulation-status running';
    statusDiv.innerHTML = `
        <h4>⏳ Running Simulation...</h4>
        <p>Please wait while the process is being simulated.</p>
    `;
    downloadBtn.disabled = true;
    
    try {
        const text = codeEditor.getValue();
        
        const response = await fetch('/api/simulate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text: text })
        });
        
        const data = await response.json();
        
        if (!data.success) {
            statusDiv.className = 'simulation-status error';
            statusDiv.innerHTML = `
                <h4>❌ Simulation Failed</h4>
                <p>${data.error || 'Unknown error'}</p>
            `;
            showToast('error', 'Simulation failed');
            return;
        }
        
        // Store results
        simulationResults = data.results;
        pfrContent = data.pfr_content;
        
        // Show success
        const convergedClass = data.converged ? 'success' : 'warning';
        const convergedIcon = data.converged ? '✅' : '⚠️';
        const convergedText = data.converged ? 'Converged' : 'Did Not Converge';
        
        statusDiv.className = `simulation-status ${convergedClass}`;
        statusDiv.innerHTML = `
            <h4>${convergedIcon} ${convergedText}</h4>
            <div class="metric">
                <span>Thermo Method:</span>
                <span class="metric-value">${data.thermo_method || 'IDEAL'}</span>
            </div>
            <div class="metric">
                <span>Iterations:</span>
                <span class="metric-value">${data.iterations}</span>
            </div>
            <div class="metric">
                <span>Mass Balance Error:</span>
                <span class="metric-value">${(data.mass_balance_error * 100).toFixed(4)}%</span>
            </div>
            <div class="metric">
                <span>Energy Balance Error:</span>
                <span class="metric-value">${(data.energy_balance_error * 100).toFixed(2)}%</span>
            </div>
            ${data.warnings && data.warnings.length > 0 ? `
                <div style="margin-top: 8px; font-size: 11px; opacity: 0.8;">
                    <strong>Warnings:</strong> ${data.warnings.length}
                </div>
            ` : ''}
        `;
        
        // Enable download button
        downloadBtn.disabled = false;
        
        showToast('success', `Simulation ${data.converged ? 'converged' : 'completed'} in ${data.iterations} iterations`);
        
    } catch (error) {
        statusDiv.className = 'simulation-status error';
        statusDiv.innerHTML = `
            <h4>❌ Error</h4>
            <p>${error.message}</p>
        `;
        showToast('error', 'Simulation error: ' + error.message);
    }
}

function downloadResults() {
    if (!pfrContent) {
        showToast('error', 'No simulation results to download');
        return;
    }
    
    // Get filename from current PFD
    let filename = document.getElementById('current-filename').textContent;
    filename = filename.replace('.pfd', '.pfr');
    
    // Create blob and download
    const blob = new Blob([pfrContent], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    a.click();
    URL.revokeObjectURL(url);
    
    showToast('success', `Downloaded ${filename}`);
}

// ============================================================================
// Initialization
// ============================================================================

function init() {
    initPalette();
    render();
    
    // Close modal when clicking outside
    document.getElementById('components-modal').addEventListener('click', (e) => {
        if (e.target.classList.contains('modal-overlay')) {
            closeModal('components-modal');
        }
    });
}

init();
