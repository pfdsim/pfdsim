import os
from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
from io import BytesIO

if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
    from .pfd_parser import (
        parse_pfd, 
        validate_pfd, 
        ProcessFlowDiagram,
        ParseError
    )
else:
    from pfd_parser import (
        parse_pfd, 
        validate_pfd, 
        ProcessFlowDiagram,
        ParseError
    )
if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
    from .dof_analyzer import analyze_dof, SpecificationStatus
else:
    from dof_analyzer import analyze_dof, SpecificationStatus
if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
    from .chemical_properties import get_database, get_chemical, validate_components
else:
    from chemical_properties import get_database, get_chemical, validate_components

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-key-change-in-production')

# Store current PFD in memory (in production, use a database or session storage)
current_pfd = None


@app.route('/')
def index():
    """Render the main editor page"""
    return render_template('index.html')


@app.route('/vle-chart')
def vle_chart_page():
    """Render the VLE/LLE phase diagram page"""
    return render_template('vle_chart.html')


@app.route('/api/parse', methods=['POST'])
def api_parse():
    """Parse PFD text and return JSON structure"""
    try:
        text = request.json.get('text', '')
        pfd = parse_pfd(text)
        errors, warnings = validate_pfd(pfd)
        
        return jsonify({
            'success': True,
            'pfd': pfd.to_dict(),
            'errors': errors,
            'warnings': warnings
        })
    except ParseError as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'line_number': e.line_number
        }), 400
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/serialize', methods=['POST'])
def api_serialize():
    """Convert JSON structure back to PFD text"""
    try:
        data = request.json.get('pfd', {})
        pfd = ProcessFlowDiagram.from_dict(data)
        text = pfd.to_pfd()
        
        return jsonify({
            'success': True,
            'text': text
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/validate', methods=['POST'])
def api_validate():
    """Validate a PFD structure including DOF analysis and chemical properties"""
    try:
        data = request.json.get('pfd', {})
        pfd = ProcessFlowDiagram.from_dict(data)
        
        # Basic structural validation
        errors, warnings = validate_pfd(pfd)
        
        # Chemical property validation
        component_symbols = [c.symbol for c in pfd.components]
        found, missing = validate_components(component_symbols)
        
        for symbol in missing:
            warnings.append(f"Component '{symbol}' not in chemical database - properties unavailable")
        
        # DOF analysis
        dof_result = analyze_dof(pfd)
        
        # Add DOF errors and warnings
        errors.extend(dof_result.errors)
        warnings.extend(dof_result.warnings)
        
        # Build DOF details for response
        dof_details = {
            'overall_status': dof_result.overall_status.value,
            'total_dof': dof_result.total_dof,
            'units': [
                {
                    'id': r.entity_id,
                    'dof': r.dof,
                    'status': r.status.value,
                    'message': r.message,
                    'details': r.details
                }
                for r in dof_result.unit_results
            ],
            'streams': [
                {
                    'id': r.entity_id,
                    'status': r.status.value,
                    'message': r.message
                }
                for r in dof_result.stream_results
                if r.status != SpecificationStatus.OK
            ],
            'suggestions': dof_result.suggestions
        }
        
        return jsonify({
            'success': len(errors) == 0,
            'errors': errors,
            'warnings': warnings,
            'dof_analysis': dof_details,
            'chemicals_found': found,
            'chemicals_missing': missing
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/chemicals', methods=['GET'])
def api_chemicals():
    """Get list of all chemicals in database"""
    try:
        db = get_database()
        chemicals = []
        for symbol in db.list_all():
            props = db.get(symbol)
            if props:
                chemicals.append({
                    'symbol': symbol,
                    'name': props.name,
                    'formula': props.formula,
                    'MW': props.MW,
                    'Tc': props.Tc,
                    'Pc': props.Pc,
                    'Tb': props.Tb,
                    'phase_at_STP': props.phase_at_STP
                })
        return jsonify({
            'success': True,
            'chemicals': chemicals
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/chemicals/<symbol>', methods=['GET'])
def api_chemical_detail(symbol):
    """Get detailed properties for a specific chemical"""
    try:
        props = get_chemical(symbol)
        if props is None:
            return jsonify({
                'success': False,
                'error': f"Chemical '{symbol}' not found"
            }), 404
        
        return jsonify({
            'success': True,
            'chemical': {
                'symbol': props.symbol,
                'name': props.name,
                'formula': props.formula,
                'CAS': props.CAS,
                'MW': props.MW,
                'Tc': props.Tc,
                'Pc': props.Pc,
                'Vc': props.Vc,
                'omega': props.omega,
                'Tb': props.Tb,
                'Tm': props.Tm,
                'Hf': props.Hf,
                'Gf': props.Gf,
                'Hvap': props.Hvap,
                'phase_at_STP': props.phase_at_STP,
                'Cp_at_300K': props.Cp(300),
                'Cp_at_500K': props.Cp(500)
            }
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/chemicals/search', methods=['GET'])
def api_chemical_search():
    """Search chemicals by name or formula"""
    try:
        query = request.args.get('q', '')
        if not query:
            return jsonify({
                'success': False,
                'error': 'Query parameter q is required'
            }), 400
        
        db = get_database()
        results = db.search(query)
        
        return jsonify({
            'success': True,
            'query': query,
            'results': [
                {
                    'symbol': props.symbol,
                    'name': props.name,
                    'formula': props.formula,
                    'MW': props.MW
                }
                for props in results
            ]
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/upload', methods=['POST'])
def api_upload():
    """Upload and parse a .pfd file"""
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'}), 400
        
        if not file.filename.endswith('.pfd'):
            return jsonify({'success': False, 'error': 'File must have .pfd extension'}), 400
        
        text = file.read().decode('utf-8')
        pfd = parse_pfd(text)
        errors, warnings = validate_pfd(pfd)
        
        # Auto-layout units if no positions set
        pfd = auto_layout(pfd)
        
        return jsonify({
            'success': True,
            'filename': secure_filename(file.filename),
            'pfd': pfd.to_dict(),
            'text': text,
            'errors': errors,
            'warnings': warnings
        })
    except ParseError as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'line_number': e.line_number
        }), 400
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/download', methods=['POST'])
def api_download():
    """Generate and download a .pfd file"""
    try:
        data = request.json.get('pfd', {})
        filename = request.json.get('filename', 'process.pfd')
        
        pfd = ProcessFlowDiagram.from_dict(data)
        text = pfd.to_pfd()
        
        buffer = BytesIO()
        buffer.write(text.encode('utf-8'))
        buffer.seek(0)
        
        return send_file(
            buffer,
            mimetype='text/plain',
            as_attachment=True,
            download_name=filename
        )
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/examples')
def api_examples_list():
    """Return list of available example PFD files"""
    import os
    examples_dir = os.path.join(os.path.dirname(__file__), 'examples')
    
    examples = []
    for filename in sorted(os.listdir(examples_dir)):
        if filename.endswith('.pfd'):
            filepath = os.path.join(examples_dir, filename)
            try:
                with open(filepath, 'r') as f:
                    content = f.read()
                
                # Extract process name from file
                name = filename.replace('.pfd', '').replace('_', ' ').title()
                for line in content.split('\n'):
                    if line.startswith('PROCESS:'):
                        name = line.split(':', 1)[1].strip()
                        break
                
                # Get thermo method
                thermo = 'IDEAL'
                for line in content.split('\n'):
                    if line.startswith('THERMO_METHOD:'):
                        thermo = line.split(':', 1)[1].strip()
                        break
                
                examples.append({
                    'filename': filename,
                    'name': name,
                    'thermo_method': thermo,
                })
            except (OSError, UnicodeError):
                pass
    
    return jsonify({
        'success': True,
        'examples': examples
    })


@app.route('/api/examples/<filename>')
def api_example_file(filename):
    """Load a specific example PFD file"""
    import os
    
    # Sanitize filename
    if not filename.endswith('.pfd'):
        filename += '.pfd'
    filename = os.path.basename(filename)  # Prevent path traversal
    
    examples_dir = os.path.join(os.path.dirname(__file__), 'examples')
    filepath = os.path.join(examples_dir, filename)
    
    if not os.path.exists(filepath):
        return jsonify({
            'success': False,
            'error': f'Example not found: {filename}'
        }), 404
    
    try:
        with open(filepath, 'r') as f:
            content = f.read()
        
        pfd = parse_pfd(content)
        pfd = auto_layout(pfd)
        errors, warnings = validate_pfd(pfd)
        
        return jsonify({
            'success': True,
            'pfd': pfd.to_dict(),
            'text': content,
            'filename': filename,
            'errors': errors,
            'warnings': warnings
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/example')
def api_example():
    """Return the default example PFD (simple_flash.pfd)"""
    return api_example_file('simple_flash.pfd')


def auto_layout(pfd: ProcessFlowDiagram) -> ProcessFlowDiagram:
    """
    Auto-layout units in a simple left-to-right flow.
    Uses a topological sort based on stream connections.
    """
    if not pfd.units:
        return pfd
    
    # Check if positions are already set
    has_positions = any(u.x != 0 or u.y != 0 for u in pfd.units)
    if has_positions:
        return pfd
    
    # Build connection graph
    unit_ids = {u.id for u in pfd.units}
    
    # Find units that receive from FEED (starting points)
    feed_units = set()
    for stream in pfd.streams:
        if stream.source.is_feed and stream.destination.unit_id:
            feed_units.add(stream.destination.unit_id)
    
    # Build adjacency list (unit -> downstream units)
    downstream = {u.id: set() for u in pfd.units}
    upstream = {u.id: set() for u in pfd.units}
    
    for stream in pfd.streams:
        src_unit = stream.source.unit_id
        dst_unit = stream.destination.unit_id
        
        if src_unit and dst_unit and src_unit in unit_ids and dst_unit in unit_ids:
            downstream[src_unit].add(dst_unit)
            upstream[dst_unit].add(src_unit)
    
    # Simple topological layout
    # Assign levels based on distance from feed
    levels = {}
    
    # Start with feed units at level 0
    if feed_units:
        queue = list(feed_units)
        for u in queue:
            levels[u] = 0
    else:
        # If no feed, start with first unit
        queue = [pfd.units[0].id]
        levels[queue[0]] = 0
    
    # BFS to assign levels
    visited = set(queue)
    while queue:
        current = queue.pop(0)
        current_level = levels[current]
        
        for next_unit in downstream[current]:
            if next_unit not in visited:
                levels[next_unit] = current_level + 1
                visited.add(next_unit)
                queue.append(next_unit)
    
    # Assign any unvisited units
    for unit in pfd.units:
        if unit.id not in levels:
            levels[unit.id] = 0
    
    # Group units by level
    level_groups = {}
    for unit_id, level in levels.items():
        if level not in level_groups:
            level_groups[level] = []
        level_groups[level].append(unit_id)
    
    # Assign positions
    x_spacing = 250
    y_spacing = 150
    x_offset = 100
    y_offset = 100
    
    for level, unit_ids_in_level in level_groups.items():
        for i, unit_id in enumerate(unit_ids_in_level):
            unit = pfd.get_unit(unit_id)
            if unit:
                unit.x = x_offset + level * x_spacing
                unit.y = y_offset + i * y_spacing
    
    return pfd


@app.route('/api/unit-templates')
def api_unit_templates():
    """Return available unit operation templates with default ports"""
    templates = {
        # Mixing and Splitting
        'Mixer': {
            'category': 'mixing',
            'ports': [
                {'id': 'in1', 'port_type': 'inlet'},
                {'id': 'in2', 'port_type': 'inlet'},
                {'id': 'out', 'port_type': 'outlet'}
            ],
            'default_params': [
                {'name': 'mode', 'value': 'adiabatic', 'unit': None}
            ]
        },
        'Splitter': {
            'category': 'mixing',
            'ports': [
                {'id': 'in', 'port_type': 'inlet'},
                {'id': 'out1', 'port_type': 'outlet'},
                {'id': 'out2', 'port_type': 'outlet'}
            ],
            'default_params': [
                {'name': 'split_frac', 'value': '0.5', 'unit': None}
            ]
        },
        
        # Pressure Change
        'Pump': {
            'category': 'pressure_change',
            'ports': [
                {'id': 'in', 'port_type': 'inlet'},
                {'id': 'out', 'port_type': 'outlet'}
            ],
            'default_params': [
                {'name': 'P_out', 'value': '', 'unit': 'bar'},
                {'name': 'eta', 'value': '0.75', 'unit': None}
            ]
        },
        'Compressor': {
            'category': 'pressure_change',
            'ports': [
                {'id': 'in', 'port_type': 'inlet'},
                {'id': 'out', 'port_type': 'outlet'}
            ],
            'default_params': [
                {'name': 'P_out', 'value': '', 'unit': 'bar'},
                {'name': 'eta_isen', 'value': '0.75', 'unit': None},
                {'name': 'eta_mech', 'value': '0.98', 'unit': None}
            ]
        },
        'Expander': {
            'category': 'pressure_change',
            'ports': [
                {'id': 'in', 'port_type': 'inlet'},
                {'id': 'out', 'port_type': 'outlet'}
            ],
            'default_params': [
                {'name': 'P_out', 'value': '', 'unit': 'bar'},
                {'name': 'eta_isen', 'value': '0.80', 'unit': None}
            ]
        },
        'Valve': {
            'category': 'pressure_change',
            'ports': [
                {'id': 'in', 'port_type': 'inlet'},
                {'id': 'out', 'port_type': 'outlet'}
            ],
            'default_params': [
                {'name': 'P_out', 'value': '', 'unit': 'bar'}
            ]
        },
        'Pipe': {
            'category': 'pressure_change',
            'ports': [
                {'id': 'in', 'port_type': 'inlet'},
                {'id': 'out', 'port_type': 'outlet'}
            ],
            'default_params': [
                {'name': 'length', 'value': '100', 'unit': 'm'},
                {'name': 'diameter', 'value': '0.1', 'unit': 'm'},
                {'name': 'material', 'value': 'commercial_steel', 'unit': None},
                {'name': 'orientation', 'value': 'horizontal', 'unit': None}
            ]
        },
        
        # Heat Transfer
        'Heater': {
            'category': 'heat_transfer',
            'ports': [
                {'id': 'in', 'port_type': 'inlet'},
                {'id': 'out', 'port_type': 'outlet'}
            ],
            'default_params': [
                {'name': 'T_out', 'value': '', 'unit': 'C'}
            ]
        },
        'Cooler': {
            'category': 'heat_transfer',
            'ports': [
                {'id': 'in', 'port_type': 'inlet'},
                {'id': 'out', 'port_type': 'outlet'}
            ],
            'default_params': [
                {'name': 'T_out', 'value': '', 'unit': 'C'}
            ]
        },
        'HeatExchanger': {
            'category': 'heat_transfer',
            'ports': [
                {'id': 'tube_in', 'port_type': 'inlet'},
                {'id': 'tube_out', 'port_type': 'outlet'},
                {'id': 'shell_in', 'port_type': 'inlet'},
                {'id': 'shell_out', 'port_type': 'outlet'}
            ],
            'default_params': [
                {'name': 'U', 'value': '500', 'unit': 'W/m2-K'},
                {'name': 'A', 'value': '', 'unit': 'm2'},
                {'name': 'type', 'value': 'countercurrent', 'unit': None}
            ]
        },
        
        # Vapor-Liquid Separation
        'Flash': {
            'category': 'separation',
            'ports': [
                {'id': 'in', 'port_type': 'inlet'},
                {'id': 'vap', 'port_type': 'vapor_outlet'},
                {'id': 'liq', 'port_type': 'liquid_outlet'}
            ],
            'default_params': [
                {'name': 'T', 'value': '', 'unit': 'C'},
                {'name': 'P', 'value': '', 'unit': 'bar'}
            ]
        },
        'FlashLLE': {
            'category': 'separation',
            'ports': [
                {'id': 'in', 'port_type': 'inlet'},
                {'id': 'liq1', 'port_type': 'liquid1_outlet'},
                {'id': 'liq2', 'port_type': 'liquid2_outlet'}
            ],
            'default_params': [
                {'name': 'T', 'value': '', 'unit': 'C'},
                {'name': 'P', 'value': '', 'unit': 'bar'},
                {'name': 'thermo_model', 'value': 'NRTL', 'unit': None}
            ]
        },
        'ThreePhaseFlash': {
            'category': 'separation',
            'ports': [
                {'id': 'in', 'port_type': 'inlet'},
                {'id': 'vapor_out', 'port_type': 'vapor_outlet'},
                {'id': 'liquid1_out', 'port_type': 'liquid1_outlet'},
                {'id': 'liquid2_out', 'port_type': 'liquid2_outlet'}
            ],
            'default_params': [
                {'name': 'T', 'value': '', 'unit': 'C'},
                {'name': 'P', 'value': '', 'unit': 'bar'},
                {'name': 'thermo_model', 'value': 'NRTL', 'unit': None}
            ]
        },
        'Decanter': {
            'category': 'separation',
            'ports': [
                {'id': 'in', 'port_type': 'inlet'},
                {'id': 'light', 'port_type': 'light_liquid_outlet'},
                {'id': 'heavy', 'port_type': 'heavy_liquid_outlet'}
            ],
            'default_params': [
                {'name': 'T', 'value': '', 'unit': 'C'},
                {'name': 'thermo_model', 'value': 'NRTL', 'unit': None}
            ]
        },
        
        # Multi-stage Separation
        'ShortcutDistillation': {
            'category': 'separation',
            'ports': [
                {'id': 'feed', 'port_type': 'inlet'},
                {'id': 'distillate', 'port_type': 'outlet'},
                {'id': 'bottoms', 'port_type': 'outlet'}
            ],
            'default_params': [
                {'name': 'N_stages', 'value': '10', 'unit': None},
                {'name': 'reflux_ratio', 'value': '', 'unit': None},
                {'name': 'condenser', 'value': 'total', 'unit': None},
                {'name': 'reboiler', 'value': 'kettle', 'unit': None}
            ],
            'description': 'Shortcut distillation using Fenske-Underwood-Gilliland'
        },
        'RigorousDistillation': {
            'category': 'separation',
            'ports': [
                {'id': 'feed', 'port_type': 'inlet'},
                {'id': 'distillate', 'port_type': 'outlet'},
                {'id': 'bottoms', 'port_type': 'outlet'}
            ],
            'default_params': [
                {'name': 'N_stages', 'value': '15', 'unit': None},
                {'name': 'feed_stage', 'value': '8', 'unit': None},
                {'name': 'reflux_ratio', 'value': '2.0', 'unit': None},
                {'name': 'P_condenser', 'value': '1.0', 'unit': 'bar'},
                {'name': 'condenser_type', 'value': 'total', 'unit': None},
                {'name': 'D_to_F', 'value': '0.5', 'unit': None}
            ],
            'description': 'Stage-by-stage equilibrium calculation with UNIFAC activity coefficients'
        },
        'DistillationLLE': {
            'category': 'separation',
            'ports': [
                {'id': 'feed', 'port_type': 'inlet'},
                {'id': 'distillate', 'port_type': 'outlet'},
                {'id': 'bottoms', 'port_type': 'outlet'}
            ],
            'default_params': [
                {'name': 'N_stages', 'value': '10', 'unit': None},
                {'name': 'feed_stage', 'value': '5', 'unit': None},
                {'name': 'reflux_ratio', 'value': '', 'unit': None},
                {'name': 'decanter_stage', 'value': '1', 'unit': None}
            ]
        },
        'ReactiveDistillation': {
            'category': 'separation',
            'ports': [
                {'id': 'feed', 'port_type': 'inlet'},
                {'id': 'distillate', 'port_type': 'outlet'},
                {'id': 'bottoms', 'port_type': 'outlet'}
            ],
            'default_params': [
                {'name': 'N_stages', 'value': '15', 'unit': None},
                {'name': 'feed_stage', 'value': '8', 'unit': None},
                {'name': 'reflux_ratio', 'value': '', 'unit': None},
                {'name': 'reactive_stages', 'value': '5-10', 'unit': None}
            ]
        },
        'Absorber': {
            'category': 'separation',
            'ports': [
                {'id': 'gas_in', 'port_type': 'inlet'},
                {'id': 'solvent_in', 'port_type': 'inlet'},
                {'id': 'gas_out', 'port_type': 'outlet'},
                {'id': 'liquid_out', 'port_type': 'outlet'}
            ],
            'default_params': [
                {'name': 'N_stages', 'value': '10', 'unit': None},
                {'name': 'tray_efficiency', 'value': '0.25', 'unit': None}
            ]
        },
        'Stripper': {
            'category': 'separation',
            'ports': [
                {'id': 'liquid_in', 'port_type': 'inlet'},
                {'id': 'stripping_in', 'port_type': 'inlet'},
                {'id': 'vapor_out', 'port_type': 'vapor_outlet'},
                {'id': 'liquid_out', 'port_type': 'liquid_outlet'}
            ],
            'default_params': [
                {'name': 'N_stages', 'value': '10', 'unit': None},
                {'name': 'stripping_agent', 'value': 'steam', 'unit': None}
            ]
        },
        'Extractor': {
            'category': 'separation',
            'ports': [
                {'id': 'feed', 'port_type': 'inlet'},
                {'id': 'solvent', 'port_type': 'inlet'},
                {'id': 'extract', 'port_type': 'outlet'},
                {'id': 'raffinate', 'port_type': 'outlet'}
            ],
            'default_params': [
                {'name': 'N_stages', 'value': '5', 'unit': None},
                {'name': 'thermo_model', 'value': 'NRTL', 'unit': None}
            ]
        },
        
        # Reactors
        'Reactor': {
            'category': 'reaction',
            'ports': [
                {'id': 'in', 'port_type': 'inlet'},
                {'id': 'out', 'port_type': 'outlet'}
            ],
            'default_params': [
                {'name': 'T', 'value': '', 'unit': 'C'},
                {'name': 'P_drop', 'value': '0', 'unit': 'bar'},
                {'name': 'mode', 'value': 'isothermal', 'unit': None}
            ]
        },
        'EquilibriumReactor': {
            'category': 'reaction',
            'ports': [
                {'id': 'in', 'port_type': 'inlet'},
                {'id': 'out', 'port_type': 'outlet'}
            ],
            'default_params': [
                {'name': 'T', 'value': '', 'unit': 'C'},
                {'name': 'approach', 'value': '1.0', 'unit': None},
                {'name': 'P_drop', 'value': '0', 'unit': 'bar'}
            ]
        },
        'GibbsReactor': {
            'category': 'reaction',
            'ports': [
                {'id': 'in', 'port_type': 'inlet'},
                {'id': 'out', 'port_type': 'outlet'}
            ],
            'default_params': [
                {'name': 'T', 'value': '', 'unit': 'C'},
                {'name': 'P_drop', 'value': '0', 'unit': 'bar'}
            ]
        },
        'PFR': {
            'category': 'reaction',
            'ports': [
                {'id': 'in', 'port_type': 'inlet'},
                {'id': 'out', 'port_type': 'outlet'}
            ],
            'default_params': [
                {'name': 'length', 'value': '', 'unit': 'm'},
                {'name': 'diameter', 'value': '', 'unit': 'm'},
                {'name': 'mode', 'value': 'isothermal', 'unit': None}
            ]
        },
        'CSTR': {
            'category': 'reaction',
            'ports': [
                {'id': 'in', 'port_type': 'inlet'},
                {'id': 'out', 'port_type': 'outlet'}
            ],
            'default_params': [
                {'name': 'volume', 'value': '', 'unit': 'm3'},
                {'name': 'mode', 'value': 'isothermal', 'unit': None}
            ]
        },
        'BatchReactor': {
            'category': 'reaction',
            'ports': [
                {'id': 'in', 'port_type': 'inlet'},
                {'id': 'out', 'port_type': 'outlet'}
            ],
            'default_params': [
                {'name': 'volume', 'value': '', 'unit': 'm3'},
                {'name': 'reaction_time', 'value': '', 'unit': 'h'}
            ]
        },
        
        # Solids Handling
        'Crystallizer': {
            'category': 'solids',
            'ports': [
                {'id': 'in', 'port_type': 'inlet'},
                {'id': 'crystals', 'port_type': 'solid_outlet'},
                {'id': 'mother_liquor', 'port_type': 'liquid_outlet'}
            ],
            'default_params': [
                {'name': 'T_out', 'value': '', 'unit': 'C'},
                {'name': 'mode', 'value': 'cooling', 'unit': None}
            ]
        },
        'Filter': {
            'category': 'solids',
            'ports': [
                {'id': 'slurry_in', 'port_type': 'inlet'},
                {'id': 'cake', 'port_type': 'solid_outlet'},
                {'id': 'filtrate', 'port_type': 'liquid_outlet'}
            ],
            'default_params': [
                {'name': 'cake_moisture', 'value': '0.1', 'unit': None}
            ]
        },
        'Dryer': {
            'category': 'solids',
            'ports': [
                {'id': 'in', 'port_type': 'inlet'},
                {'id': 'out', 'port_type': 'outlet'}
            ],
            'default_params': [
                {'name': 'outlet_moisture', 'value': '0.01', 'unit': None}
            ]
        }
    }
    
    return jsonify(templates)


@app.route('/api/simulate', methods=['POST'])
def api_simulate():
    """Run process simulation on PFD"""
    try:
        if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
            from .simulator import Simulator, SimulationError
        else:
            from simulator import Simulator, SimulationError
        
        text = request.json.get('text', '')
        max_iterations = request.json.get('max_iterations', 50)
        thermo_method = request.json.get('thermo_method', None)
        
        if not text:
            return jsonify({
                'success': False,
                'error': 'No PFD text provided'
            }), 400
        
        # Create simulator and run
        sim = Simulator.from_string(text)
        result = sim.run(max_iterations=max_iterations, thermo_method=thermo_method)
        
        # Generate PFR content
        pfr_content = sim._generate_pfr()
        
        # Get detailed results
        results_dict = sim.get_results_dict()
        
        return jsonify({
            'success': True,
            'converged': result.converged,
            'iterations': result.iterations,
            'mass_balance_error': result.mass_balance_error,
            'energy_balance_error': result.energy_balance_error,
            'thermo_method': sim.thermo_method,
            'pfr_content': pfr_content,
            'results': results_dict,
            'errors': result.errors,
            'warnings': result.warnings
        })
        
    except SimulationError as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'error_type': 'simulation'
        }), 400
    except ParseError as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'error_type': 'parse',
            'line_number': e.line_number
        }), 400
    except Exception as e:
        import traceback
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500


@app.route('/api/simulate/download', methods=['POST'])
def api_simulate_download():
    """Run simulation and return PFR file for download"""
    try:
        if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
            from .simulator import Simulator, SimulationError
        else:
            from simulator import Simulator, SimulationError
        
        text = request.json.get('text', '')
        filename = request.json.get('filename', 'results.pfr')
        
        if not text:
            return jsonify({
                'success': False,
                'error': 'No PFD text provided'
            }), 400
        
        # Ensure .pfr extension
        if not filename.endswith('.pfr'):
            filename = filename.rsplit('.', 1)[0] + '.pfr'
        
        # Create simulator and run
        sim = Simulator.from_string(text)
        sim.run()
        
        # Generate PFR content
        pfr_content = sim._generate_pfr()
        
        # Create file-like object
        buffer = BytesIO()
        buffer.write(pfr_content.encode('utf-8'))
        buffer.seek(0)
        
        return send_file(
            buffer,
            mimetype='text/plain',
            as_attachment=True,
            download_name=filename
        )
        
    except (SimulationError, ParseError) as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/vle-chart', methods=['POST'])
def api_vle_chart():
    """
    Generate VLE/VLLE equilibrium chart data using UNIFAC.
    
    Request JSON:
        comp1: First component name (can be name, formula, SMILES, or CAS)
        comp2: Second component name
        chart_type: 'Txy', 'Pxy', or 'xy'
        pressure: Pressure in bar (for Txy, xy charts)
        temperature: Temperature in °C (for Pxy charts)
        n_points: Number of data points (default 50)
        
    Returns:
        JSON with chart data including x, y, T or P arrays, azeotrope info,
        and LLE region if detected (for VLLE systems)
    """
    try:
        if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
            from .thermodynamics import create_thermodynamics
        else:
            from thermodynamics import create_thermodynamics
        if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
            from .chemical_properties import ChemicalDatabase
        else:
            from chemical_properties import ChemicalDatabase
        
        data = request.json
        comp1 = data.get('comp1', data.get('component1'))
        comp2 = data.get('comp2', data.get('component2'))
        chart_type = data.get('chart_type', data.get('type', 'Txy')).upper()
        P = data.get('pressure', data.get('P', 1.0))  # bar
        T = data.get('temperature', data.get('T', 25))  # °C
        n_points = int(data.get('n_points', 50))
        
        if not comp1 or not comp2:
            return jsonify({
                'success': False,
                'error': 'Must specify comp1 and comp2'
            }), 400
        
        # Resolve component names (supports names, formulas, SMILES, CAS)
        db = ChemicalDatabase()
        
        # Try to resolve, with online lookup if needed
        c1 = db.get(comp1, fetch_online=True)
        c2 = db.get(comp2, fetch_online=True)
        
        if c1 is None:
            return jsonify({
                'success': False,
                'error': f"Component '{comp1}' not found. Try using formula, SMILES, or CAS number."
            }), 400
        if c2 is None:
            return jsonify({
                'success': False,
                'error': f"Component '{comp2}' not found. Try using formula, SMILES, or CAS number."
            }), 400
        
        # Use internal symbol names
        comp1_sym = c1.symbol
        comp2_sym = c2.symbol
        
        # Create UNIFAC thermodynamics
        thermo = create_thermodynamics([comp1_sym, comp2_sym], db=db, method='UNIFAC')
        
        # Convert T to Kelvin
        T_K = T + 273.15
        
        result = {
            'success': True,
            'comp1': comp1_sym,
            'comp1_name': c1.name,
            'comp2': comp2_sym,
            'comp2_name': c2.name,
            'chart_type': chart_type,
        }
        
        if chart_type == 'TXY':
            chart_data = thermo.generate_Txy_data(comp1_sym, comp2_sym, P, n_points)
            result.update({
                'x': chart_data['x'],
                'y': chart_data['y'],
                'T_bubble': chart_data['T_bubble'],
                'T_dew': chart_data['T_dew'],
                'azeotrope': chart_data.get('azeotrope'),
                'lle_region': chart_data.get('lle_region'),  # Include LLE info for VLLE
                'pressure_bar': P,
            })
            
        elif chart_type == 'PXY':
            chart_data = thermo.generate_Pxy_data(comp1_sym, comp2_sym, T_K, n_points)
            result.update({
                'x': chart_data['x'],
                'y': chart_data['y'],
                'P_bubble': chart_data['P_bubble'],
                'P_dew': chart_data['P_dew'],
                'temperature_C': T,
            })
            
        elif chart_type == 'XY':
            chart_data = thermo.generate_xy_data(comp1_sym, comp2_sym, P=P, n_points=n_points)
            result.update({
                'x': chart_data['x'],
                'y': chart_data['y'],
                'diagonal_x': chart_data.get('diagonal_x', [0.0, 1.0]),
                'diagonal_y': chart_data.get('diagonal_y', [0.0, 1.0]),
                'pressure_bar': P,
            })
            
        else:
            return jsonify({
                'success': False,
                'error': f"Unknown chart type: {chart_type}. Use 'Txy', 'Pxy', or 'xy'"
            }), 400
        
        return jsonify(result)
        
    except Exception as e:
        import traceback
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500


@app.route('/api/vle-chart/components', methods=['GET'])
def api_vle_components():
    """Get list of components available for VLE calculations."""
    try:
        if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
            from .chemical_properties import ChemicalDatabase
        else:
            from chemical_properties import ChemicalDatabase
        if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
            from .unifac import KNOWN_MOLECULES
        else:
            from unifac import KNOWN_MOLECULES
        
        db = ChemicalDatabase()
        
        # Components from database that have enough data for VLE
        components = []
        for symbol, props in db.chemicals.items():
            if props.MW and (props.Tb or props.Tc):
                components.append({
                    'symbol': symbol,
                    'name': props.name,
                    'formula': props.formula or symbol,
                    'MW': props.MW,
                })
        
        # Add known UNIFAC molecules
        for name in KNOWN_MOLECULES.keys():
            if not any(c['symbol'].lower() == name.lower() or 
                      c['name'].lower() == name.lower() for c in components):
                components.append({
                    'symbol': name,
                    'name': name.title(),
                    'formula': name,
                    'MW': None,
                })
        
        return jsonify({
            'success': True,
            'components': sorted(components, key=lambda x: x['name'].lower()),
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/unifac-groups', methods=['POST'])
def api_unifac_groups():
    """Get UNIFAC groups for a component (by name or SMILES)."""
    try:
        if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
            from .chemical_properties import get_database
        else:
            from chemical_properties import get_database
        if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
            from .unifac import get_unifac_groups, parse_smiles_to_unifac, RDKIT_AVAILABLE
        else:
            from unifac import get_unifac_groups, parse_smiles_to_unifac, RDKIT_AVAILABLE
        
        data = request.json
        explicit_smiles = data.get('smiles')
        identifier = data.get('component') or data.get('name') or explicit_smiles
        variant = (data.get('variant') or data.get('method') or 'UNIFAC').upper()
        
        if not identifier:
            return jsonify({
                'success': False,
                'error': 'Must provide component name or SMILES'
            }), 400
        
        groups = None
        source = None
        
        if explicit_smiles:
            groups = parse_smiles_to_unifac(explicit_smiles, variant)
            source = 'native_smiles'
        else:
            # Try curated local groups first.
            try:
                groups = get_unifac_groups(identifier, variant=variant)
                source = 'curated'
            except ValueError:
                pass
        
            if groups is None:
                result = get_database().resolve_smiles_info(identifier, fetch_online=True)
                smiles = result.smiles if result else None
                if smiles:
                    groups = get_unifac_groups(
                        identifier,
                        smiles=smiles,
                        variant=variant,
                    )
                    source = 'resolved_smiles'
        
        if groups is None:
            return jsonify({
                'success': False,
                'error': f"Cannot determine UNIFAC groups for '{identifier}'"
            }), 400
        
        return jsonify({
            'success': True,
            'identifier': identifier,
            'groups': groups,
            'source': source,
            'rdkit_available': RDKIT_AVAILABLE,
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
