"""Flask interface for the current PFD model and isolated calculation workers."""

from dataclasses import asdict, fields
from io import BytesIO
import math
import hashlib
import json
import os
from pathlib import Path
import re
import hmac
import secrets
import fcntl

from flask import Flask, jsonify, render_template, request, send_file, session
from werkzeug.exceptions import HTTPException
from werkzeug.utils import secure_filename
from werkzeug.middleware.proxy_fix import ProxyFix

if __package__ and __package__.split(".", 1)[0] == "pfdsim":
    from .pfd_parser import (
        Component,
        Metadata,
        ParseError,
        PortType,
        ProcessFlowDiagram,
        parse_pfd,
        validate_pfd,
        configuration_field_catalog,
    )
    from .dof_analyzer import analyze_dof, SpecificationStatus, get_unit_info
    from .chemical_properties import get_database, get_chemical, validate_components
    from .perry_properties import get_perry_property_library
    from .unit_settings import unit_setting_schema, unit_supports_reactions
    from .kinetic_models import kinetic_input_catalog
    from .render import layout_flowsheet, RenderError
    from .unit_syntax import (
        UNIT_TYPE_ALIASES,
        UNIT_PORT_FAMILIES,
        NUMERIC_PORT_LAYOUTS,
        port_schema_for_unit_type,
    )
    from .thermodynamics_models.factory import SUPPORTED_METHODS
    from .fluid_phase_models import FLUID_PHASE_MODELS
    from .phase_behaviors import PHASE_BEHAVIORS
    from .solid_material_forms import SOLID_MATERIAL_FORMS
    from .recycle_controls import RECYCLE_METHOD_DEFAULTS
    from .web_jobs import JobStore, finite_json
    from .web_storage import (
        AccessLimit,
        StorageConflict,
        GUEST_CPU_SECONDS,
        ACCOUNT_CPU_SECONDS,
    )
else:
    from pfd_parser import (
        Component,
        Metadata,
        ParseError,
        PortType,
        ProcessFlowDiagram,
        parse_pfd,
        validate_pfd,
        configuration_field_catalog,
    )
    from dof_analyzer import analyze_dof, SpecificationStatus, get_unit_info
    from chemical_properties import get_database, get_chemical, validate_components
    from perry_properties import get_perry_property_library
    from unit_settings import unit_setting_schema, unit_supports_reactions
    from kinetic_models import kinetic_input_catalog
    from render import layout_flowsheet, RenderError
    from unit_syntax import (
        UNIT_TYPE_ALIASES,
        UNIT_PORT_FAMILIES,
        NUMERIC_PORT_LAYOUTS,
        port_schema_for_unit_type,
    )
    from thermodynamics_models.factory import SUPPORTED_METHODS
    from fluid_phase_models import FLUID_PHASE_MODELS
    from phase_behaviors import PHASE_BEHAVIORS
    from solid_material_forms import SOLID_MATERIAL_FORMS
    from recycle_controls import RECYCLE_METHOD_DEFAULTS
    from web_jobs import JobStore, finite_json
    from web_storage import (
        AccessLimit,
        StorageConflict,
        GUEST_CPU_SECONDS,
        ACCOUNT_CPU_SECONDS,
    )

app = Flask(__name__)
DATA_DIRECTORY = Path(
    os.environ.get("PFDSIM_WEB_DATA", Path.home() / ".local/share/pfdsim/web")
)
DATA_DIRECTORY.mkdir(parents=True, exist_ok=True, mode=0o700)


def session_secret():
    configured = os.environ.get("SECRET_KEY")
    if configured:
        return configured
    path = DATA_DIRECTORY / "session-secret"
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    with os.fdopen(descriptor, "r+b") as file:
        fcntl.flock(file, fcntl.LOCK_EX)
        value = file.read()
        if not value:
            value = secrets.token_bytes(32)
            file.write(value)
            file.flush()
            os.fsync(file.fileno())
    return value


app.config.update(
    MAX_CONTENT_LENGTH=16 * 1024 * 1024,
    JOB_DIRECTORY=DATA_DIRECTORY,
    SECRET_KEY=session_secret(),
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    SESSION_COOKIE_SECURE=os.environ.get("PFDSIM_SECURE_COOKIES") == "1",
)
proxy_hops = int(os.environ.get("PFDSIM_PROXY_HOPS", "0"))
if proxy_hops:
    app.wsgi_app = ProxyFix(app.wsgi_app, x_for=proxy_hops, x_proto=proxy_hops)
BASE_DIR = Path(__file__).resolve().parent


def jobs():
    if __package__ and __package__.split(".", 1)[0] == "pfdsim":
        from .activity_fit_store import DEFAULT_ACTIVITY_FITS_PATH
    else:
        from activity_fit_store import DEFAULT_ACTIVITY_FITS_PATH
    path = app.config.get("ACTIVITY_FITS_PATH") or (Path(app.config["JOB_DIRECTORY"])/"user_activity_fits.sqlite" if app.config.get("TESTING") else os.environ.get("PFDSIM_ACTIVITY_FITS_PATH", DEFAULT_ACTIVITY_FITS_PATH))
    return JobStore(app.config["JOB_DIRECTORY"], activity_fits_path=path)


def guest_principal():
    # A stable salted IP identity prevents resetting a guest budget just by
    # clearing cookies. Only its HMAC is stored, not the address itself.
    address = request.remote_addr or "local"
    secret = (
        app.secret_key.encode() if isinstance(app.secret_key, str) else app.secret_key
    )
    return "guest:" + hmac.new(secret, address.encode(), hashlib.sha256).hexdigest()


def identity():
    user = jobs().user(session.get("user_id", ""))
    if user:
        return "user:" + user["id"], "user:" + user["id"], ACCOUNT_CPU_SECONDS, user
    owner = "guest:" + session.get("guest_id", "")
    return owner, guest_principal(), GUEST_CPU_SECONDS, None


def owns_job(job):
    return job["owner"] in {identity()[0], "guest:" + session.get("guest_id", "")}


@app.before_request
def csrf_protection():
    if request.path.startswith("/api/") and request.method not in {
        "GET",
        "HEAD",
        "OPTIONS",
    }:
        supplied = request.headers.get("X-CSRF-Token", "")
        expected = session.get("csrf_token", "")
        if (
            not supplied
            or not expected
            or not secrets.compare_digest(supplied, expected)
        ):
            return respond(
                {
                    "success": False,
                    "error": "Session verification failed. Refresh the page and try again.",
                },
                403,
            )


@app.get("/api/session")
def api_session():
    session.setdefault("guest_id", secrets.token_urlsafe(24))
    session.setdefault("csrf_token", secrets.token_urlsafe(32))
    _, principal, limit, user = identity()
    return respond(
        {
            "success": True,
            "user": user,
            "csrf_token": session["csrf_token"],
            "quota": jobs().quota(principal, limit),
        }
    )


@app.post("/api/account/register")
@app.post("/api/account/login")
def api_account():
    data = body()
    store = jobs()
    if request.path.endswith("/register"):
        user = store.register(data.get("username"), data.get("password"), guest_principal(), setup_token=data.get("setup_token"))
    else:
        user = store.login(data.get("username"), data.get("password"), guest_principal())
    guest_id = session.get("guest_id")
    if guest_id:
        with store.connect() as db:
            db.execute(
                "UPDATE jobs SET owner=? WHERE owner=?",
                ("user:" + user["id"], "guest:" + guest_id),
            )
        activity_fit_store().claim_owner("guest:" + guest_id, "user:" + user["id"])
    session.clear()
    if guest_id:
        session["guest_id"] = guest_id
    session["user_id"] = user["id"]
    session["csrf_token"] = secrets.token_urlsafe(32)
    return api_session()


@app.post("/api/account/logout")
def api_logout():
    session.clear()
    return api_session()


@app.get("/api/flowsheets")
def api_flowsheets():
    _, _, _, user = identity()
    if not user:
        return respond(
            {"success": False, "error": "Sign in to access account laboratories."}, 401
        )
    return respond({"success": True, "flowsheets": jobs().flowsheets(user["id"])})


@app.post("/api/flowsheets/<identifier>")
def api_save_flowsheet(identifier):
    _, _, _, user = identity()
    if not user:
        return respond(
            {
                "success": False,
                "error": "Sign in to save laboratories to your account.",
            },
            401,
        )
    data = body()
    return respond(
        {
            "success": True,
            **jobs().save_flowsheet(
                user["id"], identifier, data.get("version"), data.get("document")
            ),
        }
    )


def respond(value, status=200):
    return jsonify(finite_json(value)), status


def body():
    data = request.get_json()
    if not isinstance(data, dict):
        raise ValueError("Request must contain a JSON object.")
    return data


def text_input(data):
    text = data.get("text")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Provide nonempty PFD text.")
    return text


def number(data, name, default, *, minimum=None, maximum=None, integer=False):
    value = data.get(name, default)
    if isinstance(value, bool):
        raise ValueError(f"{name} must be numeric.")
    try:
        result = float(value)
    except (ValueError, TypeError) as error:
        raise ValueError(f"{name} must be numeric.") from error
    if (
        not math.isfinite(result)
        or (minimum is not None and result < minimum)
        or (maximum is not None and result > maximum)
    ):
        raise ValueError(f"{name} must be finite and between {minimum} and {maximum}.")
    if integer and result != int(result):
        raise ValueError(f"{name} must be an integer.")
    return int(result) if integer else result


@app.errorhandler(Exception)
def handle_error(error):
    if isinstance(error, AccessLimit):
        status, message = 429, str(error)
    elif isinstance(error, StorageConflict):
        status, message = 409, str(error)
    elif isinstance(error, HTTPException):
        status, message = error.code, error.description
    elif isinstance(error, (ParseError, RenderError, ValueError, TypeError, KeyError)):
        status, message = 400, str(error)
    else:
        app.logger.exception("Web request failed")
        status, message = (
            500,
            "The server could not complete this request. See the server log for details.",
        )
    return respond(
        {
            "success": False,
            "error": message,
            "line_number": getattr(error, "line_number", None),
        },
        status,
    )


@app.after_request
def response_headers(response):
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "same-origin"
    if request.path.startswith("/api/"):
        response.headers["Cache-Control"] = "no-store"
    return response


@app.route("/")
@app.route("/editor")
@app.route("/settings")
def index():
    return render_template("index.html")


@app.route("/vle-chart")
def vle_chart_page():
    return render_template("vle_chart.html")


@app.get("/parameter-fitting")
def fitting_page():
    return render_template("parameter_fitting.html")


def fitting_module():
    if __package__ and __package__.split(".", 1)[0] == "pfdsim":
        from .thermodynamics_models import interaction_fitting
    else:
        from thermodynamics_models import interaction_fitting
    return interaction_fitting


def activity_fit_store():
    if __package__ and __package__.split(".", 1)[0] == "pfdsim":
        from .activity_fit_store import ActivityFitStore
    else:
        from activity_fit_store import ActivityFitStore
    return ActivityFitStore(jobs().activity_fits_path)


def require_fit_admin():
    _, _, _, user = identity()
    if not user or not user.get("is_admin"):
        from werkzeug.exceptions import Forbidden
        raise Forbidden("An administrator account is required.")
    return user


@app.get("/api/fitting/admin/submissions")
def api_fit_admin_list():
    require_fit_admin()
    return respond({"success":True,"submissions":activity_fit_store().list()})


@app.get("/api/fitting/admin/submissions/<identifier>")
def api_fit_admin_detail(identifier):
    require_fit_admin()
    return respond({"success":True,"submission":activity_fit_store().get(identifier)})


@app.post("/api/fitting/admin/review")
def api_fit_admin_review():
    user = require_fit_admin()
    data = body()
    reviewed = activity_fit_store().review(data.get("id"), "user:"+user["id"], data.get("action"),data.get("notes"),expected_version=data.get("version"))
    return respond({"success":True,"submission":reviewed})


@app.post("/api/fitting/admin/publish")
def api_fit_admin_publish():
    user = require_fit_admin()
    data = body()
    store = activity_fit_store()
    if data.get("job_id") or data.get("result"):
        result = completed_fit(data["job_id"]) if data.get("job_id") else data["result"]
        fitting_module().normalize_fit_request(result.get("request"))
        fitting_module().export_fit(result)
        identifier = data.get("job_id") or "local:"+hashlib.sha256(json.dumps(result,sort_keys=True,allow_nan=False).encode()).hexdigest()
        submitted = store.submit("user:"+user["id"],identifier,data.get("source"),result)
        record = store.get(submitted["id"])
        if record["status"] not in ("approved","published"):
            record = store.review(record["id"],"user:"+user["id"],"approve",data.get("notes") or "Direct administrator publication", expected_version=record["version"])
    else:
        record = store.get(data.get("id"))
    if record["status"] not in ("approved","published","publishing"):
        raise ValueError("Approve this fit before publication.")
    from_store = jobs()
    with from_store.connect() as db:
        queued = db.execute("SELECT id,payload FROM jobs WHERE kind='fit_publish' AND status IN ('queued','running')").fetchall()
    for job_id, payload in queued:
        if json.loads(payload).get("id") == record["id"] and from_store.get(job_id)["status"] in ("queued","running"):
            raise ValueError("Publication for this fit is already queued or running. Wait for it to finish, or cancel it before requesting recovery.")
    return submit("fit_publish", {"id":record["id"],"actor":"user:"+user["id"],"activity_fits_path":str(store.path),"action":"publish","notes":data.get("notes","")})


@app.post("/api/fitting/admin/withdraw")
def api_fit_admin_withdraw():
    user = require_fit_admin()
    data = body()
    record = activity_fit_store().get(data.get("id"))
    if record["status"] != "published":
        raise ValueError("Choose a published fit to withdraw.")
    return submit("fit_publish", {"id":record["id"],"actor":"user:"+user["id"],"activity_fits_path":str(activity_fit_store().path),"action":"withdraw","notes":data.get("notes","")})


@app.get("/api/fitting/catalog")
def api_fitting_catalog():
    return respond({"success": True, **fitting_module().fitting_catalog()})


@app.get("/api/fitting/sessions")
def api_fit_sessions():
    user=identity()[3]
    if not user:
        return respond({"success":False,"error":"Sign in to access account fitting sessions."},401)
    return respond({"success":True,"sessions":jobs().saved_documents("fit_sessions",user["id"])})


@app.post("/api/fitting/sessions/<identifier>")
def api_save_fit_session(identifier):
    user=identity()[3]
    if not user:
        return respond({"success":False,"error":"Sign in to save fitting sessions to your account."},401)
    data=body()
    return respond({"success":True,**jobs().save_fit_session(user["id"],identifier,data.get("version"),data.get("document"))})


@app.post("/api/fitting/parse")
def api_fitting_parse():
    data = body()
    return respond({"success": True, **fitting_module().inspect_observations(data.get("observations"),
                    import_options=data.get("import_options"), components=data.get("components"))})


@app.post("/api/fitting")
def api_fitting():
    return submit("fit", fitting_module().normalize_fit_request(body()))


@app.post("/api/fitting/prefill")
def api_fitting_prefill():
    data = body()
    data.update(model="UNIQUAC", fit_alpha=False, form="constant", cv={"method": "none"},
                observations=[{"kind": "GAMMA_INF", "T_K": 298.15, "gamma1_inf": 1}])
    data.pop("rq", None)
    data.pop("initial", None)
    data.pop("bounds", None)
    data.pop("vapor_parameters", None)
    data["extrapolation"] = "unrestricted"
    return submit("fit_prefill", fitting_module().normalize_fit_request(data))


def completed_fit(identifier):
    job = jobs().get(identifier)
    if not job or not owns_job(job) or job["kind"] != "fit" or job["status"] != "completed":
        raise ValueError("Choose one of your completed fitting jobs.")
    return job["output"]


@app.post("/api/fitting/export")
def api_fitting_export():
    data = body()
    if data.get("result") is not None:
        result = data["result"]
        if not isinstance(result, dict) or result.get("schema_version") != 1:
            raise ValueError("Provide a PFDSim fit report.")
        fitting_module().normalize_fit_request(result.get("request"))
    else:
        result = completed_fit(data.get("job_id"))
    return respond({"success": True, **fitting_module().export_fit(result, pfd_text=data.get("pfd_text"),
                    scope=data.get("scope", "global"), component_map=data.get("component_map"))})


@app.post("/api/fitting/submit")
def api_fitting_submit():
    data = body()
    identifier = data.get("job_id")
    if identifier and jobs().get(identifier) is not None:
        result = completed_fit(identifier)
    else:
        result = data.get("result")
        if not isinstance(result, dict) or result.get("schema_version") != 1:
            raise ValueError("Provide a completed fitting job or a PFDSim CLI fit report.")
        fitting_module().normalize_fit_request(result.get("request"))
        fitting_module().export_fit(result)
        result = {**result, "submission_origin": "external_report_pending_review"}
        identifier = "local:" + hashlib.sha256(json.dumps(result, sort_keys=True, allow_nan=False).encode()).hexdigest()
    submission = jobs().submit_fit(identity()[0], identifier, data.get("source"), result)
    return respond({"success": True, "submission": submission}, 201)


@app.get("/api/fitting/submissions")
def api_fitting_submissions():
    return respond({"success": True, "submissions": jobs().fit_submissions(identity()[0])})


@app.get("/api/config")
def api_config():
    component = asdict(Component(symbol="", name=""))
    component_types = {field.name: str(field.type) for field in fields(Component)}
    empty = ProcessFlowDiagram().to_dict()
    empty["metadata"]["process_name"] = "Untitled laboratory"
    return respond(
        {
            "success": True,
            "empty_pfd": empty,
            "component_defaults": component,
            "component_types": component_types,
            "metadata_defaults": asdict(Metadata()),
            "thermo_methods": list(SUPPORTED_METHODS),
            "chart_methods": list(SUPPORTED_METHODS),
            "fluid_phase_models": sorted(FLUID_PHASE_MODELS),
            "phase_behaviors": sorted(PHASE_BEHAVIORS),
            "solid_material_forms": sorted(SOLID_MATERIAL_FORMS),
            "recycle_defaults": RECYCLE_METHOD_DEFAULTS,
            "port_types": [port.value for port in PortType],
            'configuration_fields':configuration_field_catalog(),
            'kinetics':kinetic_input_catalog(),
        }
    )


def apply_layout(pfd,layout):
    """Apply shared layout coordinates to the editable PFD model."""
    for unit in pfd.units:
        unit.x, unit.y = layout['units'][unit.id]['x'],layout['units'][unit.id]['y']
    return pfd


def parsed_response(text):
    pfd = parse_pfd(text)
    errors, warnings = validate_pfd(pfd)
    layout=None
    if pfd.units or pfd.streams:
        layout=layout_flowsheet(pfd)
        apply_layout(pfd,layout)
    return {
        "success": True,
        "pfd": pfd.to_dict(),
        "text": text,
        "errors": errors,
        "warnings": warnings,
        'layout':layout,
    }


@app.post('/api/layout')
def api_layout():
    data=body()
    pfd = ProcessFlowDiagram.from_dict(data.get('pfd',{}))
    # Drawing topology is validated by the shared renderer. Missing simulation
    # conditions must not prevent editing an unfinished feed or unit.
    layout=layout_flowsheet(pfd,keep_positions=bool(data.get('keep_positions',False))) if pfd.units or pfd.streams else {'units':{},'streams':{},'bounds':{'x':0,'y':0,'width':100,'height':100}}
    apply_layout(pfd,layout)
    return respond({'success':True,'pfd':pfd.to_dict(),'layout':layout})


@app.post("/api/parse")
def api_parse():
    return respond(parsed_response(text_input(body())))


@app.post("/api/serialize")
def api_serialize():
    data = body().get("pfd")
    if not isinstance(data, dict):
        raise ValueError("Provide a PFD object.")
    pfd = ProcessFlowDiagram.from_dict(data)
    text = pfd.to_pfd()
    # Serialization must not silently emit text the current parser rejects.
    restored = parse_pfd(text)
    for unit in restored.units:
        original = pfd.get_unit(unit.id)
        unit.x, unit.y = original.x, original.y
    return respond({"success": True, "text": text, "pfd": restored.to_dict()})


@app.post("/api/validate")
def api_validate():
    data = body()
    pfd = (
        parse_pfd(text_input(data))
        if "text" in data
        else ProcessFlowDiagram.from_dict(data.get("pfd", {}))
    )
    errors, warnings = validate_pfd(pfd)
    found, missing = validate_components([c.symbol for c in pfd.components])
    warnings.extend(
        f"Component '{symbol}' is not in the local database; supplied properties or online lookup may be needed."
        for symbol in missing
    )
    dof = analyze_dof(pfd)
    errors.extend(dof.errors)
    warnings.extend(dof.warnings)
    details = {
        "overall_status": dof.overall_status.value,
        "total_dof": dof.total_dof,
        "units": [
            {
                "id": r.entity_id,
                "dof": r.dof,
                "status": r.status.value,
                "message": r.message,
                "details": r.details,
            }
            for r in dof.unit_results
        ],
        "streams": [
            {"id": r.entity_id, "status": r.status.value, "message": r.message}
            for r in dof.stream_results
            if r.status != SpecificationStatus.OK
        ],
        "suggestions": dof.suggestions,
    }
    return respond(
        {
            "success": not errors,
            "errors": errors,
            "warnings": warnings,
            "dof_analysis": details,
            "chemicals_found": found,
            "chemicals_missing": missing,
        }
    )


@app.post("/api/upload")
def api_upload():
    upload = request.files.get("file")
    if not upload or not upload.filename.lower().endswith(".pfd"):
        raise ValueError("Choose a .pfd file.")
    try:
        text = upload.read().decode("utf-8-sig")
    except UnicodeError as error:
        raise ValueError("The file must use UTF-8 text.") from error
    return respond(
        {**parsed_response(text), "filename": secure_filename(upload.filename)}
    )


def download_text(text, filename, extension):
    filename = secure_filename(filename) or f"process.{extension}"
    if not filename.lower().endswith(f".{extension}"):
        filename += f".{extension}"
    return send_file(
        BytesIO(text.encode("utf-8")),
        mimetype="text/plain",
        as_attachment=True,
        download_name=filename,
    )


@app.post("/api/download")
def api_download():
    data = body()
    text = (
        text_input(data)
        if "text" in data
        else ProcessFlowDiagram.from_dict(data.get("pfd", {})).to_pfd()
    )
    return download_text(text, data.get("filename", "process.pfd"), "pfd")


@app.get("/api/examples")
def api_examples_list():
    examples = []
    for path in sorted((BASE_DIR / "examples").glob("*.pfd")):
        try:
            text = path.read_text()
            info = {
                "filename": path.name,
                "name": path.stem.replace("_", " ").title(),
                "thermo_method": "IDEAL",
            }
            for line in text.splitlines():
                if line.startswith("PROCESS:"):
                    info["name"] = line.split(":", 1)[1].strip()
                elif line.startswith("THERMO_METHOD:"):
                    info["thermo_method"] = line.split(":", 1)[1].strip()
            examples.append(info)
        except (OSError, UnicodeError):
            continue
    return respond({"success": True, "examples": examples})


@app.get("/api/examples/<filename>")
def api_example_file(filename):
    if filename != Path(filename).name:
        raise ValueError("Invalid example filename.")
    path = BASE_DIR / "examples" / filename
    if path.suffix != ".pfd" or not path.is_file():
        return respond({"success": False, "error": "Example not found."}, 404)
    return respond({**parsed_response(path.read_text()), "filename": path.name})


@app.get("/api/example")
def api_example():
    return api_example_file("simple_flash.pfd")


@app.get("/api/unit-templates")
def api_unit_templates():
    templates = {}
    for canonical in dict.fromkeys(UNIT_TYPE_ALIASES.values()):
        family = UNIT_PORT_FAMILIES[canonical]
        schema = port_schema_for_unit_type(canonical)
        layout = NUMERIC_PORT_LAYOUTS.get(family)
        names = [name for _, name in layout] if layout else list(schema["types"])
        if family == "mixer":
            names = ["in1", "in2", "out"]
        elif family == "splitter":
            names = ["in", "out", "out2"]
        rules = get_unit_info(canonical)
        required = [
            name
            for spec in rules.get("required_specs", [])
            for name in re.split(r"[|,+]", spec)
            if re.fullmatch(r"[a-zA-Z_][a-zA-Z_0-9]*", name)
        ]
        templates[canonical] = {
            "category": family,
            "supports_reactions": unit_supports_reactions(canonical),
            "ports": [
                {"id": name, "port_type": schema["types"].get(name, "inlet")}
                for name in names
            ],
            "default_params": [],
            "variable_inlets": bool(schema.get("variable_inlets")),
            "variable_outlets": bool(schema.get("variable_outlets")),
            "description": rules.get("description", canonical),
            "specification_notes": rules.get("dof_notes", ""),
            "required_specs": rules.get("required_specs", []),
            "parameter_suggestions": {
                **{name: {} for name in required},
                **rules.get("optional_specs", {}),
            },
            "primary_parameters": list(dict.fromkeys([*required, *rules.get('primary_specs', [])])),
            'settings': unit_setting_schema(canonical),
        }
    return respond(templates)


def submit(kind, payload):
    owner, principal, limit, _ = identity()
    store = jobs()
    try:
        identifier = store.submit(
            kind, payload, owner=owner, principal=principal, cpu_limit=limit
        )
    except RuntimeError as error:
        return respond({"success": False, "error": str(error)}, 429)
    return respond({
        "success": True, "job_id": identifier, "status": "queued",
        "quota": store.quota(principal, limit),
    }, 202)


def flowsheet_payload(pfd):
    identity = json.dumps(pfd.to_dict(), sort_keys=True, separators=(",", ":"))
    return {
        "text": pfd.to_pfd(),
        "fingerprint": hashlib.sha256(identity.encode()).hexdigest(),
    }


@app.post("/api/simulate")
def api_simulate():
    data = body()
    text = text_input(data)
    pfd = parse_pfd(text)
    return submit(
        "simulation",
        {
            **flowsheet_payload(pfd),
            "max_iterations": number(
                data, "max_iterations", 100, minimum=1, maximum=10000, integer=True
            ),
            "tolerance": number(data, "tolerance", 1e-4, minimum=1e-12, maximum=0.1),
        },
    )


@app.get("/api/jobs/<identifier>")
def api_job(identifier):
    job = jobs().get(identifier)
    if job and not owns_job(job):
        job = None
    return respond(
        {
            "success": bool(job),
            "job": job,
            **({} if job else {"error": "Run not found or expired."}),
        },
        200 if job else 404,
    )


@app.post("/api/jobs/<identifier>/cancel")
def api_cancel(identifier):
    job = jobs().get(identifier)
    if job and owns_job(job):
        job = jobs().cancel(identifier)
    else:
        job = None
    return respond({"success": bool(job), "job": job}, 200 if job else 404)


@app.post("/api/simulate/download")
def api_simulate_download():
    data = body()
    job = jobs().get(data.get("job_id", ""))
    if (
        job is None
        or not owns_job(job)
        or job["kind"] != "simulation"
        or job["status"] != "completed"
    ):
        raise ValueError(
            "Provide the ID of a completed simulation; downloads never rerun calculations."
        )
    return download_text(
        job["output"]["pfr_content"], data.get("filename", "results.pfr"), "pfr"
    )


@app.post("/api/vle-chart")
def api_vle_chart():
    data = body()
    chart_type = str(data.get("chart_type", "Txy")).upper()
    if chart_type not in {"TXY", "PXY", "XY", "VLLE", "TERNARY_LLE", "TERNARY_VLLE"}:
        raise ValueError("Choose Txy, Pxy, xy, VLLE, TERNARY_LLE, or TERNARY_VLLE.")
    count = 3 if chart_type.startswith("TERNARY") else 2
    for name in ("comp1", "comp2", "comp3")[:count]:
        if not isinstance(data.get(name), str) or not data[name].strip():
            raise ValueError(f"Choose {count} components.")
    method = str(data.get("method", "UNIFAC")).upper()
    if method not in SUPPORTED_METHODS:
        raise ValueError("Choose a supported thermodynamic method.")
    if method == "STEAM":
        raise ValueError(
            "STEAM supports only water. Binary and ternary diagrams require a mixture-capable model."
        )
    if not isinstance(data.get("online_lookup", True), bool):
        raise ValueError("online_lookup must be true or false.")
    if chart_type in {"VLLE", "TERNARY_LLE", "TERNARY_VLLE"} and not method.startswith(
        ("UNIF", "NRTL", "UNIQUAC")
    ):
        raise ValueError(
            f"{method} does not support liquid-liquid equilibrium; choose an activity model."
        )
    pfd = parse_pfd(text_input(data)) if data.get("text") else ProcessFlowDiagram()
    scope = data.get("scope", "global")
    if scope == "global":
        pfd.metadata.thermo_method = method
        pfd.metadata.thermo_options = data.get(
            "thermo_options", pfd.metadata.thermo_options
        )
    else:
        if data.get("thermo_options"):
            raise ValueError("Additional thermodynamic options are supported only for the global scope.")
        selected_scope = pfd.get_thermo_scope(scope)
        if selected_scope is None:
            raise ValueError(
                f"Thermodynamic scope {scope!r} is not defined in this PFD."
            )
        selected_scope.method = method
    if not data.get("text"):
        pfd.metadata.online_lookup = data.get("online_lookup", True)
    components = []
    for index in range(count):
        identifier = data[f"comp{index + 1}"].strip()
        component = next(
            (
                c
                for c in pfd.components
                if c.symbol == identifier or c.name == identifier
            ),
            None,
        )
        if component is None:
            symbol = f"Atlas_{index + 1}"
            while pfd.get_component(symbol):
                symbol += "_"
            component = Component(symbol=symbol, name=identifier)
            pfd.components.append(component)
        if component.phase_behavior == "permanent_solid":
            raise ValueError("Fluid phase diagrams exclude permanent-solid components.")
        components.append(component.symbol)
    if len(set(components)) != count:
        raise ValueError("Choose distinct components.")
    # Import physical definitions, not the operating specifications of an
    # unfinished process. Diagram calculations do not solve its equipment.
    pfd.units = []
    pfd.streams = []
    pfd.reaction_definitions = []
    pfd.metadata.recycle_tear_streams = []
    pfd.metadata.fluid_phase_model = "VLE"
    pfd = parse_pfd(pfd.to_pfd())
    return submit(
        "chart",
        {
            **flowsheet_payload(pfd),
            "components": components,
            "method": method,
            "scope": scope,
            "chart_type": chart_type,
            "pressure": number(data, "pressure", 1, minimum=0.000001, maximum=10000),
            "temperature": number(
                data, "temperature", 25, minimum=-273.149, maximum=5000
            ),
            "minimum_temperature": number(
                data, "minimum_temperature", 25, minimum=-273.149, maximum=5000
            ),
            "n_points": number(
                data,
                "n_points",
                12 if count == 3 else 50,
                minimum=4 if count == 3 else 10,
                maximum=30 if count == 3 else 200,
                integer=True,
            ),
        },
    )


@app.get("/api/chemicals")
def api_chemicals():
    db = get_database()
    chemicals = [
        {"symbol": p.symbol, "name": p.name, "formula": p.formula, "MW": p.MW}
        for symbol in db.list_all()
        if (p := db.get(symbol)) is not None
    ]
    return respond({"success": True, "chemicals": chemicals, "components": chemicals})


@app.get('/api/vle-chart/components')
def api_chart_suggestions():
    # Suggestions are optional reference identities, never an input whitelist.
    # Read loaded metadata instead of hydrating numerical properties.
    components = {item['identifier']: {**item, 'source': 'Perry'}
                  for item in get_perry_property_library().list_chemicals()}
    for props in get_database().chemicals.values():
        identifier = props.CAS or props.symbol
        existing = components.get(identifier)
        components[identifier] = {'identifier': identifier, 'symbol': props.symbol,
            'CAS': props.CAS, 'name': props.name, 'formula': props.formula,
            'aliases': list(set((existing or {}).get('aliases', [])) | {props.symbol, props.name}),
            'source': 'Perry + chemicals.json' if existing else 'chemicals.json'}
    return respond({'success': True, 'components': sorted(components.values(), key=lambda c: c['name'].casefold())})


@app.get("/api/chemicals/search")
def api_chemical_search():
    query = request.args.get("q", "").strip()
    if not query:
        raise ValueError("Enter a component name or formula.")
    return respond(
        {
            "success": True,
            "results": [
                {"symbol": p.symbol, "name": p.name, "formula": p.formula, "MW": p.MW}
                for p in get_database().search(query)
            ],
        }
    )


@app.get("/api/chemicals/<symbol>")
def api_chemical_detail(symbol):
    props = get_chemical(symbol)
    if props is None:
        return respond({"success": False, "error": "Component not found."}, 404)
    names = (
        "symbol",
        "name",
        "formula",
        "CAS",
        "MW",
        "Tc",
        "Pc",
        "Vc",
        "omega",
        "Tb",
        "Tm",
        "Hf",
        "Gf",
        "Hvap",
        "phase_at_STP",
    )
    details = {name: getattr(props, name, None) for name in names}
    for temperature in (300, 500):
        try:
            details[f"Cp_at_{temperature}K"] = props.Cp(temperature)
        except (ValueError, TypeError):
            details[f"Cp_at_{temperature}K"] = None
    return respond({"success": True, "chemical": details})


@app.post("/api/unifac-groups")
def api_unifac_groups():
    data = body()
    identifier = data.get("component") or data.get("name") or data.get("smiles")
    if not isinstance(identifier, str) or not identifier:
        raise ValueError("Provide a component name or SMILES.")
    return submit(
        "groups",
        {
            "identifier": identifier,
            "smiles": data.get("smiles"),
            "variant": str(
                data.get("variant") or data.get("method") or "UNIFAC"
            ).upper(),
            "online_lookup": data.get("online_lookup", False),
        },
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000)
