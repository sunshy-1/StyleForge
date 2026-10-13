"""Reference-only style schema and prompt compiler.

The schema separates portable positive mechanisms, weak composition
tendencies, and literal reference containers. Only the first two groups may
reach generation wording; literal containers remain audit data.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union


SCHEMA_VERSION = "style_schema"
DEFAULT_ACTUATOR_MODE = "t3_0_dominant_chromatic_field"
T4_COMPACT_CARRIER_AREA_ACTUATOR_MODE = "t4_0_compact_carrier_area"
T4_1_PERCEPTUAL_HUE_CARRIER_ACTUATOR_MODE = "t4_1_perceptual_hue_carrier"
T4_2_SCENE_SLOT_CARRIER_BINDING_ACTUATOR_MODE = "t4_2_scene_slot_carrier_binding"
T4_3_PERCEPTUAL_CARRIER_ANCHOR_ACTUATOR_MODE = "t4_3_perceptual_carrier_anchor"
T4_4_BRIGHT_SCENE_RECAST_ACTUATOR_MODE = "t4_4_bright_scene_recast"
T4_5_CARRIER_MATERIALIZATION_ACTUATOR_MODE = "t4_5_carrier_materialization"
T4_6_COMPACT_TOPOLOGY_ACTUATOR_MODE = "t4_6_compact_topology"
T4_7_BRIGHT_CHROMA_CARRIER_ACTUATOR_MODE = "t4_7_bright_chroma_carrier"
T4_8_ROLE_METADATA_ACTUATOR_MODE = "t4_8_role_metadata"
T4_9_FIELD_PRECEDENCE_ACTUATOR_MODE = "t4_9_field_precedence"
# Keep the prior name as an alias for previously compiled artifacts.
T6_0_PICTURE_LANGUAGE_ACTUATOR_MODE = "t6_0_picture_language"
T5_0_PICTURE_LANGUAGE_ACTUATOR_MODE = T6_0_PICTURE_LANGUAGE_ACTUATOR_MODE
T6_0_SCOPEFIX_SLOTSTABLE_ACTUATOR_MODE = "t6_0_scopefix_slotstable"
T6_0_SCOPEFIX_ANCHORPROTECT_ACTUATOR_MODE = "t6_0_scopefix_anchorprotect"
T6_0_SCOPEFIX_ANCHORPROTECT_MULTIHUE_ACTUATOR_MODE = "t6_0_scopefix_anchorprotect_multihue"

POSITIVE_MECHANISM_KEYS: Tuple[str, ...] = (
    "style_identity",
    "line_rhythm",
    "ornamental_contour",
    "radial_visual_flow",
    "surface_rendering",
    "color_carrier_binding",
    "shape_proportion_language",
    "texture_mark_system",
    "detail_density",
)

HIGH_RISK_COMPILE_ORDER: Tuple[str, ...] = (
    "color_carrier_binding",
    "line_rhythm",
    "radial_visual_flow",
    "surface_rendering",
    "detail_density",
    "texture_mark_system",
    "ornamental_contour",
    "shape_proportion_language",
)

STYLE_FIRST_COMPILE_ORDER: Tuple[str, ...] = (
    "surface_rendering",
    "line_rhythm",
    "shape_proportion_language",
    "texture_mark_system",
    "detail_density",
    "ornamental_contour",
    "radial_visual_flow",
    "color_carrier_binding",
)

REQUIRED_SCHEMA_KEYS: Tuple[str, ...] = (
    "schema_version",
    "task_id",
    "source_policy",
    "positive_style_mechanisms",
    "soft_composition_tendencies",
    "literal_reference_containers",
)

_WS_RE = re.compile(r"\s+")
_CLAUSE_SPLIT_RE = re.compile(r"(?<=[.;])\s+|[;\n]+")
_NUMERIC_RATIO_RE = re.compile(r"\b\d+(?:\.\d+)?\s*(?:%|percent)\b|\b0\.\d+\b", re.IGNORECASE)
_HEX_COLOR_RE = re.compile(r"#[0-9A-Fa-f]{6}\b")
_EMPTY_PROMPT_LABEL_RE = re.compile(
    r"\b(?:edges|overlap/shadow|surface artifacts|partitions|light/shadow binding|surface units|edge/process|primitives|edge profile):\s*(?:;|\.|$)",
    re.IGNORECASE,
)

_DANGLING_PROMPT_TAIL_TOKENS = frozenset(
    {
        "and",
        "or",
        "with",
        "while",
        "selected",
        "for",
        "as",
        "avoid",
        "to",
        "by",
        "on",
        "in",
        "not",
        "material",
        "broken",
        "illuminated",
        "small",
        "subtle",
        "deep",
        "cast",
        "tiny",
        "broad",
        "large",
        "across",
        "hard",
        "strong",
        "visible",
        "fine",
        "sparse",
        "controlled",
        "lamp-lit",
        "they",
        "them",
        "these",
        "those",
        "through",
        "rim",
    }
)

_DANGLING_MODIFIER_TAIL_TOKENS = _DANGLING_PROMPT_TAIL_TOKENS | frozenset(
    {
        "amber",
        "cinematic",
        "cool",
        "crisp",
        "cyan",
        "cyan-teal",
        "glossy",
        "golden",
        "luminous",
        "muted",
        "polished",
        "reflected",
        "simplified",
        "smooth",
        "soft",
        "teal",
        "warm",
    }
)

_SOFT_COMPOSITION_TERMS = {
    "balance",
    "balanced",
    "centered",
    "central",
    "middle distance",
    "shallow",
    "depth",
    "frontal",
    "inward",
    "negative space",
    "luminous middle",
    "hierarchy",
    "layered",
}

_SOFT_POSITIVE_REMOVALS: Tuple[re.Pattern, ...] = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\b(?:build(?: the image)? around|prefer|favor|use)?\s*centered\s+(?:or\s+near-centered\s+)?(?:luminous\s+)?balance(?:\s+with\s+[^,.;]+)?",
        r"\b(?:clear\s+)?central(?:\s+or\s+near-central)?\s+focal\s+hierarchy\b",
        r"\bcentral\s+(?:staged|layered)\s+depth\b",
        r"\bluminous\s+middle\s+distance(?:\s+or\s+backglow)?\b",
        r"\bshallow\s+poster-like\s+depth(?:\s+with\s+[^.;]+)?\b",
        r"\bfrontal\s+poster-view\s+camera\b",
        r"\b(?:inward-leading|inward\s+leading|inward\s+visual)\s+(?:curves|flow)\b",
    )
)

_RADIAL_FLOW_TERMS = (
    "radial",
    "radiating",
    "fan",
    "sweeping",
    "curve",
    "curves",
    "arc",
    "arcs",
    "ribbon",
    "bands",
)

_COLOR_TOKEN_STOPWORDS = {
    "and",
    "around",
    "base",
    "bright",
    "carrier",
    "clear",
    "color",
    "colors",
    "dark",
    "deep",
    "family",
    "light",
    "local",
    "medium",
    "near",
    "only",
    "pale",
    "such",
    "very",
    "warm",
    "with",
}

_GENERIC_LITERAL_PATTERNS: Tuple[re.Pattern, ...] = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\barch(?:ed)?\s+window(?:s)?\b",
        r"\bcircular\s+(?:frame|framing|portal|window|border|container|containment)\b",
        r"\b(?:arched|arch)\s+(?:frame|framing|portal|border|container|containment)\b",
        r"\b(?:arched|arch|arches)\b",
        r"\bcircular\b",
        r"\b(?:frame|frames|framed|framing)\b",
        r"\b(?:border|borders|bordered)\b",
        r"\bportal(?:s)?\b",
        r"\bstage(?:-like)?\b",
        r"\bstaged\b",
        r"\bstaging\b",
        r"\broom(?:s)?\b",
        r"\bpanel(?:s)?\b",
        r"\bwindow(?:s)?\b",
        r"\bchurch(?:es)?\b",
        r"\bscreen(?:s)?\b",
        r"\binterface(?:s)?\b",
        r"\bUI\b",
        r"\bbox(?:es)?\b",
        r"\bdiorama(?:s)?\b",
    )
)

_CONTAINER_REPLACEMENTS: Tuple[Tuple[re.Pattern, str], ...] = tuple(
    (re.compile(pattern, re.IGNORECASE), replacement)
    for pattern, replacement in (
        (
            r"\b(?:large\s+)?(?:centered\s+)?(?:arched|arch)\s+(?:or|and)\s+circular\s+"
            r"(?:framing|frames|frame|containment|containers|container|geometry|geometries|opening|openings)\b",
            "curvilinear radial visual rhythm",
        ),
        (
            r"\b(?:arched|arch)\s+(?:or|and)\s+circular\s+(?:rhythm|visual rhythm)\b",
            "curvilinear radial visual rhythm",
        ),
        (r"\b(?:arched|arch)\s+(?:window|portal|frame|framing|border|container|containment|opening|openings)\b", "curvilinear visual rhythm"),
        (r"\bcircular\s+(?:window|portal|frame|framing|border|container|containment|opening|openings)\b", "radial visual rhythm"),
        (r"\bwindow-like\s+(?:opening|openings|light|lights)\b", "luminous negative-space carrier"),
        (r"\bpatterned\s+border(?:s)?\b", "surface-internal decorative linework"),
        (r"\bornate\s+border(?:s)?\b", "surface-internal decorative linework"),
        (r"\bdecorative\s+border(?:s)?\b", "surface-internal decorative linework"),
        (r"\b(?:frame|frames|framed|framing)\b", "surface-internal contour organization"),
        (r"\b(?:border|borders|bordered)\b", "surface-internal accent linework"),
        (r"\bportal(?:s)?\b", "curved visual transition"),
        (r"\bstage-like\b", "shallow poster-like"),
        (r"\bstaged\b", "layered"),
        (r"\bstaging\b", "visual organization"),
        (r"\bstage\b", "shallow space"),
        (r"\broom(?:s)?\b", "space"),
        (r"\bpanel(?:s)?\b", "planes"),
        (r"\bchurch\s+window(?:s)?\b", "luminous pane system"),
        (r"\bchurch(?:es)?\b", "architectural context"),
        (r"\bwindow(?:s)?\b", "light plane"),
        (r"\bscreen(?:s)?\b", "graphic plane"),
        (r"\binterface(?:s)?\b", "graphic plane"),
        (r"\bUI\b", "graphic plane"),
        (r"\bbox(?:es)?\b", "layered volume"),
        (r"\bdiorama(?:s)?\b", "layered depth"),
    )
)

_HIGH_RISK_MECHANISM_REPLACEMENTS: Tuple[Tuple[re.Pattern, str], ...] = tuple(
    (re.compile(pattern, re.IGNORECASE), replacement)
    for pattern, replacement in (
        (r"\bposter-paper\s+warmth\b", "paper warmth"),
        (r"\blocalized\s+radiating\s+line\s+rhythm\b", "curved local line rhythm"),
        (r"\blocalized\s+radial\s+line\s+rhythm\b", "curved local line rhythm"),
        (r"\bcurvilinear\s+radial\s+visual\s+rhythm\b", "curved flowing line rhythm"),
        (
            r"\bwith\s+semi-realistic\s+dimensional\s+forms\s+simplified\s+into\s+broad\s+clean\s+masses\b",
            "with idealized dimensional forms and broad clean masses",
        ),
        (
            r"\bsemi-realistic\s+dimensional\s+forms\s+simplified\s+into\s+broad\s+clean\s+masses\b",
            "idealized dimensional forms and broad clean masses",
        ),
        (r"\bornamental\b", "decorative"),
        (r"\bposter\s+illustration\b", "illustration"),
        (r"\bposter\b", "illustration"),
        (r"\bcontrolled\s+theatrical\s+highlights\b", "controlled theatrical highlights and smooth glossy reflections"),
        (r"\bouter\s+structure\b", "local shadow structure"),
        (r"\bstructural\s+bands\b", "local material bands"),
        (r"\bradial\s+fan\s+bands\b", "directional material bands"),
        (r"\bradiating\s+fan\s+bands\b", "directional material bands"),
        (r"\bfan\s+bands\b", "directional material bands"),
        (r"\bradiating\s+bands\b", "radiating material bands"),
        (r"\bradial\s+bands\b", "radiating material bands"),
        (r"\bfan\s+tracery\b", "internal branching tracery"),
        (r"\bfan\s+ribs\b", "repeated internal rib lines"),
        (r"\bfan-like\s+internal\s+tracery\b", "internal branching tracery"),
        (r"\bfan-like\s+internal\s+line\s+rhythm\b", "repeated internal line rhythm"),
        (r"\bradial\s+wedges\b", "radiating wedge-shaped color/value fields"),
        (r"\bradial\s+arcs\b", "curved internal line rhythm"),
        (r"\blayered\s+decorative\s+arcs\b", "layered local contour curves"),
        (r"\barched\b", "curved"),
        (r"\barches\b", "curves"),
        (r"\barch\b", "curve"),
        (r"\bcircular\b", "radial"),
        (r"\bedge\s+organization\b", "surface-internal contour organization"),
        (r"\bedge\s+accents\b", "surface-internal accent linework"),
        (r"\bdeliberate\s+(?:[A-Za-z]+\s+){0,3}curls\b", "flowing contour turns inside material fields"),
        (r"\bgold\s+trim\b", "gold linework inside broad material fields"),
        (r"\btrim\b", "material accent linework"),
        (r"\bfiligree\b", "etched linework inside metallic/material carriers"),
        (r"\btracery\b", "internal decorative linework"),
        (r"\bdividers\b", "internal divider lines inside large fields"),
        (r"\bopenings\b", "local light shapes"),
        (r"\bobject\s+planes\b", "object material planes"),
        (r"\bplanes\b", "material planes"),
    )
)

_TITLE_CASE_STYLE_LABEL_RE = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3}\b")

_HIGH_RISK_INTEGRATION_CLAUSE = "Do not copy reference subjects, props, settings, poses, or containers"

_STRUCTURE_APPLICATION_CLAUSE = (
    "Apply extracted shape/proportion and edge rules across subjects, objects, and background; "
    "avoid color-only style overlay"
)

_PROCESS_APPLICATION_CLAUSE = (
    "Make medium/process artifacts visible at image scale on foreground and background carriers"
)

_CONTENT_RECASTING_CLAUSE = (
    "Preserve requested object identities/actions; rebuild surfaces, edges, depth, and light through extracted carriers; "
    "content material words stay subordinate"
)

_COLOR_ROLE_DISCIPLINE_CLAUSE = (
    "role discipline: keep extracted scales; bind roles only to extracted carriers; material/light/local anchors precede dark/base fields; "
    "no generic object color or whole-image tint"
)

_LITERAL_EXTRACTORS: Tuple[Tuple[re.Pattern, str], ...] = tuple(
    (re.compile(pattern, re.IGNORECASE), label)
    for pattern, label in (
        (r"\b(?:arched|arch)\s+window(?:s)?\b", "arched window"),
        (r"\bcircular\s+portal(?:s)?\b", "circular portal"),
        (r"\bcircular\s+(?:frame|framing)\b", "circular frame"),
        (r"\b(?:arched|arch)\s+(?:frame|framing)\b", "arched frame"),
        (r"\b(?:arched|arch)\s+(?:or|and)\s+circular\s+(?:frame|framing|containment)\b", "arched or circular framing"),
        (r"\bpatterned\s+border(?:s)?\b", "patterned border"),
        (r"\bornate\s+border(?:s)?\b", "ornate border"),
        (r"\bdecorative\s+border(?:s)?\b", "decorative border"),
        (r"\b(?:frame|frames|framed|framing)\b", "frame/framing"),
        (r"\b(?:border|borders|bordered)\b", "border-as-object"),
        (r"\bportal(?:s)?\b", "portal"),
        (r"\bstage(?:-like)?\b", "stage"),
        (r"\broom(?:s)?\b", "room"),
        (r"\bpanel(?:s)?\b", "panel"),
        (r"\bwindow(?:s)?\b", "window"),
        (r"\bchurch(?:es)?\b", "church setting"),
        (r"\bscreen(?:s)?\b", "screen"),
        (r"\binterface(?:s)?\b", "interface"),
        (r"\bUI\b", "UI scene"),
        (r"\bbox(?:es)?\b", "box"),
        (r"\bdiorama(?:s)?\b", "diorama"),
    )
)


@dataclass
class StyleSchema:
    task_id: str
    source_policy: str
    positive_style_mechanisms: Dict[str, str]
    soft_composition_tendencies: List[str]
    literal_reference_containers: List[str]
    color_light_allocation: List[Dict[str, Any]] = field(default_factory=list)
    content_carrier_arbitration: List[Dict[str, str]] = field(default_factory=list)
    mechanism_transfer: Dict[str, str] = field(default_factory=dict)
    medium_construction_coverage: Dict[str, str] = field(default_factory=dict)
    style_geometry_grammar: Dict[str, str] = field(default_factory=dict)
    material_process_evidence_ladder: Dict[str, str] = field(default_factory=dict)
    color_value_distribution: Dict[str, Any] = field(default_factory=dict)
    tonal_balance: Dict[str, Any] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION
    audit_notes: List[str] = field(default_factory=list)
    source_files: Dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "StyleSchema":
        _validate_schema_mapping(data)
        positive = data["positive_style_mechanisms"]
        color_light_allocation = _color_light_allocation_from_schema(data.get("color_light_allocation"))
        content_carrier_arbitration = _content_carrier_arbitration_from_schema(data.get("content_carrier_arbitration"))
        if not content_carrier_arbitration:
            content_carrier_arbitration = _build_content_carrier_arbitration(color_light_allocation)
        return cls(
            schema_version=str(data.get("schema_version") or SCHEMA_VERSION),
            task_id=str(data.get("task_id") or "").strip(),
            source_policy=str(data.get("source_policy") or "").strip(),
            positive_style_mechanisms={key: str(positive.get(key) or "").strip() for key in POSITIVE_MECHANISM_KEYS},
            soft_composition_tendencies=[str(item).strip() for item in data.get("soft_composition_tendencies") or [] if str(item).strip()],
            literal_reference_containers=[str(item).strip() for item in data.get("literal_reference_containers") or [] if str(item).strip()],
            color_light_allocation=color_light_allocation,
            content_carrier_arbitration=content_carrier_arbitration,
            mechanism_transfer=_mechanism_transfer_from_schema(data.get("mechanism_transfer")),
            medium_construction_coverage=_medium_construction_coverage_from_schema(data.get("medium_construction_coverage")),
            style_geometry_grammar=_style_geometry_grammar_from_schema(data.get("style_geometry_grammar")),
            material_process_evidence_ladder=_material_process_evidence_ladder_from_schema(data.get("material_process_evidence_ladder")),
            color_value_distribution=_color_value_distribution_from_schema(data.get("color_value_distribution")),
            tonal_balance=_tonal_balance_from_schema(data.get("tonal_balance")),
            audit_notes=[str(item).strip() for item in data.get("audit_notes") or [] if str(item).strip()],
            source_files={str(key): str(value) for key, value in (data.get("source_files") or {}).items()},
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CompiledStyleSchema:
    positive_prompt: str
    soft_prompt: str
    generation_style_prompt: str
    positive_mechanism_keys: List[str]
    literal_reference_containers: List[str]
    actuator_mode: str = DEFAULT_ACTUATOR_MODE
    color_light_allocation: List[Dict[str, Any]] = field(default_factory=list)
    color_light_allocation_prompt: str = ""
    content_carrier_arbitration: List[Dict[str, str]] = field(default_factory=list)
    content_carrier_arbitration_prompt: str = ""
    mechanism_transfer_prompt: str = ""
    medium_construction_coverage_prompt: str = ""
    style_geometry_grammar_prompt: str = ""
    material_process_evidence_prompt: str = ""
    evidence_carrier_binding_prompt: str = ""
    visual_distribution_priority_prompt: str = ""
    dominant_chromatic_field_prompt: str = ""
    scene_carrier_distribution_prompt: str = ""
    carrier_area_actuator_prompt: str = ""
    perceptual_hue_carrier_prompt: str = ""
    scene_slot_carrier_binding_prompt: str = ""
    perceptual_carrier_anchor_prompt: str = ""
    bright_scene_recast_prompt: str = ""
    carrier_materialization_prompt: str = ""
    carrier_topology_prompt: str = ""
    bright_chroma_carrier_prompt: str = ""
    carrier_role_metadata_prompt: str = ""
    field_precedence_prompt: str = ""
    color_value_distribution_prompt: str = ""
    tonal_balance_prompt: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def build_style_schema_from_contract(
    *,
    task_id: str,
    detail_contract: Mapping[str, Any],
    auxiliary_reference_text: str = "",
    color_contract: Optional[Mapping[str, Any]] = None,
    color_light_allocation_override: Optional[Union[Mapping[str, Any], Sequence[Mapping[str, Any]]]] = None,
    color_value_distribution: Optional[Mapping[str, Any]] = None,
    tonal_balance: Optional[Mapping[str, Any]] = None,
    source_files: Optional[Mapping[str, str]] = None,
) -> StyleSchema:
    """Draft a style schema from existing reference-only contract outputs.

    This helper is intentionally generic. It maps known reference-derived
    contract fields into the schema buckets, classifies reference-container wording
    as literal audit data, and sanitizes mechanism text so containers do not
    survive as positive prompt instructions.
    """

    inner = _contract_inner(detail_contract)
    literal_items = _literal_items_from_contract(auxiliary_reference_text, detail_contract)
    soft_items = _soft_tendencies_from_sources(auxiliary_reference_text, inner)

    positive = {
        "style_identity": _style_identity_text(detail_contract, inner),
        "line_rhythm": _first_non_empty(
            inner,
            "line_edge_language",
            "line_and_edge_language",
            default="Reference-derived line rhythm and edge behavior should stay portable and content-neutral.",
        ),
        "ornamental_contour": _mechanism_role_text(
            detail_contract,
            include_terms=("ornament", "tracery", "contour", "divider", "edge", "line"),
            fallback=_first_non_empty(
                inner,
                "generator_safe_wording",
                "line_edge_language",
                "line_and_edge_language",
                default="Use reference-derived ornamental contour and surface-internal contour accents only as portable style behavior.",
            ),
        ),
        "radial_visual_flow": _mechanism_role_text(
            detail_contract,
            include_terms=("radial", "radiating", "fan", "diagonal", "flow", "curve"),
            fallback=_radial_flow_text_from_sources(auxiliary_reference_text, inner),
        ),
        "surface_rendering": _join_non_empty(
            _first_non_empty(inner, "medium_mark_system", "surface_material_mechanism"),
            _first_non_empty(inner, "surface_artifact", "texture_or_artifact_mechanism"),
            default="Preserve reference-derived surface rendering, material behavior, and finish without copying content.",
        ),
        "color_carrier_binding": _color_carrier_text(inner, color_contract, auxiliary_reference_text=auxiliary_reference_text),
        "shape_proportion_language": _first_non_empty(
            inner,
            "shape_proportion_language",
            "surface_material_mechanism",
            default="Preserve reference-derived shape simplification, proportion language, and form abstraction.",
        ),
        "texture_mark_system": _join_non_empty(
            _first_non_empty(inner, "medium_mark_system", "texture_or_artifact_mechanism"),
            _first_non_empty(inner, "surface_artifact"),
            default="Use reference-derived mark, grain, texture, or artifact behavior only where it supports style.",
        ),
        "detail_density": _first_non_empty(
            inner,
            "detail_density_policy",
            "detail_density_and_scale",
            default="Preserve reference-derived detail density: concentrate detail in style-relevant carriers and keep broad forms readable.",
        ),
    }

    sanitized_positive = {
        key: _sanitize_positive_text(value, literal_items, remove_soft=True)
        for key, value in positive.items()
    }

    sanitized_soft = [
        item
        for item in (_sanitize_positive_text(value, literal_items) for value in soft_items)
        if item and not _looks_like_literal_container(value=item)
    ]
    override_allocation = _color_light_allocation_override_from_schema(color_light_allocation_override)
    if color_light_allocation_override is not None and not override_allocation:
        raise ValueError("color_light_allocation_override did not contain any usable hex allocations")
    color_light_allocation = override_allocation or _build_color_light_allocation(
        inner,
        color_contract,
        auxiliary_reference_text=auxiliary_reference_text,
    )
    color_source_note = (
        "color_light_allocation uses region/tool-derived override when supplied; GPT-selected regions choose carriers, pixel tools compute hexes"
        if override_allocation
        else "color_light_allocation comes from reference-derived contract metadata or legacy color text"
    )

    return StyleSchema(
        task_id=task_id,
        source_policy=(
            "reference-only style schema from existing reference-derived contracts; "
            "no generated images, failure notes, test-scene prompts, or task-id branches"
        ),
        positive_style_mechanisms=sanitized_positive,
        soft_composition_tendencies=_dedupe(sanitized_soft)[:6],
        literal_reference_containers=_dedupe(literal_items),
        color_light_allocation=color_light_allocation,
        content_carrier_arbitration=_build_content_carrier_arbitration(color_light_allocation),
        color_value_distribution=_color_value_distribution_from_schema(color_value_distribution),
        tonal_balance=_tonal_balance_from_schema(tonal_balance),
        mechanism_transfer=_build_mechanism_transfer(inner, color_light_allocation),
        medium_construction_coverage=_build_medium_construction_coverage(inner),
        style_geometry_grammar=_build_style_geometry_grammar(inner),
        material_process_evidence_ladder=_build_material_process_evidence_ladder(inner),
        audit_notes=[
            "positive_style_mechanisms may enter positive generation wording",
            color_source_note,
            "color_light_allocation compiles into qualitative hex-bound color/light distribution wording without numeric percentages",
            "content_carrier_arbitration is derived from allocation, carrier geometry, and role scope to prevent semantic-object color expansion",
            "color_value_distribution compiles only qualitative reference-derived value/chroma/hue-family distribution roles",
            "mechanism_transfer compiles into generic content-recasting and carrier-binding wording",
            "medium_construction_coverage compiles into foreground/background medium application coverage",
            "style_geometry_grammar compiles into portable shape, silhouette, partition, and edge grammar",
            "material_process_evidence_ladder compiles into edge, overlap, surface artifact, partition, and light/shadow evidence coverage",
            "tonal_balance compiles into qualitative whole-image value/chroma wording without numeric percentages",
            "soft_composition_tendencies may enter only with weak optional wording",
            "literal_reference_containers are audit/risk data only and must not enter positive prompts",
        ],
        source_files=dict(source_files or {}),
    )


def _normalize_actuator_mode(value: str) -> str:
    lowered = str(value or "").strip().lower().replace(".", "_").replace("-", "_")
    if lowered in ("", "t3", "t3_0", "t3_0_dominant", "t3_0_dominant_chromatic_field"):
        return DEFAULT_ACTUATOR_MODE
    if lowered in ("t4", "t4_0", "t4_0_compact", "t4_0_compact_carrier_area"):
        return T4_COMPACT_CARRIER_AREA_ACTUATOR_MODE
    if lowered in ("t4_1", "t4_1_perceptual", "t4_1_perceptual_hue_carrier"):
        return T4_1_PERCEPTUAL_HUE_CARRIER_ACTUATOR_MODE
    if lowered in ("t4_2", "t4_2_scene_slot", "t4_2_scene_slot_carrier_binding"):
        return T4_2_SCENE_SLOT_CARRIER_BINDING_ACTUATOR_MODE
    if lowered in ("t4_3", "t4_3_perceptual", "t4_3_perceptual_carrier_anchor"):
        return T4_3_PERCEPTUAL_CARRIER_ANCHOR_ACTUATOR_MODE
    if lowered in ("t4_4", "t4_4_bright", "t4_4_bright_scene_recast"):
        return T4_4_BRIGHT_SCENE_RECAST_ACTUATOR_MODE
    if lowered in ("t4_5", "t4_5_carrier", "t4_5_carrier_materialization"):
        return T4_5_CARRIER_MATERIALIZATION_ACTUATOR_MODE
    if lowered in ("t4_6", "t4_6_compact", "t4_6_compact_topology", "t4_6_carrier_topology"):
        return T4_6_COMPACT_TOPOLOGY_ACTUATOR_MODE
    if lowered in ("t4_7", "t4_7_bright", "t4_7_bright_chroma", "t4_7_bright_chroma_carrier"):
        return T4_7_BRIGHT_CHROMA_CARRIER_ACTUATOR_MODE
    if lowered in ("t4_8", "t4_8_role", "t4_8_role_metadata", "t4_8_carrier_role_metadata"):
        return T4_8_ROLE_METADATA_ACTUATOR_MODE
    if lowered in ("t4_9", "t4_9_field", "t4_9_field_precedence", "t4_9_hue_field_precedence"):
        return T4_9_FIELD_PRECEDENCE_ACTUATOR_MODE
    if lowered in (
        "t6",
        "t6_0",
        "t6_0_picture",
        "t6_0_picture_language",
        "t5",
        "t5_0",
        "t5_0_picture",
        "t5_0_picture_language",
        "picture",
        "picture_language",
    ):
        return T6_0_PICTURE_LANGUAGE_ACTUATOR_MODE
    if lowered in (
        "t6_0_scopefix",
        "t6_0_slotstable",
        "t6_0_scopefix_slotstable",
        "t6_slotstable",
        "scopefix",
        "slotstable",
        "scopefix_slotstable",
    ):
        return T6_0_SCOPEFIX_SLOTSTABLE_ACTUATOR_MODE
    if lowered in (
        "t6_0_anchorprotect",
        "t6_0_scopefix_anchorprotect",
        "t6_anchorprotect",
        "anchorprotect",
        "scopefix_anchorprotect",
    ):
        return T6_0_SCOPEFIX_ANCHORPROTECT_ACTUATOR_MODE
    if lowered in (
        "t6_0_anchorprotect_multihue",
        "t6_0_scopefix_anchorprotect_multihue",
        "t6_anchorprotect_multihue",
        "anchorprotect_multihue",
        "scopefix_anchorprotect_multihue",
        "multihue_anchorprotect",
    ):
        return T6_0_SCOPEFIX_ANCHORPROTECT_MULTIHUE_ACTUATOR_MODE
    raise ValueError(f"unsupported style-schema actuator_mode: {value}")


def _effective_budget_for_actuator(max_chars: int, actuator_mode: str) -> int:
    budget = max(180, int(max_chars))
    if actuator_mode == T4_COMPACT_CARRIER_AREA_ACTUATOR_MODE and budget >= 1800:
        return min(budget, max(1500, int(budget * 0.84)))
    if actuator_mode == T4_1_PERCEPTUAL_HUE_CARRIER_ACTUATOR_MODE and budget >= 2000:
        return min(budget, 1850)
    if actuator_mode == T4_2_SCENE_SLOT_CARRIER_BINDING_ACTUATOR_MODE and budget >= 2000:
        return min(budget, 1950)
    if actuator_mode == T4_3_PERCEPTUAL_CARRIER_ANCHOR_ACTUATOR_MODE and budget >= 2100:
        return min(budget, 2050)
    if actuator_mode == T4_4_BRIGHT_SCENE_RECAST_ACTUATOR_MODE and budget >= 2100:
        return min(budget, 2050)
    if actuator_mode == T4_5_CARRIER_MATERIALIZATION_ACTUATOR_MODE and budget >= 2100:
        return min(budget, 2050)
    if actuator_mode == T4_6_COMPACT_TOPOLOGY_ACTUATOR_MODE and budget >= 2100:
        return min(budget, 1700)
    if actuator_mode == T4_7_BRIGHT_CHROMA_CARRIER_ACTUATOR_MODE and budget >= 2100:
        return min(budget, 1850)
    if actuator_mode == T4_8_ROLE_METADATA_ACTUATOR_MODE and budget >= 2100:
        return min(budget, 1880)
    if actuator_mode == T4_9_FIELD_PRECEDENCE_ACTUATOR_MODE and budget >= 2100:
        return min(budget, 1850)
    if actuator_mode in _PICTURE_LANGUAGE_ACTUATOR_MODES:
        # Picture-language generation prompts follow FLUX.2 vendor guidance: a
        # compact natural-language description near the 60-80 word sweet spot, not a
        # long role-metalanguage document. ~720 chars keeps ~5-6 hex-tagged carriers
        # (so the reference's main colors are not truncated) while staying concise.
        return min(budget, 720)
    return budget


def _picture_language_area_phrase(
    item: Mapping[str, Any],
    force_broad: bool = False,
    *,
    slot_stable: bool = False,
    anchor_protect: bool = False,
    strong_broad_light: bool = False,
    demote_background_field: bool = False,
    bounded_broad_material: bool = False,
) -> str:
    """Generic area/placement phrase for a color role, from allocation + scale.

    No style id, hue, or scene input; reads only the reference-derived allocation
    kind and carrier scale so the same rule applies to every style.

    `force_broad` is set by the caller only for a large material role whose hex
    matches the reference's #1 dominant hue family in a high-chroma reference. In
    that case the color genuinely occupies a large area in the reference, so the
    incidental granular-geometry demotion to "selected carrier only" is skipped and
    the broad wording is used. This is reference-driven (not style id) and repairs
    washout where the true dominant saturated color was confined to selected carriers.
    """
    allocation = str(item.get("allocation") or "").lower()
    scale = str(item.get("carrier_scale") or "").lower()
    broad = ("broad" in scale) or ("large" in scale) or ("continuous" in scale)
    # Broad illumination does not recolor objects; material colors require an
    # explicit reference-dominant hue to avoid flooding bounded carriers.
    model_marked_broad = "broad" in scale
    is_material_color = "material color" in allocation
    use_broad = force_broad or (model_marked_broad and not is_material_color)
    if (
        slot_stable
        and "large material-light" in allocation
        and not force_broad
        and _picture_language_slotstable_selected_light_scope(item)
    ):
        use_broad = False
    if "dominant field" in allocation:
        if anchor_protect and demote_background_field:
            return "in background fields"
        return "across large fields"
    if not use_broad and _allocation_item_has_selected_carrier_scope(item):
        if "large material-light" in allocation:
            return "on selected lit carriers"
        if "large material color" in allocation:
            return "on selected material carriers"
    if "large material-light" in allocation:
        if broad or use_broad:
            if anchor_protect and not strong_broad_light:
                return "on broad lit planes"
            return "flooding broad lit surfaces"
        return "on lit surfaces"
    if "large material color" in allocation:
        if broad or use_broad:
            if anchor_protect and bounded_broad_material:
                return "over broad bounded material fields"
            return "over large surfaces"
        return "on sizable surfaces"
    if "medium light" in allocation:
        return "on mid-size lit areas"
    if "structural dark" in allocation:
        return "as deep shadows and outlines"
    if "tiny accent" in allocation:
        return "as small accents"
    if "local material color" in allocation:
        return "as local touches"
    return "on compatible areas"


_PICTURE_ROLE_ORDER = (
    "dominant field",
    "large material-light",
    "large material color",
    "medium light",
    "structural dark",
    "tiny accent",
    "local material color",
)


def _hex_to_hue_family(hex_str: str) -> Tuple[Optional[str], float]:
    """Map a hex color to a coarse hue family and saturation.

    Generic reference-only helper: same hue bands the reference distribution uses,
    so a role's hex can be matched against the reference-dominant hue families. Low
    saturation returns the "neutral" family. No style id or scene input.
    """
    match = re.match(r"#?([0-9A-Fa-f]{6})", str(hex_str or ""))
    if not match:
        return None, 0.0
    value = match.group(1)
    r = int(value[0:2], 16) / 255.0
    g = int(value[2:4], 16) / 255.0
    b = int(value[4:6], 16) / 255.0
    mx = max(r, g, b)
    mn = min(r, g, b)
    delta = mx - mn
    sat = 0.0 if mx <= 1e-6 else delta / mx
    # Low saturation OR low brightness -> neutral. A near-black tile's hue is noise
    # (e.g. #181E1D reads as "cyan" only because of a 6/255 channel wobble); counting
    # it as a chromatic family lets dark tiles falsely claim the reference-dominant hue
    # slot and crowd out genuinely bright saturated carriers. Value floor prevents that.
    if sat < 0.15 or mx < 0.22 or delta <= 1e-6:
        return "neutral", sat
    if mx == r:
        hue = ((g - b) / delta % 6.0) * 60.0
    elif mx == g:
        hue = (((b - r) / delta) + 2.0) * 60.0
    else:
        hue = (((r - g) / delta) + 4.0) * 60.0
    hue %= 360.0
    if hue < 15.0 or hue >= 345.0:
        return "red", sat
    if hue < 30.0:
        return "orange-red", sat
    if hue < 65.0:
        return "orange-gold", sat
    if hue < 85.0:
        return "yellow", sat
    if hue < 165.0:
        return "green", sat
    if hue < 205.0:
        return "cyan", sat
    if hue < 255.0:
        return "blue", sat
    if hue < 305.0:
        return "violet", sat
    return "magenta", sat


_PICTURE_LANGUAGE_ACTUATOR_MODES = frozenset(
    {
        T6_0_PICTURE_LANGUAGE_ACTUATOR_MODE,
        T6_0_SCOPEFIX_SLOTSTABLE_ACTUATOR_MODE,
        T6_0_SCOPEFIX_ANCHORPROTECT_ACTUATOR_MODE,
        T6_0_SCOPEFIX_ANCHORPROTECT_MULTIHUE_ACTUATOR_MODE,
    }
)

_ANCHOR_PROTECT_ACTUATOR_MODES = frozenset(
    {
        T6_0_SCOPEFIX_ANCHORPROTECT_ACTUATOR_MODE,
        T6_0_SCOPEFIX_ANCHORPROTECT_MULTIHUE_ACTUATOR_MODE,
    }
)


def _hex_brightness(hex_str: str) -> float:
    match = re.match(r"#?([0-9A-Fa-f]{6})", str(hex_str or ""))
    if not match:
        return 0.0
    value = match.group(1)
    return max(
        int(value[0:2], 16) / 255.0,
        int(value[2:4], 16) / 255.0,
        int(value[4:6], 16) / 255.0,
    )


def _picture_language_first_hex(item: Mapping[str, Any]) -> str:
    for hex_value in item.get("hexes") or []:
        text = str(hex_value or "")
        if _HEX_COLOR_RE.match(text):
            return text
    return ""


def _picture_language_text(item: Mapping[str, Any]) -> str:
    return " ".join(
        str(item.get(key) or "")
        for key in (
            "allocation",
            "label",
            "carrier",
            "carrier_scale",
            "carrier_geometry",
            "light_role",
            "material_vs_glow_boundary",
            "generator_safe_wording",
        )
    ).lower()


def _picture_language_slotstable_selected_light_scope(item: Mapping[str, Any]) -> bool:
    """Scopefix-compatible light placement for explicit selected/patch carriers."""
    text = _picture_language_text(item)
    return any(term in text for term in ("selected", "patch", "patches"))


def _picture_language_is_dark_value_anchor(item: Mapping[str, Any]) -> bool:
    """Protect broad structural dark/value anchors from compact color-slot loss.

    Generic, reference-only: a role qualifies by allocation/text/hex value, not by
    task id. This keeps black voids, negative fields, leadwork, occlusion, and hard
    shadows available to the picture prompt before broad color/light roles compete
    for the remaining compact slots.
    """
    allocation = str(item.get("allocation") or "").lower()
    text = _picture_language_text(item)
    hex_value = _picture_language_first_hex(item)
    if not hex_value or _hex_brightness(hex_value) >= 0.24:
        return False
    if not (
        "structural dark" in allocation
        or "dominant field" in allocation
        or any(term in text for term in ("black", "dark", "void", "occlusion", "shadow", "negative", "silhouette", "leadwork", "seam"))
    ):
        return False
    if any(term in text for term in ("glint", "sparkle", "pinpoint", "tiny accent")):
        return False
    return any(
        term in text
        for term in (
            "broad",
            "large",
            "continuous",
            "field",
            "fields",
            "void",
            "negative",
            "occlusion",
            "shadow",
            "leadwork",
            "seam",
        )
    )


def _picture_language_is_background_field(item: Mapping[str, Any]) -> bool:
    allocation = str(item.get("allocation") or "").lower()
    if "dominant field" not in allocation:
        return False
    text = _picture_language_text(item)
    return any(
        term in text
        for term in (
            "atmosphere",
            "atmospheric",
            "daylight",
            "sky",
            "open negative",
            "open field",
        )
    )


def _picture_language_hue_family(item: Mapping[str, Any]) -> Optional[str]:
    family, _sat = _hex_to_hue_family(_picture_language_first_hex(item))
    return family


def _picture_language_has_bounded_material_geometry(item: Mapping[str, Any]) -> bool:
    """Detect extracted hard-edged carrier systems without using task ids.

    These terms describe reference-derived carrier geometry: panes, facets, bands,
    strips, panels, plates, wedges, and similar bounded material regions. They let a
    multi-hue material style keep several saturated carrier colors visible while the
    dark anchor still occupies a compact color slot.
    """
    text = _picture_language_text(item)
    return any(
        term in text
        for term in (
            "bounded",
            "pane",
            "panes",
            "panel",
            "panels",
            "facet",
            "facets",
            "wedge",
            "wedges",
            "block",
            "blocks",
            "strip",
            "strips",
            "band",
            "bands",
            "plate",
            "plates",
            "segment",
            "segmented",
            "hard-edged",
            "crisp edge",
            "crisp edges",
        )
    )


def _picture_language_has_multihue_bounded_material_system(
    allocation: Sequence[Mapping[str, Any]], ref_families: Sequence[str]
) -> bool:
    """Return true when the reference asks for several bounded chromatic carriers.

    The guard is intentionally structural and reference-driven: it requires multiple
    reference-dominant hue families, a protected dark/value anchor, and hard-edged
    material carrier geometry across at least three reference hue families. This
    avoids reopening broad atmospheric cyan/yellow fields while preserving multi-color
    pane/facet systems that otherwise lose a chromatic carrier to the dark anchor.
    """
    families = [str(f or "").lower() for f in ref_families if str(f or "").strip()]
    if len(families) < 3:
        return False
    if not any(_picture_language_is_dark_value_anchor(item) for item in allocation):
        return False
    bounded_families = set()
    for item in allocation:
        alloc = str(item.get("allocation") or "").lower()
        if not (
            "dominant field" in alloc
            or "large material" in alloc
            or "medium light" in alloc
        ):
            continue
        family = _picture_language_hue_family(item)
        if not family or family == "neutral" or family not in families:
            continue
        if not _picture_language_has_bounded_material_geometry(item):
            continue
        bounded_families.add(family)
    return len(bounded_families) >= 3


def _picture_language_is_secondary_bounded_material_field(
    item: Mapping[str, Any], ref_families: Sequence[str]
) -> bool:
    families = [str(f or "").lower() for f in ref_families if str(f or "").strip()]
    if len(families) < 2:
        return False
    alloc = str(item.get("allocation") or "").lower()
    if "large material color" not in alloc:
        return False
    family = _picture_language_hue_family(item)
    if not family or family == "neutral" or family == families[0] or family not in families:
        return False
    scale = str(item.get("carrier_scale") or "").lower()
    if not any(term in scale for term in ("broad", "large", "continuous")):
        return False
    return _picture_language_has_bounded_material_geometry(item)


def _picture_language_demote_non_top_background_text(text: str, ref_families: Sequence[str]) -> str:
    """Keep non-top atmospheric color from leading the compact identity clause."""
    clean = _clean_clause(text)
    if not clean or not ref_families:
        return clean
    top_family = str(ref_families[0] or "").lower()
    hue_words = ("red", "orange-red", "orange-gold", "yellow", "green", "cyan", "blue", "violet", "magenta")
    for family in hue_words:
        if family == top_family:
            continue
        family_re = re.escape(family)
        clean = re.sub(
            rf"\bbroad\s+{family_re}\s+(?:daylight|sky|atmospheric|atmosphere|background)\s+(?:fields|field|washes|wash),?\s*",
            "",
            clean,
            flags=re.IGNORECASE,
        )
        clean = re.sub(
            rf"\b(clear\s+)?{family_re}\s+(daylight|sky|atmospheric|atmosphere|background)\s+field\b",
            rf"{family} \2 background field",
            clean,
            flags=re.IGNORECASE,
        )
    clean = re.sub(r"\s+,", ",", clean)
    clean = re.sub(r",\s*,+", ", ", clean)
    return _clean_clause(clean)


def _dedupe_picture_items(items: Sequence[Mapping[str, Any]]) -> List[Mapping[str, Any]]:
    out: List[Mapping[str, Any]] = []
    seen: set = set()
    for item in items:
        label = str(item.get("label") or "").lower()
        allocation = str(item.get("allocation") or "").lower()
        hex_value = _picture_language_first_hex(item).lower()
        key = (label, allocation, hex_value)
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def _render_picture_language_generation_prompt(
    data: "StyleSchema",
    *,
    budget: int,
    literal_items: Sequence[str],
    high_risk: bool,
    slot_stable: bool = False,
    anchor_protect: bool = False,
    anchor_multihue: bool = False,
) -> str:
    """Render the compiled schema as a FLUX.2-spec natural-language style prompt.

    Generic transform: identical rules for every style, reads only the
    reference-derived schema (style_identity, color_light_allocation,
    color_value_distribution, surface/shape/line mechanisms). Follows the vendor
    FLUX.2 prompting guidance: compact natural prose (~60-80 words), most important
    elements first, hex color codes bound to named carriers for precise color, and
    no negative wording. Style transfer, not just palette: medium, line, shape and
    surface character are carried alongside color. Saturation/value emphasis is
    conditioned on each style's own extracted distribution (low-chroma styles are
    not pushed toward saturation). No style-id branch, no generated-failure text.
    """

    def clean(text: str) -> str:
        out = _sanitize_positive_text(text, literal_items, remove_soft=True, high_risk=high_risk)
        out = re.sub(r"\(\s*[/,\s]*\)", "", out)
        return _clean_clause(out)

    def first_words(text: str, n: int) -> str:
        return " ".join(clean(text).split()[:n])

    def first_clauses_words(text: str, clause_n: int, word_n: int) -> str:
        parts = [p.strip() for p in re.split(r"[.;]", str(text or "")) if p.strip()]
        return " ".join(clean(", ".join(parts[:clause_n])).split()[:word_n])

    mech = data.positive_style_mechanisms or {}
    cvd = data.color_value_distribution or {}
    chroma_key = str(cvd.get("chroma_key") or "").lower()
    high_chroma_role = str(cvd.get("high_chroma_role") or "").lower()
    value_key = str(cvd.get("value_key") or "").lower()

    saturated = ("high-chroma" in chroma_key) or ("dominant saturated" in high_chroma_role) or (
        "visible saturated" in high_chroma_role
    )
    near_mono = ("near-monochrome" in chroma_key) or ("restrained" in high_chroma_role) or (
        "neutral-led" in str(cvd.get("chromatic_role") or "").lower()
    )

    # 1. Style character first (medium + line/shape/surface features), front-loaded
    # per FLUX.2 word-order guidance. This carries the non-color style, not just hue.
    # Capped so it can never crowd out the palette clause below.
    identity = first_words(mech.get("style_identity", ""), 22)

    # 2. Palette bound to named carriers WITH hex codes (vendor-recommended precise
    # color). Ordering is reference-driven: roles whose hex matches the reference's
    # actual dominant hue families lead (so the true main color is front-loaded per
    # FLUX.2 guidance AND survives truncation), then other chromatic roles by area,
    # then neutrals. This is generic — keyed on reference pixel distribution + hex,
    # not style id — and fixes main-color loss where the dominant hue lived in a
    # small-area role (e.g. warm accents ordered last then truncated).
    ref_families = [str(f).lower() for f in (cvd.get("dominant_hue_families") or [])]
    allocation = [item for item in (data.color_light_allocation or []) if isinstance(item, Mapping)]
    multihue_bounded_material_system = bool(
        anchor_multihue and _picture_language_has_multihue_bounded_material_system(allocation, ref_families)
    )
    if anchor_protect:
        identity = _picture_language_demote_non_top_background_text(identity, ref_families)

    def _area_rank(item: Mapping[str, Any]) -> int:
        alloc = str(item.get("allocation") or "").lower()
        return next(
            (i for i, kind in enumerate(_PICTURE_ROLE_ORDER) if kind in alloc),
            len(_PICTURE_ROLE_ORDER),
        )

    def _color_priority(item: Mapping[str, Any]) -> Tuple[int, int, int, int]:
        hexes = [str(h) for h in (item.get("hexes") or []) if _HEX_COLOR_RE.match(str(h) or "")]
        family, _sat = _hex_to_hue_family(hexes[0]) if hexes else (None, 0.0)
        area = _area_rank(item)
        if family and family in ref_families:
            if multihue_bounded_material_system:
                alloc = str(item.get("allocation") or "").lower()
                minor_role = ("tiny accent" in alloc) or ("local material color" in alloc)
                return (0, 1 if minor_role else 0, ref_families.index(family), area)
            return (0, ref_families.index(family), area, 0)
        if family and family != "neutral":
            return (1, area, 0, 0)
        return (2, area, 0, 0)

    ordered = sorted(allocation, key=_color_priority)
    if anchor_protect:
        top_family = ref_families[0] if ref_families else ""
        top_hue_items = [
            item
            for item in ordered
            if top_family
            and (lambda family: family == top_family)(
                _hex_to_hue_family(_picture_language_first_hex(item))[0]
            )
        ]
        dark_anchors = [item for item in ordered if _picture_language_is_dark_value_anchor(item)]
        ordered = _dedupe_picture_items(top_hue_items[:3] + dark_anchors[:2] + ordered)
    color_bits: List[str] = []
    emphasis_shown = False
    color_slot_limit = 7 if multihue_bounded_material_system else 6
    for item in ordered[:color_slot_limit]:
        label = clean(str(item.get("label") or ""))
        if not label:
            continue
        hexes = [str(h).upper() for h in (item.get("hexes") or []) if _HEX_COLOR_RE.match(str(h) or "")]
        hex_text = f" {hexes[0]}" if hexes else ""
        alloc = str(item.get("allocation") or "").lower()
        # Reference-driven broad-scope override for tangible material colors: a large
        # material role whose hue is one of the reference's dominant hue families (in a
        # high-chroma reference) genuinely covers a large area, so do not let incidental
        # geometry words confine it to "selected carrier only" (which washes out the
        # true dominant color). Eligibility is any dominant hue family the reference
        # actually shows — the extraction/reference decides how many colors qualify, so
        # the count varies per style instead of a fixed top-N. The anti-recolor/anti-
        # drift demotion stays intact for every material color that is NOT a reference
        # dominant hue (illumination roles are honored separately in the area phrase).
        item_family, _item_sat = _hex_to_hue_family(hexes[0]) if hexes else (None, 0.0)
        broad_families = ref_families[:1] if (anchor_protect or slot_stable) else ref_families
        secondary_bounded_material = bool(
            multihue_bounded_material_system
            and _picture_language_is_secondary_bounded_material_field(item, ref_families)
        )
        force_broad = bool(
            saturated
            and item_family
            and item_family != "neutral"
            and "large material" in alloc
            and ((broad_families and item_family in broad_families) or secondary_bounded_material)
        )
        strong_broad_light = bool(
            anchor_protect
            and item_family
            and ref_families
            and item_family == ref_families[0]
        )
        demote_background_field = bool(
            anchor_protect
            and not _picture_language_is_dark_value_anchor(item)
            and _picture_language_is_background_field(item)
            and item_family
            and ref_families
            and item_family != ref_families[0]
        )
        phrase = _picture_language_area_phrase(
            item,
            force_broad=force_broad,
            slot_stable=slot_stable,
            anchor_protect=anchor_protect,
            strong_broad_light=strong_broad_light,
            demote_background_field=demote_background_field,
            bounded_broad_material=secondary_bounded_material,
        )
        is_big_color = ("dominant field" in alloc) or ("large material" in alloc)
        # State the chroma direction once, up front, then let hex + palette carry it.
        adj = ""
        if is_big_color and not emphasis_shown:
            if saturated:
                adj = "richly saturated "
                emphasis_shown = True
            elif near_mono:
                adj = "desaturated "
                emphasis_shown = True
        color_bits.append(f"{adj}{label}{hex_text} {phrase}")

    # 3. Light/value + chroma key, conditioned on each style's own distribution.
    tone_bits: List[str] = []
    if "low-key" in value_key:
        tone_bits.append("dark low-key lighting")
    elif "high-key" in value_key:
        tone_bits.append("bright high-key lighting")
    elif value_key:
        tone_bits.append("balanced mid-key lighting")
    if saturated:
        tone_bits.append("vivid saturated palette")
    elif near_mono:
        tone_bits.append("near-monochrome palette")
    elif chroma_key:
        tone_bits.append("moderately saturated palette")
    tone = ", ".join(tone_bits)

    # 4. Concise anti-recolor guardrail (positive phrasing). The full disallow-list
    # discipline stays in the structured schema/positive_prompt and its own audit;
    # here we keep only the short steering clause so the prompt stays near the
    # vendor's 60-80 word sweet spot.
    guardrail = "keep each subject's identity and local colors; do not recolor entire semantic objects"

    # Budget allocation: the palette (with hex) is the core of the transfer and must
    # never be dropped, so it is fitted first after reserving the guardrail; identity
    # and tone are added only if room remains. The color list itself is truncated
    # bit-by-bit (keeping the largest carriers) rather than dropped wholesale.
    reserve = len(guardrail) + 2
    identity_reserve = (len(identity) + 2) if identity else 0
    tone_reserve = (len(tone) + 2) if tone else 0
    color_limit = max(80, budget - reserve - identity_reserve - tone_reserve)
    color_clause = ""
    if color_bits:
        acc = "colors: "
        for i, bit in enumerate(color_bits):
            trial = acc + ("" if i == 0 else ", ") + bit
            if len(trial) > color_limit and i >= 2:
                break
            acc = trial
        color_clause = acc

    ordered_parts = [part for part in (identity, color_clause, tone) if part]
    accepted: List[str] = []
    used = 0
    hard_limit = max(120, budget - reserve)
    for part in ordered_parts:
        addition = (len(". ") if accepted else 0) + len(part)
        if used + addition > hard_limit and part != color_clause:
            continue
        accepted.append(part)
        used += addition
    accepted.append(guardrail)
    return _finish_sentence(". ".join(accepted))


def compile_style_schema(
    schema: Union[Mapping[str, Any], StyleSchema],
    *,
    max_chars: int = 1300,
    include_soft_tendencies: bool = True,
    actuator_mode: str = DEFAULT_ACTUATOR_MODE,
) -> CompiledStyleSchema:
    """Compile the style schema into prompt text without literal containers."""

    data = schema if isinstance(schema, StyleSchema) else StyleSchema.from_dict(schema)
    normalized_actuator_mode = _normalize_actuator_mode(actuator_mode)
    requested_budget = max(180, int(max_chars))
    budget = _effective_budget_for_actuator(requested_budget, normalized_actuator_mode)
    literal_items = data.literal_reference_containers
    high_risk = _has_literal_container_risk(literal_items)

    positive_clauses: List[str] = []
    identity_text = _sanitize_positive_text(
        data.positive_style_mechanisms.get("style_identity", ""),
        literal_items,
        remove_soft=True,
        high_risk=high_risk,
    )
    identity_clauses = [
        clause
        for clause in _candidate_clauses(identity_text, split_commas=False)
        if clause and not _looks_like_literal_container(clause)
    ]
    if identity_clauses:
        _append_unique_nonredundant(
            positive_clauses,
            _join_limited(identity_clauses[:1], min(300, max(220, budget // 4)), separator=", "),
        )

    color_light_allocation = data.color_light_allocation or _color_light_allocation_from_text(
        data.positive_style_mechanisms.get("color_carrier_binding", ""),
        source="compiled_color_carrier_fallback",
    )
    content_carrier_arbitration = data.content_carrier_arbitration or _build_content_carrier_arbitration(color_light_allocation)
    content_carrier_arbitration_prompt = _compile_content_carrier_arbitration_prompt(
        content_carrier_arbitration,
        max_chars=min(320, max(290, budget // 4)),
    )
    color_value_distribution_prompt = _compile_color_value_distribution_prompt(
        data.color_value_distribution,
        max_chars=min(560, max(420, budget // 3)),
    )
    visual_distribution_priority_prompt = _compile_visual_distribution_priority_prompt(
        data.color_value_distribution,
        max_chars=min(230, max(170, budget // 6)),
    )
    dominant_chromatic_field_prompt = _compile_dominant_chromatic_field_prompt(
        data.color_value_distribution,
        max_chars=min(250, max(190, budget // 6)),
    )
    scene_carrier_distribution_prompt = _compile_scene_carrier_distribution_prompt(
        data.color_value_distribution,
        max_chars=min(260, max(190, budget // 6)),
    )
    carrier_area_actuator_prompt = ""
    perceptual_hue_carrier_prompt = ""
    scene_slot_carrier_binding_prompt = ""
    perceptual_carrier_anchor_prompt = ""
    bright_scene_recast_prompt = ""
    carrier_materialization_prompt = ""
    carrier_topology_prompt = ""
    bright_chroma_carrier_prompt = ""
    carrier_role_metadata_prompt = ""
    field_precedence_prompt = ""
    bright_chroma_color_prompt = ""
    if normalized_actuator_mode == T4_COMPACT_CARRIER_AREA_ACTUATOR_MODE:
        carrier_area_actuator_prompt = _compile_t4_carrier_area_actuator_prompt(
            data.color_value_distribution,
            max_chars=min(430, max(360, budget // 4)),
        )
    elif normalized_actuator_mode == T4_1_PERCEPTUAL_HUE_CARRIER_ACTUATOR_MODE:
        perceptual_hue_carrier_prompt = _compile_t4_1_perceptual_hue_carrier_prompt(
            data.color_value_distribution,
            max_chars=min(520, max(420, budget // 3)),
        )
    elif normalized_actuator_mode == T4_2_SCENE_SLOT_CARRIER_BINDING_ACTUATOR_MODE:
        scene_slot_carrier_binding_prompt = _compile_t4_2_scene_slot_carrier_binding_prompt(
            data.color_value_distribution,
            max_chars=min(620, max(500, budget // 3)),
        )
    elif normalized_actuator_mode == T4_3_PERCEPTUAL_CARRIER_ANCHOR_ACTUATOR_MODE:
        perceptual_carrier_anchor_prompt = _compile_t4_3_perceptual_carrier_anchor_prompt(
            data.color_value_distribution,
            max_chars=min(560, max(440, budget // 3)),
        )
        scene_slot_carrier_binding_prompt = _compile_t4_2_scene_slot_carrier_binding_prompt(
            data.color_value_distribution,
            max_chars=min(560, max(440, budget // 3)),
        )
    elif normalized_actuator_mode == T4_4_BRIGHT_SCENE_RECAST_ACTUATOR_MODE:
        bright_scene_recast_prompt = _compile_t4_4_bright_scene_recast_prompt(
            data.color_value_distribution,
            max_chars=min(620, max(500, budget // 3)),
        )
        scene_slot_carrier_binding_prompt = _compile_t4_2_scene_slot_carrier_binding_prompt(
            data.color_value_distribution,
            max_chars=min(520, max(420, budget // 3)),
        )
    elif normalized_actuator_mode == T4_5_CARRIER_MATERIALIZATION_ACTUATOR_MODE:
        carrier_materialization_prompt = _compile_t4_5_carrier_materialization_prompt(
            data.color_value_distribution,
            max_chars=min(660, max(540, budget // 3)),
        )
        scene_slot_carrier_binding_prompt = _compile_t4_2_scene_slot_carrier_binding_prompt(
            data.color_value_distribution,
            max_chars=min(520, max(420, budget // 3)),
        )
    elif normalized_actuator_mode == T4_6_COMPACT_TOPOLOGY_ACTUATOR_MODE:
        carrier_topology_prompt = _compile_t4_6_compact_topology_prompt(
            data.color_value_distribution,
            max_chars=min(640, max(600, int(budget * 0.38))),
        )
    elif normalized_actuator_mode == T4_7_BRIGHT_CHROMA_CARRIER_ACTUATOR_MODE:
        bright_chroma_carrier_prompt = _compile_t4_7_bright_chroma_carrier_prompt(
            data.color_value_distribution,
            color_light_allocation,
            max_chars=min(660, max(580, int(budget * 0.38))),
        )
        bright_chroma_color_prompt = _compile_t4_7_bright_chroma_color_prompt(
            data.color_value_distribution,
            color_light_allocation,
            max_chars=min(190, max(160, int(budget * 0.11))),
        )
    elif normalized_actuator_mode == T4_8_ROLE_METADATA_ACTUATOR_MODE:
        carrier_role_metadata_prompt = _compile_t4_8_carrier_role_metadata_prompt(
            data.color_value_distribution,
            color_light_allocation,
            max_chars=min(780, max(700, int(budget * 0.42))),
        )
    elif normalized_actuator_mode == T4_9_FIELD_PRECEDENCE_ACTUATOR_MODE:
        field_precedence_prompt = _compile_t4_9_field_precedence_prompt(
            data.color_value_distribution,
            color_light_allocation,
            max_chars=min(900, max(780, int(budget * 0.50))),
        )
    if normalized_actuator_mode == T4_COMPACT_CARRIER_AREA_ACTUATOR_MODE:
        color_prompt_limit = min(680, max(620, int(budget * 0.38)))
    elif normalized_actuator_mode == T4_1_PERCEPTUAL_HUE_CARRIER_ACTUATOR_MODE:
        color_prompt_limit = min(780, max(700, int(budget * 0.42)))
    elif normalized_actuator_mode == T4_2_SCENE_SLOT_CARRIER_BINDING_ACTUATOR_MODE:
        color_prompt_limit = min(760, max(680, int(budget * 0.40)))
    elif normalized_actuator_mode == T4_3_PERCEPTUAL_CARRIER_ANCHOR_ACTUATOR_MODE:
        color_prompt_limit = min(720, max(620, int(budget * 0.36)))
    elif normalized_actuator_mode == T4_4_BRIGHT_SCENE_RECAST_ACTUATOR_MODE:
        color_prompt_limit = min(700, max(600, int(budget * 0.34)))
    elif normalized_actuator_mode == T4_5_CARRIER_MATERIALIZATION_ACTUATOR_MODE:
        color_prompt_limit = min(700, max(600, int(budget * 0.34)))
    elif normalized_actuator_mode == T4_6_COMPACT_TOPOLOGY_ACTUATOR_MODE:
        color_prompt_limit = min(700, max(620, int(budget * 0.40)))
    elif normalized_actuator_mode == T4_7_BRIGHT_CHROMA_CARRIER_ACTUATOR_MODE:
        color_prompt_limit = min(900, max(820, int(budget * 0.50)))
    elif normalized_actuator_mode == T4_8_ROLE_METADATA_ACTUATOR_MODE:
        color_prompt_limit = min(860, max(780, int(budget * 0.45)))
    elif normalized_actuator_mode == T4_9_FIELD_PRECEDENCE_ACTUATOR_MODE:
        color_prompt_limit = min(840, max(760, int(budget * 0.45)))
    else:
        color_prompt_limit = min(1220, max(1050, int(budget * 0.60)))
    color_value_distribution_for_color_prompt = (
        bright_chroma_color_prompt
        if bright_chroma_color_prompt
        else (
        ""
        if (
            carrier_area_actuator_prompt
            or perceptual_hue_carrier_prompt
            or scene_slot_carrier_binding_prompt
            or perceptual_carrier_anchor_prompt
            or bright_scene_recast_prompt
            or carrier_materialization_prompt
            or carrier_topology_prompt
            or bright_chroma_carrier_prompt
            or carrier_role_metadata_prompt
            or field_precedence_prompt
        )
        else color_value_distribution_prompt
        )
    )
    color_light_allocation_for_prompt = (
        _prepare_t4_9_color_allocation_for_prompt(color_light_allocation, data.color_value_distribution)
        if normalized_actuator_mode == T4_9_FIELD_PRECEDENCE_ACTUATOR_MODE
        else
        _prepare_t4_8_color_allocation_for_prompt(color_light_allocation, data.color_value_distribution)
        if normalized_actuator_mode == T4_8_ROLE_METADATA_ACTUATOR_MODE
        else
        _prepare_t4_7_color_allocation_for_prompt(color_light_allocation, data.color_value_distribution)
        if normalized_actuator_mode == T4_7_BRIGHT_CHROMA_CARRIER_ACTUATOR_MODE
        else color_light_allocation
    )
    color_light_prompt = _compile_color_light_allocation_prompt(
        color_light_allocation_for_prompt,
        content_carrier_arbitration=content_carrier_arbitration,
        content_carrier_arbitration_prompt=content_carrier_arbitration_prompt,
        color_value_distribution_prompt=color_value_distribution_for_color_prompt,
        high_risk=high_risk,
        max_chars=color_prompt_limit,
    )
    tonal_balance_prompt = _compile_tonal_balance_prompt(
        data.tonal_balance,
        max_chars=min(220, max(160, budget // 7)),
    )
    mechanism_transfer_prompt = _compile_mechanism_transfer_prompt(
        data.mechanism_transfer,
        color_light_allocation,
        literal_items=literal_items,
        max_chars=min(300, max(240, budget // 5)),
    )
    medium_construction_coverage_prompt = _compile_medium_construction_coverage_prompt(
        data.medium_construction_coverage,
        literal_items=literal_items,
        max_chars=min(280, max(210, budget // 5)),
    )
    style_geometry_grammar_prompt = _compile_style_geometry_grammar_prompt(
        data.style_geometry_grammar,
        literal_items=literal_items,
        max_chars=min(260, max(200, budget // 6)),
    )
    material_process_evidence_prompt = _compile_material_process_evidence_prompt(
        data.material_process_evidence_ladder,
        literal_items=literal_items,
        max_chars=min(240, max(200, budget // 6)),
    )
    evidence_carrier_binding_prompt = _compile_evidence_carrier_binding_prompt(
        data.material_process_evidence_ladder,
        data.medium_construction_coverage,
        data.style_geometry_grammar,
        content_carrier_arbitration,
        literal_items=literal_items,
        max_chars=min(240, max(190, budget // 6)),
    )

    for key in STYLE_FIRST_COMPILE_ORDER:
        if key == "style_identity":
            continue
        if key == "color_carrier_binding" and color_light_prompt:
            continue
        text = _sanitize_positive_text(
            data.positive_style_mechanisms.get(key, ""),
            literal_items,
            remove_soft=True,
            high_risk=high_risk,
        )
        if _is_noop_mechanism_text(text):
            continue
        split_commas = False
        key_clauses = [
            clause
            for clause in _candidate_clauses(text, split_commas=split_commas)
            if not _looks_like_literal_container(clause)
        ]
        if not key_clauses:
            continue
        key_limit = _mechanism_key_limit(key, high_risk=high_risk)
        clause_count = _mechanism_clause_count(key, high_risk=high_risk)
        if key == "color_carrier_binding":
            key_text = _compile_color_carrier_clauses(
                key_clauses[:clause_count],
                key_limit,
                high_risk=high_risk,
            )
        else:
            key_text = _join_limited(key_clauses[:clause_count], key_limit, separator=", ")
        _append_unique_nonredundant(positive_clauses, key_text)

    application_clauses: List[str] = []
    _append_unique_nonredundant(application_clauses, _STRUCTURE_APPLICATION_CLAUSE)
    if field_precedence_prompt:
        _append_unique_nonredundant(application_clauses, field_precedence_prompt)
    elif carrier_topology_prompt:
        _append_unique_nonredundant(application_clauses, carrier_topology_prompt)
    elif carrier_role_metadata_prompt:
        _append_unique_nonredundant(application_clauses, carrier_role_metadata_prompt)
    elif bright_chroma_carrier_prompt:
        _append_unique_nonredundant(application_clauses, bright_chroma_carrier_prompt)
    elif carrier_materialization_prompt:
        _append_unique_nonredundant(application_clauses, carrier_materialization_prompt)
        _append_unique_nonredundant(application_clauses, scene_slot_carrier_binding_prompt)
    elif bright_scene_recast_prompt:
        _append_unique_nonredundant(application_clauses, bright_scene_recast_prompt)
        _append_unique_nonredundant(application_clauses, scene_slot_carrier_binding_prompt)
    elif perceptual_carrier_anchor_prompt:
        _append_unique_nonredundant(application_clauses, perceptual_carrier_anchor_prompt)
        _append_unique_nonredundant(application_clauses, scene_slot_carrier_binding_prompt)
    elif scene_slot_carrier_binding_prompt:
        _append_unique_nonredundant(application_clauses, scene_slot_carrier_binding_prompt)
    elif perceptual_hue_carrier_prompt:
        _append_unique_nonredundant(application_clauses, perceptual_hue_carrier_prompt)
    elif carrier_area_actuator_prompt:
        _append_unique_nonredundant(application_clauses, carrier_area_actuator_prompt)
    else:
        _append_unique_nonredundant(application_clauses, visual_distribution_priority_prompt)
        _append_unique_nonredundant(application_clauses, dominant_chromatic_field_prompt)
        _append_unique_nonredundant(application_clauses, scene_carrier_distribution_prompt)
    _append_unique_nonredundant(application_clauses, _PROCESS_APPLICATION_CLAUSE)
    _append_unique_nonredundant(application_clauses, medium_construction_coverage_prompt)
    _append_unique_nonredundant(application_clauses, style_geometry_grammar_prompt)
    _append_unique_nonredundant(application_clauses, material_process_evidence_prompt)
    _append_unique_nonredundant(application_clauses, evidence_carrier_binding_prompt)
    _append_unique_nonredundant(application_clauses, mechanism_transfer_prompt)
    _append_unique_nonredundant(application_clauses, tonal_balance_prompt)
    if high_risk:
        _append_unique_nonredundant(application_clauses, _HIGH_RISK_INTEGRATION_CLAUSE)

    color_clause = ""
    if color_light_prompt:
        color_clause = _join_limited(
            [color_light_prompt],
            color_prompt_limit,
        )
    color_reserve = len(color_clause) + 2 if color_clause else 0
    core_budget = max(180, budget - color_reserve)
    mechanism_floor = 0
    if positive_clauses and core_budget >= 360:
        if normalized_actuator_mode in (
            T4_7_BRIGHT_CHROMA_CARRIER_ACTUATOR_MODE,
            T4_8_ROLE_METADATA_ACTUATOR_MODE,
            T4_9_FIELD_PRECEDENCE_ACTUATOR_MODE,
        ):
            mechanism_floor = min(180, max(140, core_budget // 6))
        else:
            mechanism_floor = min(260, max(180, core_budget // 4))
    application_budget = max(120, core_budget - mechanism_floor)
    application_text = _compile_application_text(
        (
            _STRUCTURE_APPLICATION_CLAUSE,
            field_precedence_prompt,
            carrier_topology_prompt,
            carrier_role_metadata_prompt,
            bright_chroma_carrier_prompt,
            carrier_materialization_prompt,
            bright_scene_recast_prompt,
            perceptual_carrier_anchor_prompt,
            scene_slot_carrier_binding_prompt,
            perceptual_hue_carrier_prompt,
            carrier_area_actuator_prompt,
            ""
            if (
                carrier_area_actuator_prompt
                or perceptual_hue_carrier_prompt
                or scene_slot_carrier_binding_prompt
                or perceptual_carrier_anchor_prompt
                or bright_scene_recast_prompt
                or carrier_materialization_prompt
                or carrier_topology_prompt
                or bright_chroma_carrier_prompt
                or carrier_role_metadata_prompt
                or field_precedence_prompt
            )
            else visual_distribution_priority_prompt,
            ""
            if (
                carrier_area_actuator_prompt
                or perceptual_hue_carrier_prompt
                or scene_slot_carrier_binding_prompt
                or perceptual_carrier_anchor_prompt
                or bright_scene_recast_prompt
                or carrier_materialization_prompt
                or carrier_topology_prompt
                or bright_chroma_carrier_prompt
                or carrier_role_metadata_prompt
                or field_precedence_prompt
            )
            else dominant_chromatic_field_prompt,
            ""
            if (
                carrier_area_actuator_prompt
                or perceptual_hue_carrier_prompt
                or scene_slot_carrier_binding_prompt
                or perceptual_carrier_anchor_prompt
                or bright_scene_recast_prompt
                or carrier_materialization_prompt
                or carrier_topology_prompt
                or bright_chroma_carrier_prompt
                or carrier_role_metadata_prompt
                or field_precedence_prompt
            )
            else scene_carrier_distribution_prompt,
            _PROCESS_APPLICATION_CLAUSE,
            medium_construction_coverage_prompt,
            style_geometry_grammar_prompt,
            material_process_evidence_prompt,
            evidence_carrier_binding_prompt,
            mechanism_transfer_prompt,
            tonal_balance_prompt,
            _HIGH_RISK_INTEGRATION_CLAUSE if high_risk else "",
        ),
        application_budget,
    )
    application_reserve = len(application_text) + 2 if application_text else 0
    mechanism_budget = max(0, core_budget - application_reserve)
    mechanism_prompt = _join_limited(positive_clauses, mechanism_budget) if mechanism_budget >= 120 else ""
    positive_prompt = _join_limited(
        [part for part in (mechanism_prompt, application_text) if part],
        core_budget,
    )
    if color_clause:
        positive_prompt = _join_limited(
            [part for part in (positive_prompt, color_clause) if part],
            budget,
            separator="; ",
        )

    soft_prompt = ""
    if include_soft_tendencies and not high_risk:
        soft_clauses: List[str] = []
        for tendency in data.soft_composition_tendencies:
            clean = _sanitize_positive_text(tendency, literal_items)
            if not clean or _looks_like_literal_container(clean):
                continue
            clean = _strip_optional_prefix(clean)
            if clean:
                _append_unique(soft_clauses, clean)
        if soft_clauses:
            soft_body = _join_limited(soft_clauses, max(120, budget // 3))
            soft_prompt = f"Optional, only when it supports the requested content: {soft_body}"

    generation_style_prompt = _join_limited(
        [part for part in (positive_prompt, soft_prompt) if part],
        budget,
        separator=". ",
    )
    generation_style_prompt = _finish_sentence(generation_style_prompt)
    if normalized_actuator_mode in _PICTURE_LANGUAGE_ACTUATOR_MODES:
        # Keep the structured positive_prompt (schema/audit reference) intact, but
        # generate from a compact picture-language description instead of the
        # role-metalanguage form the distilled generator does not follow.
        picture_prompt = _render_picture_language_generation_prompt(
            data,
            budget=budget,
            literal_items=literal_items,
            high_risk=high_risk,
            slot_stable=normalized_actuator_mode == T6_0_SCOPEFIX_SLOTSTABLE_ACTUATOR_MODE,
            anchor_protect=normalized_actuator_mode in _ANCHOR_PROTECT_ACTUATOR_MODES,
            anchor_multihue=normalized_actuator_mode == T6_0_SCOPEFIX_ANCHORPROTECT_MULTIHUE_ACTUATOR_MODE,
        )
        if picture_prompt:
            generation_style_prompt = picture_prompt
    positive_prompt = _finish_sentence(positive_prompt)
    soft_prompt = _finish_sentence(soft_prompt) if soft_prompt else ""

    return CompiledStyleSchema(
        positive_prompt=positive_prompt,
        soft_prompt=soft_prompt,
        generation_style_prompt=generation_style_prompt,
        positive_mechanism_keys=list(POSITIVE_MECHANISM_KEYS),
        literal_reference_containers=list(literal_items),
        actuator_mode=normalized_actuator_mode,
        color_light_allocation=list(color_light_allocation),
        color_light_allocation_prompt=color_light_prompt,
        content_carrier_arbitration=list(content_carrier_arbitration),
        content_carrier_arbitration_prompt=content_carrier_arbitration_prompt,
        mechanism_transfer_prompt=mechanism_transfer_prompt,
        medium_construction_coverage_prompt=medium_construction_coverage_prompt,
        style_geometry_grammar_prompt=style_geometry_grammar_prompt,
        material_process_evidence_prompt=material_process_evidence_prompt,
        evidence_carrier_binding_prompt=evidence_carrier_binding_prompt,
        visual_distribution_priority_prompt=visual_distribution_priority_prompt,
        dominant_chromatic_field_prompt=dominant_chromatic_field_prompt,
        scene_carrier_distribution_prompt=scene_carrier_distribution_prompt,
        carrier_area_actuator_prompt=carrier_area_actuator_prompt,
        perceptual_hue_carrier_prompt=perceptual_hue_carrier_prompt,
        scene_slot_carrier_binding_prompt=scene_slot_carrier_binding_prompt,
        perceptual_carrier_anchor_prompt=perceptual_carrier_anchor_prompt,
        bright_scene_recast_prompt=bright_scene_recast_prompt,
        carrier_materialization_prompt=carrier_materialization_prompt,
        carrier_topology_prompt=carrier_topology_prompt,
        bright_chroma_carrier_prompt=bright_chroma_carrier_prompt,
        carrier_role_metadata_prompt=carrier_role_metadata_prompt,
        field_precedence_prompt=field_precedence_prompt,
        color_value_distribution_prompt=color_value_distribution_prompt,
        tonal_balance_prompt=tonal_balance_prompt,
    )


def audit_style_schema(
    schema: Union[Mapping[str, Any], StyleSchema],
    compiled: Optional[CompiledStyleSchema] = None,
    *,
    max_chars: int = 1300,
) -> Dict[str, Any]:
    """Return a prompt-only audit for the schema and compiled prompt text."""

    errors: List[str] = []
    warnings: List[str] = []
    try:
        data = schema if isinstance(schema, StyleSchema) else StyleSchema.from_dict(schema)
    except ValueError as exc:
        return {"ok": False, "errors": [str(exc)]}

    compiled = compiled or compile_style_schema(data, max_chars=max_chars)
    literal_items = data.literal_reference_containers
    # Picture-language mode renders the generation prompt as a compact natural
    # scene description, so the abstract format-header "reach generation prompt"
    # checks do not apply; the structured positive_prompt still carries and audits
    # the full reference-only contract.
    is_picture = compiled.actuator_mode in _PICTURE_LANGUAGE_ACTUATOR_MODES
    prompt_hits = {
        "positive_prompt": literal_reference_hits(compiled.positive_prompt, literal_items),
        "generation_style_prompt": literal_reference_hits(compiled.generation_style_prompt, literal_items),
    }
    for label, hits in prompt_hits.items():
        if hits:
            errors.append(f"{label} contains literal reference container wording: {', '.join(hits)}")

    if data.task_id and data.task_id.lower() in compiled.generation_style_prompt.lower():
        errors.append("compiled prompt contains task id")
    if _NUMERIC_RATIO_RE.search(compiled.generation_style_prompt):
        errors.append("compiled prompt contains numeric ratio or percentage")
    if _EMPTY_PROMPT_LABEL_RE.search(compiled.generation_style_prompt):
        errors.append("compiled prompt contains empty evidence label")
    if len(compiled.generation_style_prompt) > max_chars + 1:
        errors.append(f"compiled prompt too long: {len(compiled.generation_style_prompt)} > {max_chars}")
    if not compiled.positive_prompt.strip():
        errors.append("compiled positive prompt is empty")
    if len(data.literal_reference_containers) == 0:
        errors.append("literal_reference_containers is empty; schema should carry audit/risk data")
    positive_keys = set(data.positive_style_mechanisms.keys())
    required_positive = set(POSITIVE_MECHANISM_KEYS) - {"style_identity"}
    if not required_positive.issubset(positive_keys) or not positive_keys.issubset(set(POSITIVE_MECHANISM_KEYS)):
        errors.append("positive_style_mechanisms keys do not match the style schema")
    allocation_hexes = _dedupe_hexes(
        hex_value
        for item in compiled.color_light_allocation
        for hex_value in (item.get("hexes") or [])
    )
    if allocation_hexes and not compiled.color_light_allocation_prompt:
        errors.append("color_light_allocation exists but did not compile into prompt-ready wording")
    if data.mechanism_transfer and not compiled.mechanism_transfer_prompt:
        errors.append("mechanism_transfer exists but did not compile into prompt-ready wording")
    if data.medium_construction_coverage and not compiled.medium_construction_coverage_prompt:
        errors.append("medium_construction_coverage exists but did not compile into prompt-ready wording")
    elif data.medium_construction_coverage and not is_picture and "medium construction:" not in compiled.generation_style_prompt.lower():
        errors.append("medium_construction_coverage exists but did not reach generation prompt")
    if data.style_geometry_grammar and not compiled.style_geometry_grammar_prompt:
        errors.append("style_geometry_grammar exists but did not compile into prompt-ready wording")
    elif data.style_geometry_grammar and not is_picture and "geometry grammar:" not in compiled.generation_style_prompt.lower():
        errors.append("style_geometry_grammar exists but did not reach generation prompt")
    if data.material_process_evidence_ladder and not compiled.material_process_evidence_prompt:
        errors.append("material_process_evidence_ladder exists but did not compile into prompt-ready wording")
    elif data.material_process_evidence_ladder and not is_picture and "material process evidence:" not in compiled.generation_style_prompt.lower():
        errors.append("material_process_evidence_ladder exists but did not reach generation prompt")
    if (
        data.material_process_evidence_ladder
        and data.medium_construction_coverage
        and not compiled.evidence_carrier_binding_prompt
    ):
        errors.append("evidence_carrier_binding exists but did not compile into prompt-ready wording")
    elif compiled.evidence_carrier_binding_prompt and not is_picture and "evidence carriers:" not in compiled.generation_style_prompt.lower():
        errors.append("evidence_carrier_binding exists but did not reach generation prompt")
    generation_prompt_lower = compiled.generation_style_prompt.lower()
    color_value_distribution_reaches_prompt = "color/value distribution:" in generation_prompt_lower
    if (
        not color_value_distribution_reaches_prompt
        and compiled.actuator_mode == T4_2_SCENE_SLOT_CARRIER_BINDING_ACTUATOR_MODE
        and compiled.scene_slot_carrier_binding_prompt
        and "scene-slot carrier binding:" in generation_prompt_lower
        and "visual distribution lock:" in generation_prompt_lower
        and "scene carrier map:" in generation_prompt_lower
    ):
        color_value_distribution_reaches_prompt = True
    if (
        not color_value_distribution_reaches_prompt
        and compiled.actuator_mode == T4_3_PERCEPTUAL_CARRIER_ANCHOR_ACTUATOR_MODE
        and compiled.perceptual_carrier_anchor_prompt
        and "perceptual carrier anchors:" in generation_prompt_lower
        and "visual distribution lock:" in generation_prompt_lower
        and "scene carrier map:" in generation_prompt_lower
    ):
        color_value_distribution_reaches_prompt = True
    if (
        not color_value_distribution_reaches_prompt
        and compiled.actuator_mode == T4_4_BRIGHT_SCENE_RECAST_ACTUATOR_MODE
        and compiled.bright_scene_recast_prompt
        and "bright-scene recast:" in generation_prompt_lower
        and "visual distribution lock:" in generation_prompt_lower
        and "scene carrier map:" in generation_prompt_lower
    ):
        color_value_distribution_reaches_prompt = True
    if (
        not color_value_distribution_reaches_prompt
        and compiled.actuator_mode == T4_5_CARRIER_MATERIALIZATION_ACTUATOR_MODE
        and compiled.carrier_materialization_prompt
        and "carrier materialization:" in generation_prompt_lower
        and "visual distribution lock:" in generation_prompt_lower
        and "bounded carrier geometry:" in generation_prompt_lower
    ):
        color_value_distribution_reaches_prompt = True
    if (
        not color_value_distribution_reaches_prompt
        and compiled.actuator_mode == T4_6_COMPACT_TOPOLOGY_ACTUATOR_MODE
        and compiled.carrier_topology_prompt
        and "carrier topology:" in generation_prompt_lower
        and "visual distribution lock:" in generation_prompt_lower
        and "large carrier slots" in generation_prompt_lower
    ):
        color_value_distribution_reaches_prompt = True
    if (
        not color_value_distribution_reaches_prompt
        and compiled.actuator_mode == T4_7_BRIGHT_CHROMA_CARRIER_ACTUATOR_MODE
        and compiled.bright_chroma_carrier_prompt
        and "bright chroma carriers:" in generation_prompt_lower
        and "visual distribution lock:" in generation_prompt_lower
        and "large carrier slots" in generation_prompt_lower
    ):
        color_value_distribution_reaches_prompt = True
    if (
        not color_value_distribution_reaches_prompt
        and compiled.actuator_mode == T4_8_ROLE_METADATA_ACTUATOR_MODE
        and compiled.carrier_role_metadata_prompt
        and "carrier role metadata:" in generation_prompt_lower
        and "visual distribution lock:" in generation_prompt_lower
        and "large carrier slots" in generation_prompt_lower
    ):
        color_value_distribution_reaches_prompt = True
    if (
        not color_value_distribution_reaches_prompt
        and compiled.actuator_mode == T4_9_FIELD_PRECEDENCE_ACTUATOR_MODE
        and compiled.field_precedence_prompt
        and "field precedence:" in generation_prompt_lower
        and "visual distribution lock:" in generation_prompt_lower
        and "largest compatible visible carrier fields" in generation_prompt_lower
    ):
        color_value_distribution_reaches_prompt = True
    if data.color_value_distribution and not compiled.color_value_distribution_prompt:
        errors.append("color_value_distribution exists but did not compile into prompt-ready wording")
    elif data.color_value_distribution and not is_picture and not color_value_distribution_reaches_prompt:
        errors.append("color_value_distribution exists but did not reach generation prompt")
    if data.color_value_distribution and not compiled.visual_distribution_priority_prompt:
        errors.append("color_value_distribution exists but did not compile into priority visual distribution wording")
    elif data.color_value_distribution and not is_picture and "visual distribution lock:" not in compiled.generation_style_prompt.lower():
        errors.append("priority visual distribution wording did not reach generation prompt")
    if _dominant_chromatic_field_required(data.color_value_distribution) and not is_picture:
        if not compiled.dominant_chromatic_field_prompt:
            errors.append("color_value_distribution requires dominant chromatic field wording but did not compile it")
        elif (
            "dominant chromatic field:" not in compiled.generation_style_prompt.lower()
            and not (
                compiled.actuator_mode == T4_6_COMPACT_TOPOLOGY_ACTUATOR_MODE
                and "carrier topology:" in generation_prompt_lower
                and "large carrier slots" in generation_prompt_lower
            )
            and not (
                compiled.actuator_mode == T4_7_BRIGHT_CHROMA_CARRIER_ACTUATOR_MODE
                and "bright chroma carriers:" in generation_prompt_lower
                and "large carrier slots" in generation_prompt_lower
            )
            and not (
                compiled.actuator_mode == T4_8_ROLE_METADATA_ACTUATOR_MODE
                and "carrier role metadata:" in generation_prompt_lower
                and "large carrier slots" in generation_prompt_lower
            )
            and not (
                compiled.actuator_mode == T4_9_FIELD_PRECEDENCE_ACTUATOR_MODE
                and "field precedence:" in generation_prompt_lower
                and "largest compatible visible carrier fields" in generation_prompt_lower
            )
        ):
            errors.append("dominant chromatic field wording did not reach generation prompt")
    if data.color_value_distribution and not compiled.scene_carrier_distribution_prompt:
        errors.append("color_value_distribution exists but did not compile into scene carrier distribution wording")
    elif (
        data.color_value_distribution
        and not is_picture
        and "scene carrier map:" not in compiled.generation_style_prompt.lower()
        and not (
            compiled.actuator_mode == T4_9_FIELD_PRECEDENCE_ACTUATOR_MODE
            and "field precedence:" in generation_prompt_lower
            and "carrier field map:" in generation_prompt_lower
        )
    ):
        errors.append("scene carrier distribution wording did not reach generation prompt")
    high_risk = _has_literal_container_risk(literal_items)
    if high_risk:
        if _HEX_COLOR_RE.search(data.positive_style_mechanisms.get("color_carrier_binding", "")) and not allocation_hexes:
            errors.append("high-risk hex color carrier binding lacks color_light_allocation metadata")
    content_carrier_arbitration = (
        compiled.content_carrier_arbitration
        or data.content_carrier_arbitration
        or _build_content_carrier_arbitration(compiled.color_light_allocation)
    )
    color_role_prompt_for_audit = (
        compiled.generation_style_prompt
        if (
            compiled.actuator_mode == T4_9_FIELD_PRECEDENCE_ACTUATOR_MODE
            and compiled.field_precedence_prompt
        )
        else compiled.color_light_allocation_prompt or compiled.generation_style_prompt
    )
    color_role_coverage = _audit_color_role_coverage(
        compiled.color_light_allocation,
        color_role_prompt_for_audit,
    )
    errors.extend(color_role_coverage.get("errors") or [])
    warnings.extend(color_role_coverage.get("warnings") or [])
    content_carrier_arbitration_audit = _audit_content_carrier_arbitration(
        compiled.color_light_allocation,
        content_carrier_arbitration,
        compiled.generation_style_prompt,
        require_prompt_terms=not is_picture,
    )
    errors.extend(content_carrier_arbitration_audit.get("errors") or [])
    warnings.extend(content_carrier_arbitration_audit.get("warnings") or [])

    return {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "rules": [
            "positive_style_mechanisms compile into positive prompt",
            "color_light_allocation compiles qualitative dominant/large/medium/structural hex roles into prompt-ready wording",
            "soft_composition_tendencies compile only as optional weak wording, and are suppressed when literal-container risk is present",
            "literal_reference_containers remain audit-only and never enter compiled positive wording",
            "no task-id-specific prompt branches",
            "no generated images, failure notes, or test-scene prompts are used",
            "no numeric color ratios or percentages in prompt-ready text",
            "selected/accent/local color roles include content-carrier arbitration against whole semantic-object recoloring",
            "reference color/value distribution roles compile as qualitative constraints without numeric ratios",
            "reference visual distribution lock is front-loaded from generic value/chroma/hue-family roles, not style ids or generated failures",
            "scene-slot carrier binding maps reference high-value/chromatic roles to generic requested-scene carrier slots without style-id branches",
            "perceptual carrier anchors bind reference high-value hue families to broad visible scene carriers before small accents",
            "bright-scene recast promotes reference high-value hue families into generic lighting structure and scene-plane carriers",
            "carrier materialization maps reference-dominant hue families to bounded visible carrier geometry and material evidence",
            "compact carrier topology front-loads large carrier slots, anti-strip collapse, and dark/base support limits",
            "field precedence front-loads reference-dominant hue families as largest compatible carrier fields before local base/accent/dark roles",
            "medium construction coverage applies extracted medium/process to foreground objects and background carriers",
            "style geometry grammar applies extracted primitive/partition/edge rules across requested content",
            "material/process evidence ladder preserves edge thickness, overlap shadows, surface artifacts, partitions, and light/shadow binding",
            "evidence carrier binding attaches process cues to object contours, internal seams, local facets, contact shadows, and background planes",
        ],
        "actuator_mode": compiled.actuator_mode,
        "prompt_literal_hits": prompt_hits,
        "compiled_positive_chars": len(compiled.positive_prompt),
        "compiled_generation_chars": len(compiled.generation_style_prompt),
        "literal_container_risk_mode": _has_literal_container_risk(literal_items),
        "schema_keys": list(REQUIRED_SCHEMA_KEYS),
        "positive_mechanism_keys": list(POSITIVE_MECHANISM_KEYS),
        "color_light_allocation_count": len(compiled.color_light_allocation),
        "color_light_allocation_hexes": allocation_hexes,
        "color_light_allocation_prompt": compiled.color_light_allocation_prompt,
        "content_carrier_arbitration_count": len(content_carrier_arbitration),
        "content_carrier_arbitration": content_carrier_arbitration,
        "content_carrier_arbitration_prompt": compiled.content_carrier_arbitration_prompt,
        "content_carrier_arbitration_audit": content_carrier_arbitration_audit,
        "mechanism_transfer_prompt": compiled.mechanism_transfer_prompt,
        "medium_construction_coverage": data.medium_construction_coverage,
        "medium_construction_coverage_prompt": compiled.medium_construction_coverage_prompt,
        "style_geometry_grammar": data.style_geometry_grammar,
        "style_geometry_grammar_prompt": compiled.style_geometry_grammar_prompt,
        "material_process_evidence_ladder": data.material_process_evidence_ladder,
        "material_process_evidence_prompt": compiled.material_process_evidence_prompt,
        "evidence_carrier_binding_prompt": compiled.evidence_carrier_binding_prompt,
        "color_value_distribution": data.color_value_distribution,
        "visual_distribution_priority_prompt": compiled.visual_distribution_priority_prompt,
        "dominant_chromatic_field_prompt": compiled.dominant_chromatic_field_prompt,
        "scene_carrier_distribution_prompt": compiled.scene_carrier_distribution_prompt,
        "scene_slot_carrier_binding_prompt": compiled.scene_slot_carrier_binding_prompt,
        "perceptual_carrier_anchor_prompt": compiled.perceptual_carrier_anchor_prompt,
        "bright_scene_recast_prompt": compiled.bright_scene_recast_prompt,
        "carrier_materialization_prompt": compiled.carrier_materialization_prompt,
        "carrier_topology_prompt": compiled.carrier_topology_prompt,
        "bright_chroma_carrier_prompt": compiled.bright_chroma_carrier_prompt,
        "carrier_role_metadata_prompt": compiled.carrier_role_metadata_prompt,
        "field_precedence_prompt": compiled.field_precedence_prompt,
        "color_value_distribution_prompt": compiled.color_value_distribution_prompt,
        "color_role_coverage": color_role_coverage,
    }


def _audit_color_role_coverage(
    allocation: Sequence[Mapping[str, Any]],
    prompt: str,
) -> Dict[str, Any]:
    clean_items = _dedupe_color_light_allocation(allocation)
    prompt_text = str(prompt or "")
    prompt_lower = prompt_text.lower()
    records: List[Dict[str, Any]] = []
    errors: List[str] = []
    warnings: List[str] = []

    def hex_present(item: Mapping[str, Any]) -> bool:
        return any(str(hex_value).lower() in prompt_lower for hex_value in (item.get("hexes") or []))

    def label_present(item: Mapping[str, Any]) -> bool:
        label = _sanitize_prompt_fragment_text(item.get("label") or "")
        if label and label.lower() in prompt_lower:
            return True
        wording = _compact_color_phrase_for_label(str(item.get("generator_safe_wording") or ""))
        wording_words = [word for word in re.findall(r"[a-z0-9-]+", wording.lower()) if len(word) >= 4]
        if not wording_words:
            return False
        matched = sum(1 for word in wording_words[:5] if word in prompt_lower)
        return matched >= min(3, len(wording_words[:5]))

    def present(item: Mapping[str, Any]) -> bool:
        if _allocation_is_primary_material_role(str(item.get("allocation") or "")):
            return hex_present(item) and label_present(item)
        return hex_present(item)

    dark_survives = any(
        present(item)
        for item in clean_items
        if _allocation_is_dark_role(str(item.get("allocation") or "")) or _allocation_item_is_shadow_like(item)
    )

    for item in clean_items:
        allocation_name = str(item.get("allocation") or "")
        allocation_lower = allocation_name.lower()
        label = str(item.get("label") or "reference color")
        item_present = present(item)
        snippet = _prompt_window_for_allocation_item(prompt_text, item)
        carrier_terms = _expected_color_carrier_terms(item)
        carrier_terms_present = not carrier_terms or any(term in prompt_lower for term in carrier_terms)
        weakened_scope = (
            item_present
            and _allocation_is_primary_material_role(allocation_lower)
            and _prompt_window_has_local_only_scope(snippet)
        )
        expanded_selected_scope = (
            item_present
            and _allocation_item_has_selected_carrier_scope(item)
            and _prompt_window_first_scope_marker(snippet) == "broad field"
        )
        records.append(
            {
                "allocation": allocation_name,
                "label": label,
                "hexes": list(item.get("hexes") or []),
                "present": item_present,
                "hex_present": hex_present(item),
                "label_present": label_present(item),
                "carrier_terms": carrier_terms,
                "carrier_terms_present": carrier_terms_present,
                "selected_carrier_scope": _allocation_item_has_selected_carrier_scope(item),
                "snippet": snippet,
            }
        )
        if "large material color" in allocation_lower:
            if not item_present:
                message = f"{allocation_name} anchor missing from compiled color prompt: {label}"
                if dark_survives:
                    message += " while dark/base role survives"
                errors.append(message)
            elif weakened_scope:
                errors.append(f"{allocation_name} anchor weakened to local/tiny scope: {label}")
            elif expanded_selected_scope:
                errors.append(f"{allocation_name} selected-carrier anchor expanded to broad field scope: {label}")
            elif not carrier_terms_present:
                warnings.append(f"{allocation_name} anchor lacks extracted carrier wording: {label}")
        elif "large material-light" in allocation_lower:
            if not item_present and dark_survives:
                warnings.append(f"{allocation_name} anchor missing while dark/base role survives: {label}")
            elif weakened_scope:
                warnings.append(f"{allocation_name} anchor weakened to local/tiny scope: {label}")
            elif expanded_selected_scope:
                warnings.append(f"{allocation_name} selected-carrier anchor expanded to broad field scope: {label}")
            elif item_present and not carrier_terms_present:
                warnings.append(f"{allocation_name} anchor lacks extracted carrier wording: {label}")
        elif dark_survives and _allocation_is_supporting_non_dark_role(allocation_lower) and not item_present:
            warnings.append(f"{allocation_name} anchor missing while dark/base role survives: {label}")

    return {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "dark_role_survives": dark_survives,
        "records": records,
    }


def _audit_content_carrier_arbitration(
    allocation: Sequence[Mapping[str, Any]],
    arbitration: Sequence[Mapping[str, str]],
    prompt: str,
    *,
    require_prompt_terms: bool = True,
) -> Dict[str, Any]:
    records = _dedupe_content_carrier_arbitration(arbitration) or _build_content_carrier_arbitration(allocation)
    prompt_lower = str(prompt or "").lower()
    errors: List[str] = []
    warnings: List[str] = []
    protected = [item for item in records if _arbitration_record_needs_prompt_control(item)]
    # When the generation prompt intentionally uses a short natural guardrail
    # (picture-language mode), the full disallow-list token check is enforced on the
    # structured schema records below, not on the compact prompt text.
    needs_prompt_control = bool(protected) and require_prompt_terms
    required_terms = [
        "whole subject body",
        "whole clothing mass",
        "whole tabletop",
        "whole background",
        "do not recolor entire semantic objects",
    ]
    protected_records: List[Dict[str, Any]] = []
    for item in protected:
        disallowed = str(item.get("disallowed_content_carriers") or "").lower()
        expansion = str(item.get("expansion_control") or "").lower()
        missing_record_terms = [
            term
            for term in required_terms[:4]
            if term not in disallowed
        ]
        if missing_record_terms:
            errors.append(
                "content-carrier arbitration missing disallowed whole semantic carriers for "
                f"{item.get('allocation')}: {item.get('role_label')}"
            )
        if "do not recolor entire semantic objects" not in expansion:
            errors.append(
                "content-carrier arbitration missing expansion control for "
                f"{item.get('allocation')}: {item.get('role_label')}"
            )
        protected_records.append(
            {
                "role_label": item.get("role_label"),
                "allocation": item.get("allocation"),
                "role_scope": item.get("role_scope"),
                "disallowed_content_carriers": item.get("disallowed_content_carriers"),
                "expansion_control": item.get("expansion_control"),
                "prompt_control_required": True,
            }
        )
    if needs_prompt_control:
        missing_prompt_terms = [term for term in required_terms if term not in prompt_lower]
        if missing_prompt_terms:
            errors.append(
                "content-carrier expansion control missing from compiled prompt: "
                + ", ".join(missing_prompt_terms)
            )
    dominant_records = [
        item
        for item in records
        if "dominant field" in str(item.get("allocation") or "").lower()
    ]
    for item in records:
        allocation_name = str(item.get("allocation") or "").lower()
        if "dominant field" in allocation_name:
            continue
        allowed = str(item.get("allowed_content_carriers") or "").lower()
        if "whole background" in allowed or "whole subject body" in allowed or "whole clothing mass" in allowed or "whole tabletop" in allowed:
            errors.append(
                "non-dominant content carrier arbitration allows whole semantic carrier for "
                f"{item.get('allocation')}: {item.get('role_label')}"
            )
    if not needs_prompt_control and any(_role_needs_expansion_control(item, str(item.get("role_scope") or "")) for item in allocation):
        warnings.append("allocation contains expansion-controlled roles but no content-carrier arbitration prompt was required")
    return {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "protected_role_count": len(protected),
        "dominant_field_count": len(dominant_records),
        "records": protected_records,
    }


def _allocation_is_dark_role(allocation: str) -> bool:
    lowered = allocation.lower()
    return "structural dark" in lowered or "dark" in lowered


def _allocation_is_primary_material_role(allocation: str) -> bool:
    lowered = allocation.lower()
    return "large material color" in lowered or "large material-light" in lowered


def _allocation_is_supporting_non_dark_role(allocation: str) -> bool:
    lowered = allocation.lower()
    return any(term in lowered for term in ("medium light", "local material color", "tiny accent"))


def _prompt_window_for_allocation_item(prompt: str, item: Mapping[str, Any], *, radius: int = 120) -> str:
    lowered = prompt.lower()
    label = str(item.get("label") or "").lower()
    if label:
        label_index = lowered.find(label)
        if label_index >= 0:
            clause_end = prompt.find(";", label_index)
            if clause_end < 0:
                clause_end = min(len(prompt), label_index + radius * 2)
            return _clean_clause(prompt[label_index:clause_end])
    for anchor in _allocation_item_prompt_anchor_phrases(item):
        anchor_index = lowered.find(anchor)
        if anchor_index >= 0:
            clause_end = prompt.find(";", anchor_index)
            if clause_end < 0:
                clause_end = min(len(prompt), anchor_index + radius * 2)
            return _clean_clause(prompt[anchor_index:clause_end])
    indexes = [
        lowered.find(str(hex_value).lower())
        for hex_value in (item.get("hexes") or [])
        if lowered.find(str(hex_value).lower()) >= 0
    ]
    if not indexes:
        return ""
    index = min(indexes)
    start = max(0, index - radius)
    end = min(len(prompt), index + radius)
    return _clean_clause(prompt[start:end])


def _allocation_item_prompt_anchor_phrases(item: Mapping[str, Any]) -> List[str]:
    """Return distinctive text anchors for isolating one color role in a prompt.

    Some compact prompts normalize labels ("panel" -> "planes") and collect
    hexes for multiple roles at the end of a shared clause. Falling back directly
    to the hex can then sweep in the previous role's scope marker. These anchors
    let the audit start at the current role's own label/wording whenever possible.
    """
    skip_words = {
        "anchor",
        "bind",
        "color",
        "deep",
        "large",
        "light",
        "material",
        "only",
        "reference",
        "role",
        "surface",
        "surfaces",
    }
    anchors: List[str] = []
    for raw in (
        item.get("label"),
        _compact_color_phrase_for_label(str(item.get("generator_safe_wording") or "")),
        item.get("carrier"),
        item.get("carrier_geometry"),
    ):
        clean = _sanitize_prompt_fragment_text(raw or "").lower()
        if not clean:
            continue
        if len(clean) >= 8:
            anchors.append(clean)
        words = [
            word
            for word in re.findall(r"[a-z0-9-]+", clean)
            if len(word) >= 4 and word not in skip_words
        ]
        for size in (4, 3, 2):
            for index in range(0, max(0, len(words) - size + 1)):
                phrase = " ".join(words[index : index + size])
                if phrase and phrase not in anchors:
                    anchors.append(phrase)
    return anchors


def _prompt_window_has_local_only_scope(snippet: str) -> bool:
    lowered = snippet.lower()
    return (
        "[tiny/local only]" in lowered
        or "[local only]" in lowered
    )


def _prompt_window_first_scope_marker(snippet: str) -> str:
    match = re.search(r"\[([a-z0-9 /-]+)\]", str(snippet or "").lower())
    return _clean_clause(match.group(1)) if match else ""


def _expected_color_carrier_terms(item: Mapping[str, Any]) -> List[str]:
    allocation = str(item.get("allocation") or "").lower()
    if not _allocation_is_primary_material_role(allocation):
        return []
    text = " ".join(
        str(item.get(key) or "")
        for key in ("carrier", "carrier_geometry", "generator_safe_wording")
    ).lower()
    candidates = [
        "bands",
        "patches",
        "facets",
        "planes",
        "surfaces",
        "fields",
        "stripes",
        "rims",
        "glints",
        "material faces",
    ]
    terms = [term for term in candidates if term in text]
    specific = [
        term
        for term in terms
        if term not in {"surfaces", "fields", "planes"}
    ]
    return _dedupe(specific or terms[:2])


def literal_reference_hits(text: str, literal_items: Sequence[str]) -> List[str]:
    """Find literal container wording in text."""

    hits: List[str] = []
    normalized = _normalize_for_match(text)
    for item in literal_items:
        clean = _normalize_for_match(item)
        if not clean:
            continue
        if len(clean.split()) >= 2 and clean in normalized:
            _append_unique(hits, item)
        elif len(clean.split()) == 1 and re.search(rf"\b{re.escape(clean)}\b", normalized):
            _append_unique(hits, item)
    for pattern in _GENERIC_LITERAL_PATTERNS:
        match = pattern.search(text)
        if match:
            _append_unique(hits, match.group(0))
    return hits


def _validate_schema_mapping(data: Mapping[str, Any]) -> None:
    missing = [key for key in REQUIRED_SCHEMA_KEYS if key not in data]
    if missing:
        raise ValueError("StyleSchema missing required keys: " + ", ".join(missing))
    if str(data.get("schema_version") or "") != SCHEMA_VERSION:
        raise ValueError(f"unsupported schema_version: {data.get('schema_version')}")
    if not str(data.get("task_id") or "").strip():
        raise ValueError("task_id is required")
    if not str(data.get("source_policy") or "").strip():
        raise ValueError("source_policy is required")
    positive = data.get("positive_style_mechanisms")
    if not isinstance(positive, Mapping):
        raise ValueError("positive_style_mechanisms must be an object")
    missing_positive = [key for key in POSITIVE_MECHANISM_KEYS if key not in positive]
    missing_positive = [key for key in missing_positive if key != "style_identity"]
    if missing_positive:
        raise ValueError("positive_style_mechanisms missing keys: " + ", ".join(missing_positive))
    for key in POSITIVE_MECHANISM_KEYS:
        if key == "style_identity" and key not in positive:
            continue
        if not isinstance(positive.get(key), str) or not str(positive.get(key)).strip():
            raise ValueError(f"positive_style_mechanisms.{key} must be a non-empty string")
    if not isinstance(data.get("soft_composition_tendencies"), list):
        raise ValueError("soft_composition_tendencies must be a list")
    if not isinstance(data.get("literal_reference_containers"), list):
        raise ValueError("literal_reference_containers must be a list")
    if "color_light_allocation" in data and not isinstance(data.get("color_light_allocation"), list):
        raise ValueError("color_light_allocation must be a list")
    if "content_carrier_arbitration" in data and not isinstance(data.get("content_carrier_arbitration"), list):
        raise ValueError("content_carrier_arbitration must be a list")
    if "mechanism_transfer" in data and not isinstance(data.get("mechanism_transfer"), Mapping):
        raise ValueError("mechanism_transfer must be an object")
    if "medium_construction_coverage" in data and not isinstance(data.get("medium_construction_coverage"), Mapping):
        raise ValueError("medium_construction_coverage must be an object")
    if "style_geometry_grammar" in data and not isinstance(data.get("style_geometry_grammar"), Mapping):
        raise ValueError("style_geometry_grammar must be an object")
    if "material_process_evidence_ladder" in data and not isinstance(data.get("material_process_evidence_ladder"), Mapping):
        raise ValueError("material_process_evidence_ladder must be an object")
    if "color_value_distribution" in data and not isinstance(data.get("color_value_distribution"), Mapping):
        raise ValueError("color_value_distribution must be an object")
    if "tonal_balance" in data and not isinstance(data.get("tonal_balance"), Mapping):
        raise ValueError("tonal_balance must be an object")


def _contract_inner(contract: Mapping[str, Any]) -> Mapping[str, Any]:
    for key in ("contract", "style_detail_contract"):
        value = contract.get(key)
        if isinstance(value, Mapping):
            return value
    return {}


def _literal_items_from_contract(auxiliary_reference_text: str, contract: Mapping[str, Any]) -> List[str]:
    items: List[str] = []
    for clause in _split_clauses(auxiliary_reference_text):
        if _looks_like_literal_container(clause):
            items.extend(_extract_literal_labels(clause) or [clause])

    inner = _contract_inner(contract)
    for key in (
        "portable_vs_literal_carrier_policy",
        "composition_depth_prior",
        "composition_and_camera_prior",
        "generator_safe_wording",
        "detail_density_policy",
        "detail_density_and_scale",
    ):
        value = inner.get(key)
        if isinstance(value, str):
            items.extend(_extract_literal_labels(value))

    for list_key in ("literalization_risks", "reference_content_not_to_copy", "dangerous_wording_to_avoid"):
        value = contract.get(list_key)
        if isinstance(value, list):
            for item in value:
                item_text = str(item).strip()
                labels = _extract_literal_labels(item_text)
                if labels:
                    items.extend(labels)
                elif list_key == "reference_content_not_to_copy" and item_text:
                    items.append(item_text)

    inner_risks = inner.get("literalization_risks")
    if isinstance(inner_risks, list):
        for item in inner_risks:
            labels = _extract_literal_labels(str(item))
            items.extend(labels or [str(item).strip()])

    return [item for item in _dedupe(items) if item]


def _soft_tendencies_from_sources(auxiliary_reference_text: str, inner: Mapping[str, Any]) -> List[str]:
    tendencies: List[str] = []
    for clause in _split_clauses(auxiliary_reference_text):
        if _looks_like_literal_container(clause):
            continue
        if _looks_like_soft_composition(clause):
            tendencies.append(clause)
    for key in ("composition_depth_prior", "composition_and_camera_prior"):
        value = inner.get(key)
        if isinstance(value, str):
            for clause in _split_clauses(value):
                if _looks_like_literal_container(clause):
                    continue
                if _looks_like_soft_composition(clause):
                    tendencies.append(clause)
    return _dedupe(tendencies)


def _color_carrier_text(
    inner: Mapping[str, Any],
    color_contract: Optional[Mapping[str, Any]],
    *,
    auxiliary_reference_text: str = "",
) -> str:
    direct = _first_non_empty(inner, "color_carrier_binding")
    if direct:
        return _attach_hex_anchors_to_color_text(direct, color_contract, auxiliary_reference_text=auxiliary_reference_text)
    if color_contract:
        roles = color_contract.get("color_roles")
        if isinstance(roles, list):
            notes = [
                str(role.get("generator_safe_wording") or "").strip()
                for role in roles
                if isinstance(role, Mapping) and str(role.get("generator_safe_wording") or "").strip()
            ]
            if notes:
                return _attach_hex_anchors_to_color_text(" ".join(notes[:4]), color_contract, auxiliary_reference_text=auxiliary_reference_text)
        summary = str(color_contract.get("style_color_summary") or "").strip()
        if summary:
            return _attach_hex_anchors_to_color_text(summary, color_contract, auxiliary_reference_text=auxiliary_reference_text)
    return "Use reference-derived colors by visual carrier and local material role, not as a numeric ratio or global wash."


def _color_light_allocation_from_schema(value: Any) -> List[Dict[str, Any]]:
    if not isinstance(value, list):
        return []
    out: List[Dict[str, Any]] = []
    for item in value:
        if not isinstance(item, Mapping):
            continue
        hexes = _coerce_hex_list(item.get("hexes"))
        if not hexes:
            continue
        out.append(
            {
                "allocation": _clean_clause(item.get("allocation") or "local"),
                "label": _clean_clause(item.get("label") or "reference color"),
                "hexes": hexes,
                "carrier": _clean_clause(item.get("carrier") or "local material carriers"),
                "carrier_scale": _clean_clause(item.get("carrier_scale") or ""),
                "carrier_geometry": _clean_clause(item.get("carrier_geometry") or ""),
                "light_role": _clean_clause(item.get("light_role") or "material color"),
                "material_vs_glow_boundary": _clean_clause(item.get("material_vs_glow_boundary") or ""),
                "evidence_across_references": _clean_clause(item.get("evidence_across_references") or ""),
                "generator_safe_wording": _clean_clause(item.get("generator_safe_wording") or ""),
                "source": _clean_clause(item.get("source") or "schema"),
            }
        )
    return _dedupe_color_light_allocation(out)


def _color_light_allocation_override_from_schema(value: Any) -> List[Dict[str, Any]]:
    if value is None:
        return []
    if isinstance(value, Mapping):
        value = value.get("color_light_allocation")
    return _color_light_allocation_from_schema(value)


def _content_carrier_arbitration_from_schema(value: Any) -> List[Dict[str, str]]:
    if not isinstance(value, list):
        return []
    out: List[Dict[str, str]] = []
    for item in value:
        if not isinstance(item, Mapping):
            continue
        record = {
            "role_label": _clean_clause(item.get("role_label") or item.get("label") or ""),
            "allocation": _clean_clause(item.get("allocation") or ""),
            "role_scope": _clean_clause(item.get("role_scope") or ""),
            "allowed_content_carriers": _clean_clause(item.get("allowed_content_carriers") or ""),
            "disallowed_content_carriers": _clean_clause(item.get("disallowed_content_carriers") or ""),
            "expansion_control": _clean_clause(item.get("expansion_control") or ""),
        }
        if record["role_label"] and record["role_scope"] and (
            record["allowed_content_carriers"] or record["disallowed_content_carriers"] or record["expansion_control"]
        ):
            out.append(record)
    return _dedupe_content_carrier_arbitration(out)


def _build_content_carrier_arbitration(allocation: Sequence[Mapping[str, Any]]) -> List[Dict[str, str]]:
    records: List[Dict[str, str]] = []
    for item in _dedupe_color_light_allocation(allocation):
        label = _clean_clause(item.get("label") or "reference role")
        allocation_name = _clean_clause(item.get("allocation") or "")
        scope = _color_scope_marker(item) or _fallback_role_scope(item)
        allowed = _allowed_content_carriers_for_role(item, scope)
        disallowed = _disallowed_content_carriers_for_role(item, scope)
        expansion = _expansion_control_for_role(item, scope)
        if not label or not scope:
            continue
        records.append(
            {
                "role_label": label,
                "allocation": allocation_name,
                "role_scope": scope,
                "allowed_content_carriers": allowed,
                "disallowed_content_carriers": disallowed,
                "expansion_control": expansion,
            }
        )
    return _dedupe_content_carrier_arbitration(records)


def _dedupe_content_carrier_arbitration(items: Sequence[Mapping[str, str]]) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    seen = set()
    for item in items:
        key = (
            str(item.get("role_label") or "").lower(),
            str(item.get("allocation") or "").lower(),
            str(item.get("role_scope") or "").lower(),
        )
        if not key[0] or key in seen:
            continue
        seen.add(key)
        out.append(
            {
                "role_label": _clean_clause(item.get("role_label") or ""),
                "allocation": _clean_clause(item.get("allocation") or ""),
                "role_scope": _clean_clause(item.get("role_scope") or ""),
                "allowed_content_carriers": _clean_clause(item.get("allowed_content_carriers") or ""),
                "disallowed_content_carriers": _clean_clause(item.get("disallowed_content_carriers") or ""),
                "expansion_control": _clean_clause(item.get("expansion_control") or ""),
            }
        )
    return out


def _fallback_role_scope(item: Mapping[str, Any]) -> str:
    allocation = str(item.get("allocation") or "").lower()
    if "dominant field" in allocation:
        return "broad field"
    if "medium light" in allocation:
        return "light only"
    if "structural dark" in allocation:
        return "dark structure only"
    if "tiny accent" in allocation:
        return "tiny/local only"
    if "local material color" in allocation:
        return "local only"
    if _allocation_is_primary_material_role(allocation):
        return "material carrier"
    return "carrier-bound"


def _allowed_content_carriers_for_role(item: Mapping[str, Any], scope: str) -> str:
    allocation = str(item.get("allocation") or "").lower()
    scope_lower = str(scope or "").lower()
    text = _allocation_item_carrier_text(item)
    allowed: List[str] = []
    if "dominant field" in allocation:
        allowed.extend(["broad background planes", "environment fields", "large non-identity material fields"])
    elif "medium light" in allocation or "light only" in scope_lower:
        allowed.extend(["highlight planes", "rim light", "reflected light", "negative-space light", "small glow carriers"])
    elif "structural dark" in allocation or "dark structure" in scope_lower:
        allowed.extend(["silhouettes", "contact shadows", "occlusion wedges", "linework", "negative spaces"])
    elif _role_needs_expansion_control(item, scope):
        allowed.extend(["small object surfaces", "local bands", "edge inserts", "side facets", "detail patches"])
    elif _allocation_is_primary_material_role(allocation):
        allowed.extend(["compatible material surfaces", "object material planes", "bounded surface fields"])
    else:
        allowed.extend(["local material surfaces", "detail carriers", "small object surfaces"])
    if re.search(r"\bpatch(?:es)?\b", text):
        _append_unique(allowed, "detail patches")
    if re.search(r"\binsert(?:s)?\b", text):
        _append_unique(allowed, "edge inserts")
    if re.search(r"\bstripe(?:s)?\b|\bband(?:s)?\b", text):
        _append_unique(allowed, "local bands")
    if re.search(r"\bfacet(?:s)?\b|\bside\s+face(?:s)?\b", text):
        _append_unique(allowed, "side facets")
    if re.search(r"\bglint(?:s)?\b|\bspark(?:le|les)?\b|\bspecular\b", text):
        _append_unique(allowed, "tiny glints")
    return _join_limited(_dedupe(allowed), 190, separator=", ")


def _disallowed_content_carriers_for_role(item: Mapping[str, Any], scope: str) -> str:
    allocation = str(item.get("allocation") or "").lower()
    if "dominant field" in allocation:
        return "whole subject body and whole clothing mass unless explicitly requested by content"
    if _role_needs_expansion_control(item, scope):
        return "whole subject body, whole clothing mass, whole tabletop, whole background"
    if "medium light" in allocation:
        return "ordinary surface color, whole clothing mass, whole tabletop"
    if "structural dark" in allocation:
        return "generic black clothing, generic object paint, whole tabletop"
    if "local material color" in allocation or "tiny accent" in allocation:
        return "whole subject body, whole clothing mass, whole tabletop, whole background"
    return ""


def _expansion_control_for_role(item: Mapping[str, Any], scope: str) -> str:
    allocation = str(item.get("allocation") or "").lower()
    if "dominant field" in allocation:
        return "may occupy broad fields while preserving requested subject identity colors"
    if _role_needs_expansion_control(item, scope):
        return "do not recolor entire semantic objects for selected/accent/local roles"
    if "medium light" in allocation:
        return "do not turn light roles into generic material color"
    if "structural dark" in allocation:
        return "do not turn dark structure into generic black object paint"
    return "keep role carrier-bound and avoid whole-image tint"


def _role_needs_expansion_control(item: Mapping[str, Any], scope: str) -> bool:
    allocation = str(item.get("allocation") or "").lower()
    if "dominant field" in allocation:
        return False
    if "tiny accent" in allocation or "local material color" in allocation:
        return True
    scope_lower = str(scope or "").lower()
    if any(term in scope_lower for term in ("selected carrier", "local only", "tiny/local")):
        return True
    text = _allocation_item_carrier_text(item)
    if any(term in text for term in ("selected", "accent", "accents", "patch", "insert", "stripe", "side facet", "local surface")):
        return True
    return False


def _allocation_item_carrier_text(item: Mapping[str, Any]) -> str:
    return " ".join(
        str(item.get(key) or "")
        for key in (
            "label",
            "allocation",
            "carrier",
            "carrier_scale",
            "carrier_geometry",
            "light_role",
            "material_vs_glow_boundary",
            "generator_safe_wording",
        )
    ).lower()


def _medium_construction_coverage_from_schema(value: Any) -> Dict[str, str]:
    if not isinstance(value, Mapping):
        return {}
    out: Dict[str, str] = {}
    for key in (
        "construction_units",
        "surface_topology",
        "edge_transition",
        "foreground_object_application",
        "background_field_application",
        "coverage_scope",
    ):
        text = _clean_clause(value.get(key) or "")
        if text:
            out[key] = text
    return out


def _style_geometry_grammar_from_schema(value: Any) -> Dict[str, str]:
    if not isinstance(value, Mapping):
        return {}
    out: Dict[str, str] = {}
    for key in (
        "primitive_shapes",
        "silhouette_policy",
        "internal_partitioning",
        "edge_profile",
        "proportion_shift",
        "portable_geometry",
    ):
        text = _clean_clause(value.get(key) or "")
        if text:
            out[key] = text
    return out


def _material_process_evidence_ladder_from_schema(value: Any) -> Dict[str, str]:
    if not isinstance(value, Mapping):
        return {}
    out: Dict[str, str] = {}
    for key in (
        "edge_thickness_evidence",
        "overlap_shadow_evidence",
        "surface_artifact_evidence",
        "internal_partition_evidence",
        "highlight_shadow_binding",
        "coverage_requirement",
    ):
        text = _clean_clause(value.get(key) or "")
        if text:
            out[key] = text
    return out


def _build_medium_construction_coverage(inner: Mapping[str, Any]) -> Dict[str, str]:
    transfer = _mechanism_transfer_from_schema(inner.get("mechanism_transfer"))
    construction = _join_non_empty(
        _first_non_empty(inner, "medium_mark_system", "surface_material_mechanism"),
        _first_non_empty(inner, "surface_artifact", "texture_or_artifact_mechanism"),
    )
    surface_topology = _join_non_empty(
        _first_non_empty(inner, "surface_artifact", "texture_or_artifact_mechanism"),
        _first_non_empty(inner, "detail_density_policy", "detail_density_and_scale"),
    )
    edge_transition = _first_non_empty(
        inner,
        "line_edge_language",
        "line_and_edge_language",
        default="Use extracted edge transitions and process marks on visible carriers.",
    )
    foreground_application = _join_non_empty(
        transfer.get("construction_rule", ""),
        transfer.get("application_scope", ""),
        default=(
            "Requested subjects and objects must be visibly rebuilt with the extracted medium/process, "
            "not left as generic content placed over a styled backdrop."
        ),
    )
    background_application = _join_non_empty(
        _first_non_empty(inner, "composition_depth_prior", "composition_and_camera_prior"),
        _first_non_empty(inner, "detail_density_policy", "detail_density_and_scale"),
        default="Background planes use the same extracted medium construction with lower detail where appropriate.",
    )
    return _medium_construction_coverage_from_schema(
        {
            "construction_units": construction or "Extracted medium/process construction units.",
            "surface_topology": surface_topology or construction,
            "edge_transition": edge_transition,
            "foreground_object_application": foreground_application,
            "background_field_application": background_application,
            "coverage_scope": (
                "Apply the extracted medium construction to foreground subjects, requested objects, background planes, "
                "shadows, highlights, and local detail carriers."
            ),
        }
    )


def _build_style_geometry_grammar(inner: Mapping[str, Any]) -> Dict[str, str]:
    transfer = _mechanism_transfer_from_schema(inner.get("mechanism_transfer"))
    shape_text = _first_non_empty(
        inner,
        "shape_proportion_language",
        "surface_material_mechanism",
        default="Use extracted shape simplification and proportion language.",
    )
    line_text = _first_non_empty(
        inner,
        "line_edge_language",
        "line_and_edge_language",
        default="Use extracted silhouette and edge language.",
    )
    partition_text = _join_non_empty(
        _first_non_empty(inner, "detail_density_policy", "detail_density_and_scale"),
        transfer.get("construction_rule", ""),
    )
    portable_geometry = _join_non_empty(
        transfer.get("construction_rule", ""),
        transfer.get("carrier_binding_rule", ""),
        _first_non_empty(inner, "composition_depth_prior", "composition_and_camera_prior"),
    )
    return _style_geometry_grammar_from_schema(
        {
            "primitive_shapes": shape_text,
            "silhouette_policy": line_text,
            "internal_partitioning": partition_text,
            "edge_profile": line_text,
            "proportion_shift": shape_text,
            "portable_geometry": portable_geometry or shape_text,
        }
    )


def _build_material_process_evidence_ladder(inner: Mapping[str, Any]) -> Dict[str, str]:
    transfer = _mechanism_transfer_from_schema(inner.get("mechanism_transfer"))
    edge_text = _join_non_empty(
        _first_non_empty(inner, "line_edge_language", "line_and_edge_language"),
        _first_non_empty(inner, "surface_artifact", "texture_or_artifact_mechanism"),
    )
    overlap_text = _join_non_empty(
        _first_non_empty(inner, "surface_artifact", "texture_or_artifact_mechanism"),
        _first_non_empty(inner, "composition_depth_prior", "composition_and_camera_prior"),
        transfer.get("construction_rule", ""),
    )
    surface_text = _join_non_empty(
        _first_non_empty(inner, "medium_mark_system", "surface_material_mechanism"),
        _first_non_empty(inner, "surface_artifact", "texture_or_artifact_mechanism"),
    )
    partition_text = _join_non_empty(
        _first_non_empty(inner, "detail_density_policy", "detail_density_and_scale"),
        _first_non_empty(inner, "shape_proportion_language"),
        transfer.get("construction_rule", ""),
    )
    binding_text = _join_non_empty(
        _first_non_empty(inner, "line_edge_language", "line_and_edge_language"),
        _first_non_empty(inner, "color_carrier_binding"),
        transfer.get("carrier_binding_rule", ""),
    )
    return _material_process_evidence_ladder_from_schema(
        {
            "edge_thickness_evidence": edge_text or "Extracted edge/process evidence must remain visible.",
            "overlap_shadow_evidence": overlap_text or "Extracted overlap and contact-shadow evidence must carry depth.",
            "surface_artifact_evidence": surface_text or "Extracted surface artifacts must appear on visible carriers.",
            "internal_partition_evidence": partition_text or "Extracted internal partitions must appear on ordinary content surfaces.",
            "highlight_shadow_binding": binding_text or "Highlights and shadows follow extracted material boundaries.",
            "coverage_requirement": (
                "Show the extracted evidence on main subjects, requested objects, background planes, "
                "shadows, highlights, and local detail carriers."
            ),
        }
    )


def _schema_prompt_fragment(
    text: str,
    *,
    literal_items: Sequence[str],
    limit: int,
    high_risk: bool = False,
    prioritize_geometry: bool = False,
) -> str:
    clean = _sanitize_positive_text(
        str(text or ""),
        literal_items,
        remove_soft=True,
        high_risk=high_risk,
    )
    priority: List[str] = _portable_geometry_cues(clean) if prioritize_geometry else []
    clauses = [
        clause
        for clause in _candidate_clauses(clean, split_commas=False)
        if clause and not _looks_like_literal_container(clause)
    ]
    return _join_limited([*priority, *clauses[:2]], limit, separator=", ")


def _portable_geometry_cues(text: str) -> List[str]:
    lowered = str(text or "").lower()
    cues: List[str] = []
    for cue in (
        "radiating material bands",
        "sweeping curves",
        "internal divider lines",
        "repeated internal rib lines",
        "curved internal line rhythm",
        "surface-internal decorative linework",
        "pane divisions",
        "angular facets",
        "folded material planes",
        "layered overlap",
        "thin seams",
        "broad rounded wedges",
        "slabs",
    ):
        if cue in lowered:
            _append_unique(cues, cue)
    if re.search(r"\bpane(?:s)?\b|\bleaded\b|\bcame\b", lowered):
        _append_unique(cues, "pane divisions")
    if re.search(r"\bfacet(?:s)?\b|\btriangles?\b|\btrapezoids?\b|\bshards?\b", lowered):
        _append_unique(cues, "angular facets")
    if re.search(r"\bfold(?:ed|s)?\b|\bcreases?\b", lowered):
        _append_unique(cues, "folded material planes")
    if re.search(r"\bseam(?:s)?\b|\bchannels?\b|\bpartitions?\b", lowered):
        _append_unique(cues, "thin seams")
    if re.search(r"\bcurve(?:s)?\b|\bcurving\b|\bs-curve(?:s)?\b", lowered):
        _append_unique(cues, "sweeping curves")
    if re.search(r"\bband(?:s)?\b|\bstripe(?:s)?\b", lowered):
        if not any("material bands" in cue for cue in cues):
            _append_unique(cues, "material bands")
    return cues[:5]


def _compile_medium_construction_coverage_prompt(
    coverage: Mapping[str, Any],
    *,
    literal_items: Sequence[str],
    max_chars: int = 220,
) -> str:
    data = _medium_construction_coverage_from_schema(coverage)
    if not data:
        return ""
    artifacts = _schema_prompt_fragment(
        data.get("construction_units") or data.get("surface_topology") or "",
        literal_items=literal_items,
        limit=84,
        high_risk=True,
    )
    edge = _schema_prompt_fragment(
        data.get("edge_transition") or "",
        literal_items=literal_items,
        limit=64,
        high_risk=True,
    )
    scope = _schema_prompt_fragment(
        data.get("coverage_scope") or data.get("foreground_object_application") or "",
        literal_items=literal_items,
        limit=112,
        high_risk=True,
    )
    clauses: List[str] = [
        (
            "coverage: foreground subjects/objects and background planes expose extracted units; "
            "shadows/highlights follow constructed edges; not backdrop-only"
        ),
    ]
    if artifacts:
        clauses.append(f"surface units: {artifacts}")
    if edge:
        clauses.append(f"edge/process: {edge}")
    if scope:
        clauses.append(scope)
    body = _join_limited(clauses, max(80, int(max_chars) - len("Medium construction: ")), separator="; ")
    return _clean_clause("Medium construction: " + body)


def _compile_style_geometry_grammar_prompt(
    grammar: Mapping[str, Any],
    *,
    literal_items: Sequence[str],
    max_chars: int = 220,
) -> str:
    data = _style_geometry_grammar_from_schema(grammar)
    if not data:
        return ""
    primitives = _schema_prompt_fragment(
        data.get("primitive_shapes") or data.get("proportion_shift") or "",
        literal_items=literal_items,
        limit=82,
        high_risk=True,
        prioritize_geometry=True,
    )
    partition = _schema_prompt_fragment(
        data.get("internal_partitioning") or data.get("portable_geometry") or "",
        literal_items=literal_items,
        limit=86,
        high_risk=True,
        prioritize_geometry=True,
    )
    edge = _schema_prompt_fragment(
        data.get("edge_profile") or data.get("silhouette_policy") or "",
        literal_items=literal_items,
        limit=72,
        high_risk=True,
    )
    clauses: List[str] = [
        "one geometry grammar: shape primitives, internal partitions, detail carriers, background"
    ]
    if partition:
        clauses.append(f"partitions: {partition}")
    if primitives:
        clauses.append(f"primitives: {primitives}")
    if edge:
        clauses.append(f"edge profile: {edge}")
    body = _join_limited(clauses, max(80, int(max_chars) - len("Geometry grammar: ")), separator="; ")
    return _clean_clause("Geometry grammar: " + body)


def _compile_material_process_evidence_prompt(
    ladder: Mapping[str, Any],
    *,
    literal_items: Sequence[str],
    max_chars: int = 240,
) -> str:
    data = _material_process_evidence_ladder_from_schema(ladder)
    if not data:
        return ""
    edge = _schema_prompt_fragment(
        data.get("edge_thickness_evidence") or "",
        literal_items=literal_items,
        limit=50,
        high_risk=True,
        prioritize_geometry=True,
    )
    overlap = _schema_prompt_fragment(
        data.get("overlap_shadow_evidence") or "",
        literal_items=literal_items,
        limit=50,
        high_risk=True,
        prioritize_geometry=True,
    )
    surface = _schema_prompt_fragment(
        data.get("surface_artifact_evidence") or "",
        literal_items=literal_items,
        limit=58,
        high_risk=True,
    )
    partition = _schema_prompt_fragment(
        data.get("internal_partition_evidence") or "",
        literal_items=literal_items,
        limit=50,
        high_risk=True,
        prioritize_geometry=True,
    )
    binding = _schema_prompt_fragment(
        data.get("highlight_shadow_binding") or data.get("coverage_requirement") or "",
        literal_items=literal_items,
        limit=54,
        high_risk=True,
    )
    clauses: List[str] = [
        "main objects and backgrounds show process evidence, not just palette"
    ]
    if edge:
        clauses.append(f"edges: {edge}")
    if overlap:
        clauses.append(f"overlap/shadow: {overlap}")
    if surface:
        clauses.append(f"surface artifacts: {surface}")
    if partition:
        clauses.append(f"partitions: {partition}")
    if binding:
        clauses.append(f"light/shadow binding: {binding}")
    body = _join_limited(clauses, max(80, int(max_chars) - len("Material process evidence: ")), separator="; ")
    return _clean_clause("Material process evidence: " + body)


def _compile_evidence_carrier_binding_prompt(
    ladder: Mapping[str, Any],
    coverage: Mapping[str, Any],
    grammar: Mapping[str, Any],
    arbitration: Sequence[Mapping[str, str]],
    *,
    literal_items: Sequence[str],
    max_chars: int = 220,
) -> str:
    ladder_data = _material_process_evidence_ladder_from_schema(ladder)
    coverage_data = _medium_construction_coverage_from_schema(coverage)
    grammar_data = _style_geometry_grammar_from_schema(grammar)
    if not ladder_data and not coverage_data and not grammar_data:
        return ""

    body = (
        "process cues on object contours, internal seams, facets/patches, contact shadows, background planes; "
        "not backdrop/palette"
    )
    return _clean_clause(
        "Evidence carriers: "
        + _truncate_words(body, max(90, int(max_chars) - len("Evidence carriers: ")))
    )


def _portable_evidence_cues(text: str) -> List[str]:
    lowered = str(text or "").lower()
    cues: List[str] = []
    cue_patterns = (
        (r"\bcut|cut[- ]edge|edge thickness|thick edge|silhouette", "cut/thick edges"),
        (r"\bseam|divider|line|contour|rim", "seams/contours"),
        (r"\bfacet|plane|slab|wedge|pane|partition", "facets/partitions"),
        (r"\boverlap|contact shadow|cast shadow|shadow", "overlap/contact shadows"),
        (r"\bgrain|fiber|mottle|texture|surface artifact|brush|wash|glaze", "surface artifacts"),
        (r"\bhighlight|rim light|key light|glint|reflection", "highlight/shadow boundaries"),
    )
    for pattern, cue in cue_patterns:
        if re.search(pattern, lowered):
            _append_unique(cues, cue)
    if not cues:
        _append_unique(cues, "edge/surface evidence")
        _append_unique(cues, "partition evidence")
    return cues[:4]


def _portable_evidence_carrier_cues(
    text: str,
    arbitration: Sequence[Mapping[str, str]],
) -> List[str]:
    lowered = str(text or "").lower()
    for item in arbitration:
        lowered += " " + " ".join(str(item.get(key) or "").lower() for key in ("allowed_content_carriers", "role_scope"))
    carriers: List[str] = []
    _append_unique(carriers, "object contours")
    _append_unique(carriers, "internal seams")
    if re.search(r"\bfacet|plane|slab|wedge|pane|patch|insert|band|surface", lowered):
        _append_unique(carriers, "facets/patches")
    if re.search(r"\boverlap|shadow|contact|cast", lowered):
        _append_unique(carriers, "contact shadows")
    if re.search(r"\bbackground|field|plane", lowered):
        _append_unique(carriers, "background planes")
    if re.search(r"\bhighlight|rim|glint|reflection|light", lowered):
        _append_unique(carriers, "light-facing surfaces")
    return carriers[:5]


def _compile_application_text(clauses: Sequence[str], max_chars: int) -> str:
    clean = [_clean_clause(clause) for clause in clauses if _clean_clause(clause)]
    if not clean:
        return ""
    marker_specs = (
        ("field precedence:", 350, 560),
        ("carrier role metadata:", 340, 520),
        ("bright chroma carriers:", 390, 610),
        ("carrier topology:", 350, 560),
        ("carrier materialization:", 370, 650),
        ("bright-scene recast:", 420, 620),
        ("perceptual carrier anchors:", 380, 560),
        ("scene-slot carrier binding:", 380, 620),
        ("perceptual hue carriers:", 330, 520),
        ("carrier-area actuator:", 280, 430),
        ("visual distribution lock:", 90, 160),
        ("dominant chromatic field:", 124, 176),
        ("scene carrier map:", 86, 164),
        ("medium construction:", 76, 150),
        ("geometry grammar:", 76, 150),
        ("shape/proportion and edge rules", 82, 106),
        ("mechanism transfer:", 88, 150),
        ("material process evidence:", 76, 150),
        ("evidence carriers:", 72, 150),
        ("medium/process artifacts", 92, 135),
        ("tonal balance:", 80, 130),
        ("do not copy reference subjects", 66, 90),
    )
    marker_order = tuple(spec[0] for spec in marker_specs)
    marker_limits = {spec[0]: (spec[1], spec[2]) for spec in marker_specs}
    priority: List[str] = []
    for marker in marker_order:
        for clause in clean:
            if marker in clause.lower():
                _append_unique_nonredundant(priority, clause)
    for clause in clean:
        _append_unique_nonredundant(priority, clause)
    limit = max(120, int(max_chars))
    full = "; ".join(priority).strip(" ,.;")
    if len(full) <= limit:
        return full

    required: List[Tuple[str, str]] = []
    seen_markers = set()
    has_geometry_grammar = any(
        _application_marker_for_clause(clause, marker_order) == "geometry grammar:"
        for clause in priority
    )
    for clause in priority:
        marker = _application_marker_for_clause(clause, marker_order)
        if not marker or marker in seen_markers:
            continue
        if marker in {
            "shape/proportion and edge rules" if has_geometry_grammar else "",
            "medium/process artifacts",
            "tonal balance:",
            "do not copy reference subjects",
        }:
            continue
        required.append((marker, clause))
        seen_markers.add(marker)

    if required:
        selected_pairs = _compact_required_application_clauses(required, limit, marker_limits)
        selected = [text for _marker, text in selected_pairs]
        selected_markers = {marker for marker, _text in selected_pairs}
        for clause in priority:
            marker = _application_marker_for_clause(clause, marker_order)
            if marker and marker in selected_markers:
                continue
            used = len("; ".join(selected))
            remaining = limit - used - (2 if selected else 0)
            if remaining < 84:
                break
            addition = clause if len(clause) <= remaining else _truncate_words(clause, min(remaining, 150))
            if addition:
                _append_unique_nonredundant(selected, addition)
        if selected:
            return "; ".join(selected).strip(" ,.;")

    selected: List[str] = []
    for clause in priority:
        candidate = "; ".join(selected + [clause])
        if len(candidate) <= limit:
            selected.append(clause)
            continue
        used = len("; ".join(selected))
        remaining = limit - used - (2 if selected else 0)
        if remaining >= 90:
            clipped = _truncate_words(clause, remaining)
            if clipped:
                selected.append(clipped)
        break
    return "; ".join(selected).strip(" ,.;")


def _application_marker_for_clause(clause: str, marker_order: Sequence[str]) -> str:
    lowered = str(clause or "").lower()
    for marker in marker_order:
        if marker in lowered:
            return marker
    return ""


def _compact_required_application_clauses(
    required: Sequence[Tuple[str, str]],
    limit: int,
    marker_limits: Mapping[str, Tuple[int, int]],
) -> List[Tuple[str, str]]:
    items = list(required)
    optional_markers = {"tonal balance:", "do not copy reference subjects"}
    while items and _application_minimum_total(items, marker_limits) > limit:
        drop_index = next(
            (
                index
                for index in range(len(items) - 1, -1, -1)
                if items[index][0] in optional_markers
            ),
            -1,
        )
        if drop_index < 0:
            break
        items.pop(drop_index)
    if not items:
        return []

    separator_total = 2 * max(0, len(items) - 1)
    available = max(0, int(limit) - separator_total)
    min_caps: List[int] = []
    max_caps: List[int] = []
    for marker, clause in items:
        min_limit, max_limit = marker_limits.get(marker, (96, 150))
        marker_floor = min(len(clause), max(len(marker) + 24, min_limit))
        min_caps.append(marker_floor)
        max_caps.append(min(len(clause), max(max_limit, marker_floor)))

    if sum(min_caps) > available:
        items = _pressure_order_required_application_clauses(items)
        min_caps = []
        for marker, clause in items:
            min_limit, _max_limit = marker_limits.get(marker, (96, 150))
            min_caps.append(min(len(clause), max(len(marker) + 24, min_limit)))
        selected: List[Tuple[str, str]] = []
        for (marker, clause), cap in zip(items, min_caps):
            used = len("; ".join(text for _marker, text in selected))
            remaining = int(limit) - used - (2 if selected else 0)
            if remaining < max(48, len(marker) + 8):
                continue
            compact = _compact_application_clause(marker, clause, min(cap, remaining))
            if compact:
                selected.append((marker, compact))
        return selected

    caps = list(min_caps)
    extra = available - sum(caps)
    for index, max_cap in enumerate(max_caps):
        if extra <= 0:
            break
        add = min(extra, max(0, max_cap - caps[index]))
        caps[index] += add
        extra -= add
    selected: List[Tuple[str, str]] = []
    for (marker, clause), cap in zip(items, caps):
        compact = _compact_application_clause(marker, clause, cap)
        if compact:
            selected.append((marker, compact))
    return selected


def _pressure_order_required_application_clauses(
    items: Sequence[Tuple[str, str]],
) -> List[Tuple[str, str]]:
    markers = {marker for marker, _clause in items}
    frontloaded_scene_actuator = bool(
        markers
        & {
            "bright-scene recast:",
            "field precedence:",
            "carrier role metadata:",
            "bright chroma carriers:",
            "carrier materialization:",
            "carrier topology:",
            "perceptual carrier anchors:",
        }
    )
    if not frontloaded_scene_actuator or "scene-slot carrier binding:" not in markers:
        return list(items)
    delayed = [item for item in items if item[0] == "scene-slot carrier binding:"]
    return [item for item in items if item[0] != "scene-slot carrier binding:"] + delayed


def _compact_application_clause(marker: str, clause: str, cap: int) -> str:
    clean = _clean_clause(clause)
    if not clean:
        return ""
    if len(clean) <= cap:
        return clean
    if marker == "mechanism transfer:":
        return _compact_mechanism_application_clause(clean, cap)
    if marker == "material process evidence:":
        return _compact_material_process_application_clause(clean, cap)
    if marker == "medium construction:":
        return _compact_medium_application_clause(clean, cap)
    if marker == "geometry grammar:":
        return _compact_geometry_application_clause(clean, cap)
    if marker == "carrier-area actuator:":
        return _compact_carrier_area_actuator_application_clause(clean, cap)
    if marker == "carrier topology:":
        return _compact_carrier_topology_application_clause(clean, cap)
    if marker == "carrier role metadata:":
        return _compact_carrier_role_metadata_application_clause(clean, cap)
    if marker == "field precedence:":
        return _compact_field_precedence_application_clause(clean, cap)
    if marker == "bright chroma carriers:":
        return _compact_bright_chroma_carrier_application_clause(clean, cap)
    if marker == "carrier materialization:":
        return _compact_carrier_materialization_application_clause(clean, cap)
    if marker == "bright-scene recast:":
        return _compact_bright_scene_recast_application_clause(clean, cap)
    if marker == "perceptual carrier anchors:":
        return _compact_perceptual_carrier_anchor_application_clause(clean, cap)
    if marker == "scene-slot carrier binding:":
        return _compact_scene_slot_carrier_binding_application_clause(clean, cap)
    if marker == "perceptual hue carriers:":
        return _compact_perceptual_hue_carrier_application_clause(clean, cap)
    if marker == "visual distribution lock:":
        return _compact_visual_distribution_application_clause(clean, cap)
    if marker == "dominant chromatic field:":
        return _compact_dominant_chromatic_application_clause(clean, cap)
    if marker == "scene carrier map:":
        return _compact_scene_carrier_application_clause(clean, cap)
    return _truncate_words(clean, cap)


def _compact_carrier_area_actuator_application_clause(clause: str, cap: int) -> str:
    prefix = "Carrier-area actuator: "
    body = _strip_prompt_prefix(clause, "carrier-area actuator:")
    parts = [_clean_clause(part) for part in body.split(";") if _clean_clause(part)]
    if not parts:
        return _truncate_words(clause, cap)
    selected: List[str] = []
    required_labels = (
        "visual distribution lock:",
        "dominant chromatic field:",
        "scene carrier map:",
        "color/value distribution:",
    )
    for label in required_labels:
        part = next((item for item in parts if item.lower().startswith(label)), "")
        if not part:
            continue
        used = len(prefix) + len("; ".join(selected))
        remaining = cap - used - (2 if selected else 0)
        if remaining < len(label) + 18:
            continue
        selected.append(_truncate_words(part, remaining))
    if selected:
        return _clean_clause(prefix + "; ".join(selected))
    return _truncate_words(clause, cap)


def _compact_carrier_topology_application_clause(clause: str, cap: int) -> str:
    prefix = "Carrier topology: "
    body = _strip_prompt_prefix(clause, "carrier topology:")
    parts = [_clean_clause(part) for part in body.split(";") if _clean_clause(part)]
    if not parts:
        return _truncate_words(clause, cap)

    visual = next((item for item in parts if item.lower().startswith("visual distribution lock:")), "")
    topology = next((item for item in parts if item.lower().startswith("dominant hue topology:")), "")
    family_text = ""
    source = topology or visual
    match = re.match(
        r"(?:visual distribution lock:|dominant hue topology:)\s*(.*?)\s+must\s+own\s+large\s+carrier\s+slots\b",
        source,
        flags=re.IGNORECASE,
    )
    if match:
        family_text = match.group(1).strip(" ,.;")
        family_items = [
            _clean_clause(part)
            for part in re.split(r"\s*,\s*|\s+\band\b\s+", family_text)
            if _clean_clause(part)
        ]
        if family_items:
            family_text = _human_join(family_items[:3])
    if family_text:
        visual = f"visual distribution lock: {family_text} large carrier slots before small accents"
    else:
        visual = "visual distribution lock: dominant hues own large carrier slots before small accents"

    selected = [
        visual,
        (
            "scene carrier map: large carrier slots = open-light/background plane, "
            "floor/reflection band, large material face, repeated bounded surface fields"
        ),
        "anti-strip collapse: linear lamps, sign strips, rim lines do not count as main dominant-hue carriers",
        (
            "color role order: colored material-light and dominant chromatic fields before pale base and dark/base; "
            "dark/base carriers support/partition only and cannot become largest filled area unless reference dark is dominant"
        ),
    ]
    compact = _clean_clause(prefix + "; ".join(selected))
    if len(compact) <= max(cap, 470):
        return compact

    selected[1] = (
        "scene carrier map: large carrier slots = open-light/background plane, "
        "floor/reflection band, large material face"
    )
    compact = _clean_clause(prefix + "; ".join(selected))
    if len(compact) <= max(cap, 430):
        return compact

    selected[2] = "anti-strip collapse: linear lamps/sign strips/rim lines do not count"
    selected[3] = (
        "dark/base carriers support/partition only; cannot become largest filled area unless reference dark is dominant"
    )
    return _clean_clause(prefix + "; ".join(selected))


def _compact_bright_chroma_carrier_application_clause(clause: str, cap: int) -> str:
    prefix = "Bright chroma carriers: "
    body = _strip_prompt_prefix(clause, "bright chroma carriers:")
    parts = [_clean_clause(part) for part in body.split(";") if _clean_clause(part)]
    if not parts:
        return _truncate_words(clause, cap)

    visual = next((item for item in parts if item.lower().startswith("visual distribution lock:")), "")
    anchors = next((item for item in parts if item.lower().startswith("priority anchors:")), "")
    slots = next(
        (
            item
            for item in parts
            if item.lower().startswith("scene carrier map:") or item.lower().startswith("large carrier slots:")
        ),
        "",
    )
    strip = next((item for item in parts if item.lower().startswith("anti-strip collapse:")), "")
    wash = next((item for item in parts if item.lower().startswith("wash guard:")), "")
    dark = next((item for item in parts if item.lower().startswith("dark/base guard:")), "")

    selected = [
        visual or "visual distribution lock: dominant light hue families keep saturated visible carrier color before pale/base/dark",
        anchors,
        slots
        or (
            "scene carrier map: large carrier slots = open-light/background plane, floor/reflection band, "
            "large material face, repeated bounded surface fields"
        ),
        strip or "anti-strip collapse: linear lamps/sign strips/rim lines/glints do not count; not strip-only",
        wash or "wash guard: not cream/white/pale wash",
        dark or "dark/base guard: dark/base supports/partitions only",
    ]
    compact = _clean_clause(prefix + "; ".join(item for item in selected if item))
    if len(compact) <= max(cap, 480):
        return compact

    selected = [
        selected[0],
        anchors,
        "scene carrier map: large carrier slots = open-light/background plane, floor/reflection band, large material face",
        "anti-strip collapse: linear lamps/sign strips/rim lines do not count; not strip-only",
        "wash guard: not cream/white/pale wash",
        "dark/base supports/partitions only",
    ]
    compact = _clean_clause(prefix + "; ".join(item for item in selected if item))
    if len(compact) <= max(cap, 430):
        return compact

    selected = [
        selected[0],
        "scene carrier map: large carrier slots = open-light/background plane, floor/reflection band, large material face",
        "anti-strip collapse: linear lamps/sign strips/rim lines do not count; not strip-only",
        "not cream/white/pale wash; dark/base supports/partitions only",
    ]
    return _clean_clause(prefix + "; ".join(item for item in selected if item))


def _compact_carrier_role_metadata_application_clause(clause: str, cap: int) -> str:
    prefix = "Carrier role metadata: "
    body = _strip_prompt_prefix(clause, "carrier role metadata:")
    parts = [_clean_clause(part) for part in body.split(";") if _clean_clause(part)]
    if not parts:
        return _truncate_words(clause, cap)

    wanted_labels = (
        "visual distribution lock:",
        "bright chroma primary:",
        "bounded multihue material fields:",
        "pale/base local:",
        "linear/accent only:",
        "dark/base support:",
        "scene carrier map:",
        "anti-collapse:",
    )
    selected: List[str] = []
    for label in wanted_labels:
        part = next((item for item in parts if item.lower().startswith(label)), "")
        if part:
            selected.append(part)
    if not selected:
        return _truncate_words(clause, cap)

    compact = _clean_clause(prefix + "; ".join(selected))
    if len(compact) <= cap:
        return compact

    shortened: List[str] = []
    for part in selected:
        lowered = part.lower()
        if lowered.startswith("scene carrier map:"):
            shortened.append(
                "scene carrier map: large carrier slots = open-light/background plane, floor/reflection band, large material face"
            )
        elif lowered.startswith("anti-collapse:"):
            shortened.append("anti-collapse: strips/signs/glints/partition lines cannot replace main carrier area")
        elif lowered.startswith("dark/base support:"):
            shortened.append(_truncate_words(part, 110))
        elif lowered.startswith("bounded multihue material fields:"):
            shortened.append(_truncate_words(part, 150))
        elif lowered.startswith("bright chroma primary:"):
            shortened.append(_truncate_words(part, 150))
        else:
            shortened.append(_truncate_words(part, 120))
    compact = _clean_clause(prefix + "; ".join(shortened))
    if len(compact) <= cap:
        return compact

    required = [
        _truncate_words(next((item for item in shortened if item.lower().startswith("visual distribution lock:")), ""), 100),
        _truncate_words(next((item for item in shortened if item.lower().startswith("bright chroma primary:")), ""), 115),
        _truncate_words(next((item for item in shortened if item.lower().startswith("bounded multihue material fields:")), ""), 125),
        _truncate_words(next((item for item in shortened if item.lower().startswith("pale/base local:")), ""), 90),
        _truncate_words(next((item for item in shortened if item.lower().startswith("linear/accent only:")), ""), 90),
        _truncate_words(next((item for item in shortened if item.lower().startswith("dark/base support:")), ""), 90),
        "scene carrier map: large carrier slots = open-light/background plane, floor/reflection band, large material face",
        "anti-collapse: strips/signs/glints/partition lines cannot replace main carrier area",
    ]
    required = [item for item in required if item]
    return _clean_clause(prefix + "; ".join(required))


def _compact_field_precedence_application_clause(clause: str, cap: int) -> str:
    prefix = "Field precedence: "
    body = _strip_prompt_prefix(clause, "field precedence:")
    parts = [_clean_clause(part) for part in body.split(";") if _clean_clause(part)]
    if not parts:
        return _truncate_words(clause, cap)

    wanted_labels = (
        "visual distribution lock:",
        "field anchors:",
        "bounded multihue fields:",
        "carrier field map:",
        "linear/accent guard:",
        "pale/base guard:",
        "dark/partition guard:",
    )
    selected: List[str] = []
    for label in wanted_labels:
        part = next((item for item in parts if item.lower().startswith(label)), "")
        if part:
            selected.append(part)
    if not selected:
        return _truncate_words(clause, cap)

    compact = _clean_clause(prefix + "; ".join(selected))
    if len(compact) <= cap:
        return compact

    shortened: List[str] = []
    for part in selected:
        lowered = part.lower()
        if lowered.startswith("visual distribution lock:"):
            shortened.append(_truncate_words(part, 150))
        elif lowered.startswith("field anchors:"):
            shortened.append(_truncate_words(part, 190))
        elif lowered.startswith("bounded multihue fields:"):
            bounded_part = _truncate_words(part, 230)
            if "keep separate large saturated fields" not in bounded_part.lower():
                bounded_head = _truncate_words(part.split(" keep separate", 1)[0], 180)
                bounded_part = _clean_clause(bounded_head + " keep separate large saturated fields")
            shortened.append(bounded_part)
        elif lowered.startswith("carrier field map:"):
            shortened.append(
                "carrier field map: open light/background plane, floor/reflection band, broad material face, bounded pane/patch fields"
            )
        elif lowered.startswith("linear/accent guard:"):
            shortened.append("linear/accent guard: strips/signs/glints/rim lines stay secondary; not only evidence")
        elif lowered.startswith("pale/base guard:"):
            shortened.append(_truncate_words(part, 94))
        elif lowered.startswith("dark/partition guard:"):
            shortened.append(_truncate_words(part, 104))
        else:
            shortened.append(_truncate_words(part, 90))
    compact = _clean_clause(prefix + "; ".join(shortened))
    if len(compact) <= cap:
        return compact

    required = [
        _truncate_words(next((item for item in shortened if item.lower().startswith("visual distribution lock:")), ""), 128),
        _truncate_words(next((item for item in shortened if item.lower().startswith("field anchors:")), ""), 150),
        _truncate_words(next((item for item in shortened if item.lower().startswith("bounded multihue fields:")), ""), 150),
        "carrier field map: open light/background plane, floor/reflection band, broad material face, bounded pane/patch fields",
        "linear/accent, pale/base, and dark/partition guards stay subordinate",
    ]
    required = [item for item in required if item]
    return _clean_clause(prefix + "; ".join(required))


def _compact_carrier_materialization_application_clause(clause: str, cap: int) -> str:
    prefix = "Carrier materialization: "
    body = _strip_prompt_prefix(clause, "carrier materialization:")
    parts = [_clean_clause(part) for part in body.split(";") if _clean_clause(part)]
    if not parts:
        return _truncate_words(clause, cap)

    visual = next((item for item in parts if item.lower().startswith("visual distribution lock:")), "")
    if visual:
        match = re.match(
            r"(visual distribution lock:\s*)(.*?)\s+must\s+materialize\s+as\s+the\s+largest\s+visible\s+bounded\s+carrier\s+area\b",
            visual,
            flags=re.IGNORECASE,
        )
        if match:
            family_text = match.group(2).strip(" ,.;")
            family_items = [
                _clean_clause(part)
                for part in re.split(r"\s*,\s*|\s+\band\b\s+", family_text)
                if _clean_clause(part)
            ]
            if family_items:
                family_text = _human_join(family_items[:3])
            visual = (
                f"visual distribution lock: {family_text} largest visible bounded carrier area "
                "before accents"
            )
        else:
            visual = re.sub(
                r"\bmust\s+materialize\s+as\s+the\s+",
                "",
                visual,
                flags=re.IGNORECASE,
            )
            if "largest visible bounded carrier area" not in visual.lower():
                visual = "visual distribution lock: largest visible bounded carrier area before accents"
            visual = _truncate_words(visual, 132)
    else:
        visual = "visual distribution lock: largest visible bounded carrier area before accents"

    selected = [
        visual,
        (
            "bounded carrier geometry: large wall/background planes, floor/reflection bands, "
            "open-light fields"
        ),
        "material evidence: edges, seams, surface boundaries, reflection bands",
        "anti-accent collapse: not global tint; not neutral white glare",
    ]

    compact = _clean_clause(prefix + "; ".join(selected))
    if len(compact) <= max(cap, 370):
        return compact

    selected[0] = "visual distribution lock: dominant hues largest visible bounded carrier area before accents"
    compact = _clean_clause(prefix + "; ".join(selected))
    if len(compact) <= max(cap, 370):
        return compact

    selected[1] = "bounded carrier geometry: large wall/background planes, floor/reflection bands"
    compact = _clean_clause(prefix + "; ".join(selected))
    if len(compact) <= max(cap, 340):
        return compact

    return _truncate_words(compact, max(300, cap))


def _compact_bright_scene_recast_application_clause(clause: str, cap: int) -> str:
    prefix = "Bright-scene recast: "
    body = _strip_prompt_prefix(clause, "bright-scene recast:")
    parts = [_clean_clause(part) for part in body.split(";") if _clean_clause(part)]
    if not parts:
        return _truncate_words(clause, cap)
    visual = next((item for item in parts if item.lower().startswith("visual distribution lock:")), "")
    selected: List[str] = []
    if visual:
        match = re.match(
            r"(visual distribution lock:\s*)(.*?)\s+"
            r"(?:colored\s+)?high-value\s+(?:warm\s+hue\s+families|light\s+fields|warm\s+carriers)\b",
            visual,
            flags=re.IGNORECASE,
        )
        if match:
            family_text = match.group(2).strip(" ,.;")
            visual = (
                f"visual distribution lock: {family_text} colored high-value light fields "
                "on large scene/material carriers, not accents and not neutral white glare"
            )
        else:
            compact_slots = "large scene/material carriers, not accents and not neutral white glare"
            visual = re.sub(
                r"large scene/material carriers:.*$",
                compact_slots,
                visual,
                flags=re.IGNORECASE,
            )
            visual = re.sub(
                r"large visible scene planes:.*$",
                compact_slots,
                visual,
                flags=re.IGNORECASE,
            )
            visual = _truncate_words(visual, 170)
        selected.append(visual)

    selected.append(
        "scene carrier map: open light planes, floor/wet reflections, sign/light surfaces, "
        "wall/ceiling/background planes, large object material surfaces"
    )
    selected.append(
        "color/value distribution: bright hue weight before white/black wash or local accent collapse"
    )

    compact_lighting = (
        "lighting structure: broad cast light and overexposed edges stay colored"
    )
    compact_arbitration = (
        "carrier arbitration: small selected/accent roles cannot shrink these bright carriers"
    )
    for optional in (compact_lighting, compact_arbitration):
        candidate = _clean_clause(prefix + "; ".join(selected + [optional]))
        if len(candidate) <= max(cap, 430):
            selected.append(optional)

    compact = _clean_clause(prefix + "; ".join(selected))
    if len(compact) <= max(cap, 430):
        return compact

    required = selected[:3]
    compact = _clean_clause(prefix + "; ".join(required))
    if len(compact) <= max(cap, 430):
        return compact

    required[0] = _truncate_words(required[0], 145)
    return _clean_clause(prefix + "; ".join(required))


def _compact_perceptual_carrier_anchor_application_clause(clause: str, cap: int) -> str:
    prefix = "Perceptual carrier anchors: "
    body = _strip_prompt_prefix(clause, "perceptual carrier anchors:")
    parts = [_clean_clause(part) for part in body.split(";") if _clean_clause(part)]
    if not parts:
        return _truncate_words(clause, cap)
    selected: List[str] = []
    required_labels = (
        "visual distribution lock:",
        "color/value distribution:",
        "scene carrier map:",
        "dominant chromatic field:",
    )
    for label in required_labels:
        part = next((item for item in parts if item.lower().startswith(label)), "")
        if not part:
            continue
        used = len(prefix) + len("; ".join(selected))
        remaining = cap - used - (2 if selected else 0)
        if remaining < len(label) + 18:
            continue
        selected.append(_truncate_words(part, remaining))
    if selected:
        return _clean_clause(prefix + "; ".join(selected))
    return _truncate_words(clause, cap)


def _compact_perceptual_hue_carrier_application_clause(clause: str, cap: int) -> str:
    prefix = "Perceptual hue carriers: "
    body = _strip_prompt_prefix(clause, "perceptual hue carriers:")
    parts = [_clean_clause(part) for part in body.split(";") if _clean_clause(part)]
    if not parts:
        return _truncate_words(clause, cap)
    selected: List[str] = []
    required_labels = (
        "visual distribution lock:",
        "dominant chromatic field:",
        "scene carrier map:",
        "color/value distribution:",
    )
    for label in required_labels:
        part = next((item for item in parts if item.lower().startswith(label)), "")
        if not part:
            continue
        used = len(prefix) + len("; ".join(selected))
        remaining = cap - used - (2 if selected else 0)
        if remaining < len(label) + 18:
            continue
        selected.append(_truncate_words(part, remaining))
    if selected:
        return _clean_clause(prefix + "; ".join(selected))
    return _truncate_words(clause, cap)


def _compact_scene_slot_carrier_binding_application_clause(clause: str, cap: int) -> str:
    prefix = "Scene-slot carrier binding: "
    body = _strip_prompt_prefix(clause, "scene-slot carrier binding:")
    parts = [_clean_clause(part) for part in body.split(";") if _clean_clause(part)]
    if not parts:
        return _truncate_words(clause, cap)
    selected: List[str] = []
    required_labels = (
        "visual distribution lock:",
        "dominant chromatic field:",
        "scene carrier map:",
        "color/value distribution:",
    )
    for label in required_labels:
        part = next((item for item in parts if item.lower().startswith(label)), "")
        if not part:
            continue
        used = len(prefix) + len("; ".join(selected))
        remaining = cap - used - (2 if selected else 0)
        if remaining < len(label) + 18:
            continue
        selected.append(_truncate_words(part, remaining))
    if selected:
        return _clean_clause(prefix + "; ".join(selected))
    return _truncate_words(clause, cap)


def _compact_visual_distribution_application_clause(clause: str, cap: int) -> str:
    prefix = "Visual distribution lock: "
    body = _strip_prompt_prefix(clause, "visual distribution lock:")
    parts = [_clean_clause(part) for part in body.split(";") if _clean_clause(part)]
    if not parts:
        return _truncate_words(clause, cap)
    selected: List[str] = []
    for part in parts:
        used = len(prefix) + len("; ".join(selected))
        remaining = cap - used - (2 if selected else 0)
        if remaining < 34:
            break
        compact_part = _compact_visual_distribution_part(part, remaining)
        if compact_part:
            selected.append(compact_part)
    compact = prefix + "; ".join(part for part in selected if part).strip(" ,.;")
    return _truncate_words(compact, cap)


def _compact_visual_distribution_part(part: str, remaining: int) -> str:
    clean = _clean_clause(part)
    if not clean:
        return ""
    if len(clean) <= remaining:
        return clean
    lowered = clean.lower()
    replacements = (
        (
            "large planes become high-chroma carriers",
            "large planes become chroma carriers, not accents/wash",
        ),
        (
            "hue families occupy broad visible carrier regions",
            clean.replace("hue families occupy broad visible carrier regions", "hues occupy broad carrier regions"),
        ),
        (
            "low-key near-monochrome values keep readable midtones and bright anchors",
            "low-key values keep readable midtones and bright anchors",
        ),
        (
            "avoid shadow crush and flat black fields",
            "avoid shadow crush/flat black",
        ),
    )
    for needle, replacement in replacements:
        if needle in lowered and len(replacement) <= remaining:
            return replacement
    return _truncate_words(clean, remaining)


def _compact_dominant_chromatic_application_clause(clause: str, cap: int) -> str:
    prefix = "Dominant chromatic field: "
    body = _strip_prompt_prefix(clause, "dominant chromatic field:")
    parts = [_clean_clause(part) for part in body.split(";") if _clean_clause(part)]
    if not parts:
        return _truncate_words(clause, cap)
    guard = next(
        (
            part
            for part in parts
            if "chromatic before white" in part.lower()
            or "hue weight" in part.lower()
            or "tiny accents" in part.lower()
        ),
        "",
    )
    if guard and len(prefix) + len(parts[0]) + 2 + len(guard) > cap:
        compact_guard = (
            "high-value color stays chromatic"
            if "chromatic before white" in guard.lower()
            else _compact_dominant_chromatic_part(guard, 44)
        )
        field_cap = max(34, cap - len(prefix) - len(compact_guard) - 2)
        field = _compact_dominant_chromatic_part(parts[0], field_cap)
        if len(prefix) + len(field) + 2 + len(compact_guard) > cap:
            field = (
                "colored light fills broad planes"
                if "colored light" in parts[0].lower()
                else "chromatic fields occupy broad planes"
            )
        selected = [field]
        used = len(prefix) + len("; ".join(selected))
        remaining = cap - used - 2
        if compact_guard and remaining >= 28:
            selected.append(_truncate_words(compact_guard, remaining))
        compact = prefix + "; ".join(part for part in selected if part).strip(" ,.;")
        return _truncate_words(compact, cap)
    selected: List[str] = []
    field = parts[0]
    if len(prefix) + len(field) > cap:
        field = _compact_dominant_chromatic_part(field, max(36, cap - len(prefix)))
    selected.append(field)
    for part in parts[1:]:
        lowered = part.lower()
        if "chromatic before white" in lowered or "hue weight" in lowered or "tiny accents" in lowered:
            used = len(prefix) + len("; ".join(selected))
            remaining = cap - used - 2
            if remaining < 34:
                continue
            selected.append(_compact_dominant_chromatic_part(part, remaining))
    compact = prefix + "; ".join(part for part in selected if part).strip(" ,.;")
    return _truncate_words(compact, cap)


def _compact_dominant_chromatic_part(part: str, remaining: int) -> str:
    clean = _clean_clause(part)
    if not clean:
        return ""
    if len(clean) <= remaining:
        return clean
    lowered = clean.lower()
    replacements = (
        (
            "colored light fields occupy broad visible planes",
            "colored light fills broad planes",
        ),
        (
            "colored light islands stay visibly chromatic",
            "colored light islands stay chromatic",
        ),
        (
            "chromatic fields occupy broad scene/material planes",
            "chromatic fields occupy broad planes",
        ),
        (
            "chromatic fields remain carrier-sized, not accents",
            "chromatic fields stay carrier-sized",
        ),
        (
            "high-value color stays chromatic before white, not neutral glare or pale wash",
            "high-value color stays chromatic, not neutral glare/white wash",
        ),
        (
            "broad light fields must carry hue weight before becoming white",
            "broad light carries hue before white",
        ),
        (
            "dominant color cannot collapse into tiny accents, thin lines, or local object trim",
            "dominant color cannot collapse to accents/trim",
        ),
    )
    for needle, replacement in replacements:
        if needle in lowered:
            compact = clean.replace(needle, replacement)
            if len(compact) <= remaining:
                return compact
            return _truncate_words(replacement, remaining)
    return _truncate_words(clean, remaining)


def _compact_scene_carrier_application_clause(clause: str, cap: int) -> str:
    prefix = "Scene carrier map: "
    body = _strip_prompt_prefix(clause, "scene carrier map:")
    parts = [_clean_clause(part) for part in body.split(";") if _clean_clause(part)]
    if not parts:
        return _truncate_words(clause, cap)
    selected: List[str] = []
    for part in parts:
        used = len(prefix) + len("; ".join(selected))
        remaining = cap - used - (2 if selected else 0)
        if remaining < 34:
            break
        compact_part = _compact_scene_carrier_part(part, remaining)
        if compact_part:
            selected.append(compact_part)
    compact = prefix + "; ".join(part for part in selected if part).strip(" ,.;")
    return _truncate_words(compact, cap)


def _compact_scene_carrier_part(part: str, remaining: int) -> str:
    clean = _clean_clause(part)
    if not clean:
        return ""
    if len(clean) <= remaining:
        return clean
    lowered = clean.lower()
    replacements = (
        (
            "distribution maps onto broad background/environment planes, material surfaces, and repeated patches",
            "distribution maps to broad scene/material planes",
        ),
        (
            "large light/shadow fields may carry chroma weight before local accents",
            "light/shadow fields carry chroma before accents",
        ),
        (
            "dark fields, midtone planes, and bright islands share the value load",
            "dark/midtone/highlight carriers share value load",
        ),
        (
            "background/environment shadow carriers stay readable, not all-black",
            "shadow carriers stay readable, not all-black",
        ),
        (
            "broad light planes keep edges and chroma anchors, not white wash",
            "light planes keep edges/chroma, not white wash",
        ),
        (
            "spread value contrast across scene planes, material fields, and highlight islands",
            "spread contrast across scene/material planes and highlights",
        ),
    )
    for needle, replacement in replacements:
        if needle in lowered:
            if needle.startswith("distribution maps"):
                compact = clean.replace(needle, replacement)
                if len(compact) <= remaining:
                    return compact
                return _truncate_words("dominant hue distribution maps to broad scene/material planes", remaining)
            return _truncate_words(replacement, remaining)
    return _truncate_words(clean, remaining)


def _compact_mechanism_application_clause(clause: str, cap: int) -> str:
    prefix = "Mechanism transfer: "
    body = _strip_prompt_prefix(clause, "mechanism transfer:")
    parts = [_clean_clause(part) for part in body.split(";") if _clean_clause(part)]
    if not parts:
        return _truncate_words(clause, cap)

    recast = next((part for part in parts if "requested" in part.lower() and "carrier" in part.lower()), parts[0])
    carrier = next(
        (
            part
            for part in parts
            if part != recast and " not " in f" {part.lower()} "
        ),
        "",
    )
    if not carrier:
        carrier = next(
        (
            part
            for part in parts
            if part != recast
            and (
                "stay on" in part.lower()
                or "bound" in part.lower()
                or "carrier" in part.lower()
            )
        ),
        "",
        )
    if "preserve requested object identities/actions" in recast.lower():
        recast = "Preserve requested object identities/actions through extracted carriers"
    else:
        recast = _truncate_words(recast, 74)
    clauses = [recast]
    remaining = cap - len(prefix) - len(recast) - 2
    if carrier and remaining >= 44:
        clauses.append(_truncate_preserving_not_tail(carrier, remaining))
    compact = prefix + "; ".join(part for part in clauses if part).strip(" ,.;")
    return _truncate_words(compact, cap)


def _compact_geometry_application_clause(clause: str, cap: int) -> str:
    prefix = "Geometry grammar: "
    body = _strip_prompt_prefix(clause, "geometry grammar:")
    parts = [_clean_clause(part) for part in body.split(";") if _clean_clause(part)]
    if not parts:
        return _truncate_words(clause, cap)
    base = "one grammar across objects/background"
    partitions = next((part for part in parts if part.lower().startswith("partitions:")), "")
    primitives = next((part for part in parts if part.lower().startswith("primitives:")), "")
    edge = next((part for part in parts if part.lower().startswith("edge profile:")), "")
    selected = [base]
    for part in (primitives, partitions, edge):
        if not part:
            continue
        used = len(prefix) + len("; ".join(selected))
        remaining = cap - used - 2
        if remaining < 28:
            break
        selected.append(_truncate_words(part, remaining))
    compact = prefix + "; ".join(part for part in selected if part).strip(" ,.;")
    return _truncate_words(compact, cap)


def _compact_material_process_application_clause(clause: str, cap: int) -> str:
    prefix = "Material process evidence: "
    body = _strip_prompt_prefix(clause, "material process evidence:")
    parts = [_clean_process_label(part) for part in body.split(";")]
    parts = [part for part in parts if part]
    if not parts:
        return _truncate_words(clause, cap)
    base = "object/background process evidence"
    selected = [base]
    for label in ("edges:", "overlap/shadow:", "surface artifacts:", "partitions:", "light/shadow binding:"):
        part = next((candidate for candidate in parts if candidate.lower().startswith(label)), "")
        if not part:
            continue
        used = len(prefix) + len("; ".join(selected))
        remaining = cap - used - 2
        if remaining < 30:
            break
        selected.append(_truncate_words(part, remaining))
    compact = prefix + "; ".join(part for part in selected if part).strip(" ,.;")
    return _truncate_words(compact, cap)


def _compact_medium_application_clause(clause: str, cap: int) -> str:
    prefix = "Medium construction: "
    body = _strip_prompt_prefix(clause, "medium construction:")
    parts = [_clean_process_label(part) for part in body.split(";")]
    parts = [part for part in parts if part]
    if not parts:
        return _truncate_words(clause, cap)
    base = "foreground objects/background expose units"
    selected = [base]
    for label in ("surface units:", "edge/process:", "coverage:"):
        part = next((candidate for candidate in parts if candidate.lower().startswith(label)), "")
        if not part:
            continue
        used = len(prefix) + len("; ".join(selected))
        remaining = cap - used - 2
        if remaining < 30:
            break
        if label == "coverage:" and any(candidate.lower().startswith(("surface units:", "edge/process:")) for candidate in selected):
            continue
        selected.append(_truncate_words(part, remaining))
    compact = prefix + "; ".join(part for part in selected if part).strip(" ,.;")
    return _truncate_words(compact, cap)


def _strip_prompt_prefix(clause: str, prefix: str) -> str:
    clean = _clean_clause(clause)
    lowered = clean.lower()
    if lowered.startswith(prefix):
        return clean[len(prefix):].strip(" ,.;")
    return clean


def _clean_process_label(text: str) -> str:
    clean = _clean_clause(text)
    if re.match(r"^[A-Za-z /-]+:\s*$", clean):
        return ""
    return clean


def _truncate_preserving_not_tail(text: str, limit: int) -> str:
    clean = _clean_clause(text)
    if len(clean) <= limit:
        return clean
    lowered = clean.lower()
    not_index = lowered.find(" not ")
    if not_index < 0:
        return _truncate_words(clean, limit)
    tail = clean[not_index:].strip(" ,.;")
    if len(tail) + 8 >= limit:
        return _truncate_words(clean, limit)
    head_limit = max(24, limit - len(tail) - 2)
    head = _truncate_words(clean[:not_index], head_limit)
    return _clean_clause(f"{head}, {tail}")


def _application_minimum_total(
    items: Sequence[Tuple[str, str]],
    marker_limits: Mapping[str, Tuple[int, int]],
) -> int:
    total = 2 * max(0, len(items) - 1)
    for marker, clause in items:
        min_limit, _max_limit = marker_limits.get(marker, (96, 150))
        total += min(len(clause), max(len(marker) + 24, min_limit))
    return total


def _mechanism_transfer_from_schema(value: Any) -> Dict[str, str]:
    if not isinstance(value, Mapping):
        return {}
    out: Dict[str, str] = {}
    for key in (
        "construction_rule",
        "carrier_binding_rule",
        "content_recasting_rule",
        "application_scope",
    ):
        text = _clean_clause(value.get(key) or "")
        if text:
            out[key] = text
    return out


def _build_mechanism_transfer(
    inner: Mapping[str, Any],
    color_light_allocation: Sequence[Mapping[str, Any]],
) -> Dict[str, str]:
    direct = _mechanism_transfer_from_schema(inner.get("mechanism_transfer"))
    mechanism_text = _join_non_empty(
        _first_non_empty(inner, "medium_mark_system", "surface_material_mechanism"),
        _first_non_empty(inner, "shape_proportion_language"),
        _first_non_empty(inner, "surface_artifact", "texture_or_artifact_mechanism"),
        default="Use extracted material/process construction as the way requested content is visibly made.",
    )
    carrier_text = _first_non_empty(
        inner,
        "color_carrier_binding",
        default="Bind each extracted color, light, and material role to its own carrier, scale, and geometry.",
    )
    if color_light_allocation:
        carrier_text = _join_non_empty(
            carrier_text,
            "Color/light roles stay on extracted carriers and do not become unrelated object color or global tint.",
        )
    if direct:
        direct.setdefault("construction_rule", mechanism_text)
        direct.setdefault("carrier_binding_rule", carrier_text)
        direct.setdefault("content_recasting_rule", _CONTENT_RECASTING_CLAUSE)
        direct.setdefault(
            "application_scope",
            "Apply the extracted material/process construction across subjects, objects, and background planes when those carriers are visible.",
        )
        return _mechanism_transfer_from_schema(direct)
    return _mechanism_transfer_from_schema(
        {
            "construction_rule": mechanism_text,
            "carrier_binding_rule": carrier_text,
            "content_recasting_rule": _CONTENT_RECASTING_CLAUSE,
            "application_scope": (
                "Apply the extracted material/process construction across subjects, objects, and background planes "
                "when those carriers are visible."
            ),
        }
    )


def _compile_mechanism_transfer_prompt(
    mechanism_transfer: Mapping[str, Any],
    color_light_allocation: Sequence[Mapping[str, Any]],
    *,
    literal_items: Sequence[str],
    max_chars: int = 220,
) -> str:
    data = _mechanism_transfer_from_schema(mechanism_transfer)
    if not data:
        return ""
    clauses: List[str] = []
    for key in (
        "content_recasting_rule",
        "carrier_binding_rule",
        "application_scope",
        "construction_rule",
    ):
        if key == "content_recasting_rule":
            text = _CONTENT_RECASTING_CLAUSE
        else:
            text = _sanitize_positive_text(str(data.get(key) or ""), literal_items, remove_soft=True)
            if key == "carrier_binding_rule":
                text = _truncate_words(text, 118)
            elif key == "application_scope":
                text = _truncate_words(text, 88)
            elif key == "construction_rule":
                text = _truncate_words(text, 96)
        if text and not _looks_like_literal_container(text):
            _append_unique_nonredundant(clauses, text)
    if not clauses:
        return ""
    body = _join_limited(clauses, max(80, int(max_chars) - len("Mechanism transfer: ")), separator="; ")
    return _clean_clause("Mechanism transfer: " + body)


def _color_value_distribution_from_schema(value: Any) -> Dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    out: Dict[str, Any] = {}
    for key in (
        "value_key",
        "dark_field_role",
        "light_field_role",
        "chroma_key",
        "chromatic_role",
        "high_chroma_role",
        "chromatic_light_role",
        "contrast_key",
        "value_contrast_role",
    ):
        text = _clean_clause(value.get(key) or "")
        if text:
            out[key] = text
    families = value.get("dominant_hue_families")
    if isinstance(families, Sequence) and not isinstance(families, (str, bytes, bytearray)):
        clean_families = [_clean_clause(item) for item in families if _clean_clause(item)]
        if clean_families:
            out["dominant_hue_families"] = _dedupe(clean_families)[:4]
    light_families = value.get("dominant_light_hue_families")
    if isinstance(light_families, Sequence) and not isinstance(light_families, (str, bytes, bytearray)):
        clean_light_families = [_clean_clause(item) for item in light_families if _clean_clause(item)]
        if clean_light_families:
            out["dominant_light_hue_families"] = _dedupe(clean_light_families)[:4]
    for key in (
        "luma_mean",
        "luma_p05",
        "luma_p95",
        "luma_range_p95_p05",
        "dark_fraction",
        "light_fraction",
        "saturation_mean",
        "high_saturation_fraction",
        "strong_saturation_fraction",
        "chromatic_fraction",
        "chromatic_light_fraction",
        "dark_cluster_fraction",
        "white_cluster_fraction",
        "chromatic_cluster_fraction",
        "chromatic_light_cluster_fraction",
    ):
        raw = value.get(key)
        if isinstance(raw, (int, float)):
            out[key] = float(raw)
    return out


def _compile_visual_distribution_priority_prompt(distribution: Mapping[str, Any], *, max_chars: int = 200) -> str:
    data = _color_value_distribution_from_schema(distribution)
    if not data:
        return ""
    value_key = str(data.get("value_key") or "").lower()
    dark_role = str(data.get("dark_field_role") or "").lower()
    light_role = str(data.get("light_field_role") or "").lower()
    chroma_key = str(data.get("chroma_key") or "").lower()
    chromatic_role = str(data.get("chromatic_role") or "").lower()
    high_chroma_role = str(data.get("high_chroma_role") or "").lower()
    value_contrast_role = str(data.get("value_contrast_role") or "").lower()
    families = [str(item).lower() for item in data.get("dominant_hue_families") or [] if str(item).strip()]
    family_text = _human_join(families)

    clauses: List[str] = []
    if chroma_key == "near-monochrome":
        if "low" in value_key or "dark" in dark_role:
            clauses.append("low-key near-monochrome values keep readable midtones and bright anchors")
        else:
            clauses.append("near-monochrome values keep grayscale structure")
        if family_text:
            clauses.append(f"only extracted {family_text} hue carriers keep color")
        clauses.append("avoid shadow crush and flat black fields")
    else:
        needs_chroma_area = (
            chroma_key == "high-chroma"
            or "large chromatic" in chromatic_role
            or "saturated" in high_chroma_role
        )
        if needs_chroma_area:
            if family_text:
                clauses.append(f"{family_text} hue families occupy broad visible carrier regions")
            else:
                clauses.append("reference hue families occupy broad visible carrier regions")
            clauses.append("large planes become high-chroma carriers, not small accents or white wash")
        elif family_text:
            clauses.append(f"{family_text} hue families remain visible on repeated carrier areas")

    if "broad" in light_role or "white base" in light_role or value_key == "high-key":
        clauses.append("light fields keep chromatic carrier weight, not white wash")
    if "dominant" in dark_role or "major" in dark_role or value_key == "low-key":
        clauses.append("dark fields keep readable midtones and bright anchors")
    elif "wide" in value_contrast_role:
        clauses.append("wide value range keeps dark anchors, midtones, and bright islands")

    if not clauses:
        return ""
    body = _join_limited(_dedupe(clauses), max(80, int(max_chars) - len("Visual distribution lock: ")), separator="; ")
    return _clean_clause("Visual distribution lock: " + body)


def _compile_t4_carrier_area_actuator_prompt(distribution: Mapping[str, Any], *, max_chars: int = 360) -> str:
    data = _color_value_distribution_from_schema(distribution)
    if not data:
        return ""
    chroma_key = str(data.get("chroma_key") or "").lower()
    value_key = str(data.get("value_key") or "").lower()
    dark_role = str(data.get("dark_field_role") or "").lower()
    families = [
        str(item).lower()
        for item in data.get("dominant_hue_families") or []
        if str(item).strip()
    ]
    light_families = [
        str(item).lower()
        for item in data.get("dominant_light_hue_families") or []
        if str(item).strip()
    ]
    family_text = _human_join(_dedupe(families + light_families))
    clauses: List[str] = []
    if chroma_key == "near-monochrome":
        if family_text:
            clauses.append(f"visual distribution lock: only extracted {family_text} hue carriers keep color")
        else:
            clauses.append("visual distribution lock: chroma stays limited to extracted carriers")
        clauses.append("scene carrier map: grayscale value load spans broad background/material planes")
        if value_key == "low-key" or "dominant" in dark_role or "major" in dark_role:
            clauses.append("color/value distribution: preserve readable dark/midtone/bright anchors, not flat black")
        else:
            clauses.append("color/value distribution: preserve readable grayscale value steps")
    else:
        hue_label = family_text or "reference-dominant hues"
        clauses.append(
            f"visual distribution lock: {hue_label} take largest compatible visible carrier areas before accents/trim/thin lines"
        )
        clauses.append("dominant chromatic field: high-value light stays chromatic")
        clauses.append(
            "scene carrier map: background planes, material surfaces, repeated patches/bands, light-facing fields"
        )
        clauses.append("color/value distribution: hue weight before accents, not white/black wash")
    body = _join_limited(_dedupe(clauses), max(120, int(max_chars) - len("Carrier-area actuator: ")), separator="; ")
    return _clean_clause("Carrier-area actuator: " + body)


def _compile_t4_1_perceptual_hue_carrier_prompt(distribution: Mapping[str, Any], *, max_chars: int = 460) -> str:
    data = _color_value_distribution_from_schema(distribution)
    if not data:
        return ""
    chroma_key = str(data.get("chroma_key") or "").lower()
    value_key = str(data.get("value_key") or "").lower()
    dark_role = str(data.get("dark_field_role") or "").lower()
    families = [
        str(item).lower()
        for item in data.get("dominant_hue_families") or []
        if str(item).strip()
    ]
    light_families = [
        str(item).lower()
        for item in data.get("dominant_light_hue_families") or []
        if str(item).strip()
    ]
    all_families = _dedupe(light_families + families)
    if chroma_key == "near-monochrome":
        family_text = _human_join(all_families)
        clauses = [
            (
                f"visual distribution lock: only extracted {family_text} hue carriers keep color"
                if family_text
                else "visual distribution lock: chroma stays limited to extracted carriers"
            ),
            "scene carrier map: grayscale value load spans broad background/material planes",
        ]
        if value_key == "low-key" or "dominant" in dark_role or "major" in dark_role:
            clauses.append("color/value distribution: preserve readable dark/midtone/bright anchors, not flat black")
        else:
            clauses.append("color/value distribution: preserve readable grayscale value steps")
        body = _join_limited(_dedupe(clauses), max(120, int(max_chars) - len("Perceptual hue carriers: ")), separator="; ")
        return _clean_clause("Perceptual hue carriers: " + body)

    warm = [family for family in all_families if _hue_family_is_warm(family)]
    cool = [family for family in all_families if _hue_family_is_cool(family)]
    supporting = [
        family
        for family in all_families
        if family not in warm and family not in cool
    ]

    clauses: List[str] = []
    if warm:
        clauses.append(
            f"visual distribution lock: {_human_join(warm)} keep high-value chromatic light"
        )
    elif all_families:
        clauses.append(
            f"visual distribution lock: {_human_join(all_families[:3])} keep perceptually salient chromatic carriers"
        )
    else:
        clauses.append("visual distribution lock: bright chromatic carriers keep visible luminance/chroma")

    if cool:
        verb = "keeps" if len(cool) == 1 else "keep"
        clauses.append(
            f"dominant chromatic field: {_human_join(cool)} {verb} separate cool/ambient carrier coverage"
        )
    elif len(all_families) > 1:
        clauses.append(
            f"dominant chromatic field: {_human_join(all_families[:3])} keep separate carrier coverage"
        )
    else:
        clauses.append("dominant chromatic field: high-value chromatic carriers remain separate from neutral fields")

    if supporting:
        verb = "stays" if len(supporting) == 1 else "stay"
        clauses.append(
            f"scene carrier map: {_human_join(supporting[:2])} {verb} supporting, not replacement"
        )
    elif warm and cool:
        clauses.append("scene carrier map: warm/cool carriers both remain visible")
    else:
        clauses.append("scene carrier map: each dominant hue family keeps its own compatible carriers")

    if warm and cool:
        clauses.append("color/value distribution: warm/cool carriers both remain visible; no dominant hue family absorbs another; not gray/green/white wash")
    else:
        clauses.append("color/value distribution: no dominant hue family absorbs another; not gray/green/white wash")

    body = _join_limited(_dedupe(clauses), max(120, int(max_chars) - len("Perceptual hue carriers: ")), separator="; ")
    return _clean_clause("Perceptual hue carriers: " + body)


def _compile_t4_2_scene_slot_carrier_binding_prompt(distribution: Mapping[str, Any], *, max_chars: int = 560) -> str:
    data = _color_value_distribution_from_schema(distribution)
    if not data:
        return ""
    chroma_key = str(data.get("chroma_key") or "").lower()
    value_key = str(data.get("value_key") or "").lower()
    dark_role = str(data.get("dark_field_role") or "").lower()
    families = [
        str(item).lower()
        for item in data.get("dominant_hue_families") or []
        if str(item).strip()
    ]
    light_families = [
        str(item).lower()
        for item in data.get("dominant_light_hue_families") or []
        if str(item).strip()
    ]
    all_families = _dedupe(light_families + families)
    family_text = _human_join(all_families)
    slot_text = (
        "floor/ground planes, wet/reflective surfaces, open light planes, "
        "sign/light surfaces, wall/ceiling/background planes, and large object material surfaces"
    )

    if chroma_key == "near-monochrome":
        clauses = [
            (
                f"visual distribution lock: only extracted {family_text} hue carriers keep color"
                if family_text
                else "visual distribution lock: chroma stays limited to extracted carriers"
            ),
            f"scene carrier map: grayscale value load maps to largest requested-scene slots: {slot_text}",
        ]
        if value_key == "low-key" or "dominant" in dark_role or "major" in dark_role:
            clauses.append("color/value distribution: readable dark/midtone/bright anchors, not flat black/white wash")
        else:
            clauses.append("color/value distribution: readable grayscale value steps, not flat black/white wash")
        body = _join_limited(
            _dedupe(clauses),
            max(150, int(max_chars) - len("Scene-slot carrier binding: ")),
            separator="; ",
        )
        return _clean_clause("Scene-slot carrier binding: " + body)

    warm = [family for family in all_families if _hue_family_is_warm(family)]
    cool = [family for family in all_families if _hue_family_is_cool(family)]
    supporting = [
        family
        for family in all_families
        if family not in warm and family not in cool
    ]
    warm_text = _human_join(warm)
    cool_text = _human_join(cool)
    supporting_text = _human_join(supporting[:2])
    primary_text = warm_text or family_text or "bright chromatic"

    clauses: List[str] = [
        (
            f"visual distribution lock: {primary_text} high-value warm carriers map to largest compatible requested-scene slots"
            if warm_text
            else f"visual distribution lock: {primary_text} carriers map to largest compatible requested-scene slots"
        )
    ]
    if cool_text:
        verb = "keeps" if len(cool) == 1 else "keep"
        clauses.append(f"dominant chromatic field: {cool_text} {verb} separate cool/ambient slots")
    elif len(all_families) > 1:
        clauses.append(f"dominant chromatic field: {_human_join(all_families[:3])} keep separate carrier slots")
    else:
        clauses.append("dominant chromatic field: high-value chromatic carriers stay separate from neutral fields")

    scene_parts = [f"scene carrier map: bind those carriers to {slot_text}"]
    if supporting_text:
        verb = "stays" if len(supporting[:2]) == 1 else "stay"
        scene_parts.append(f"{supporting_text} {verb} supporting, not replacement")
    elif warm and cool:
        scene_parts.append("warm/cool slots both remain visible")
    else:
        scene_parts.append("dominant hue families keep separate compatible slots")
    clauses.append("; ".join(scene_parts))

    if warm and cool:
        clauses.append(
            "color/value distribution: warm/cool carriers both remain visible; not accent-only; no dominant hue family absorbs another; not gray/green/white wash"
        )
    else:
        clauses.append(
            "color/value distribution: carriers remain visible; not accent-only; no dominant hue family absorbs another; not gray/white wash"
        )

    body = _join_limited(
        _dedupe(clauses),
        max(150, int(max_chars) - len("Scene-slot carrier binding: ")),
        separator="; ",
    )
    return _clean_clause("Scene-slot carrier binding: " + body)


def _compile_t4_3_perceptual_carrier_anchor_prompt(distribution: Mapping[str, Any], *, max_chars: int = 520) -> str:
    data = _color_value_distribution_from_schema(distribution)
    if not data:
        return ""
    chroma_key = str(data.get("chroma_key") or "").lower()
    value_key = str(data.get("value_key") or "").lower()
    dark_role = str(data.get("dark_field_role") or "").lower()
    families = [
        str(item).lower()
        for item in data.get("dominant_hue_families") or []
        if str(item).strip()
    ]
    light_families = [
        str(item).lower()
        for item in data.get("dominant_light_hue_families") or []
        if str(item).strip()
    ]
    all_families = _dedupe(light_families + families)
    family_text = _human_join(all_families)
    slot_text = (
        "open light planes, wet/reflective surfaces, sign/light surfaces, floor/ground reflections, "
        "wall/ceiling/background planes, and large object material surfaces"
    )

    if chroma_key == "near-monochrome":
        clauses = [
            (
                f"visual distribution lock: only extracted {family_text} hue carriers keep color"
                if family_text
                else "visual distribution lock: chroma stays limited to extracted carriers"
            ),
            f"scene carrier map: high-value and midtone value anchors occupy broad requested-scene carriers: {slot_text}",
        ]
        if value_key == "low-key" or "dominant" in dark_role or "major" in dark_role:
            clauses.append("color/value distribution: readable dark/midtone/bright anchors, not flat black/white wash")
        else:
            clauses.append("color/value distribution: readable grayscale value steps, not flat black/white wash")
        body = _join_limited(
            _dedupe(clauses),
            max(150, int(max_chars) - len("Perceptual carrier anchors: ")),
            separator="; ",
        )
        return _clean_clause("Perceptual carrier anchors: " + body)

    warm = [family for family in all_families if _hue_family_is_warm(family)]
    cool = [family for family in all_families if _hue_family_is_cool(family)]
    supporting = [
        family
        for family in all_families
        if family not in warm and family not in cool
    ]
    anchor_families = warm or light_families or all_families[:3]
    anchor_text = _human_join(anchor_families) or "reference-dominant bright chromatic"

    clauses: List[str] = [
        (
            f"visual distribution lock: {anchor_text} high-value perceptual hue carriers must occupy visible carrier area "
            f"on {slot_text}, not only tiny lamps/signs/trim/rim accents"
        ),
        (
            "color/value distribution: keep those high-value hue carriers bright and chromatic before dark fields, "
            "white wash, neutral glare, or local accent collapse; no dominant hue family absorbs another"
        ),
        f"scene carrier map: bind perceptual hue anchors to {slot_text} before small accents",
    ]
    if cool:
        verb = "keeps" if len(cool) == 1 else "keep"
        clauses.append(f"dominant chromatic field: {_human_join(cool)} {verb} separate cool/ambient slots")
    elif len(all_families) > 1:
        clauses.append(f"dominant chromatic field: {_human_join(all_families[:3])} keep separate carrier slots")
    else:
        clauses.append("dominant chromatic field: high-value chromatic carriers stay separate from neutral fields")
    if supporting:
        verb = "stays" if len(supporting[:2]) == 1 else "stay"
        clauses.append(f"dominant chromatic field: {_human_join(supporting[:2])} {verb} supporting, not replacement")

    body = _join_limited(
        _dedupe(clauses),
        max(180, int(max_chars) - len("Perceptual carrier anchors: ")),
        separator="; ",
    )
    return _clean_clause("Perceptual carrier anchors: " + body)


def _compile_t4_4_bright_scene_recast_prompt(distribution: Mapping[str, Any], *, max_chars: int = 580) -> str:
    data = _color_value_distribution_from_schema(distribution)
    if not data:
        return ""
    chroma_key = str(data.get("chroma_key") or "").lower()
    value_key = str(data.get("value_key") or "").lower()
    dark_role = str(data.get("dark_field_role") or "").lower()
    families = [
        str(item).lower()
        for item in data.get("dominant_hue_families") or []
        if str(item).strip()
    ]
    light_families = [
        str(item).lower()
        for item in data.get("dominant_light_hue_families") or []
        if str(item).strip()
    ]
    all_families = _dedupe(light_families + families)
    family_text = _human_join(all_families)
    slot_text = (
        "large scene/material carriers: open light planes, floor/ground planes, wet/reflective surfaces, "
        "sign/light surfaces, wall/ceiling/background planes, and large object material surfaces"
    )

    if chroma_key == "near-monochrome":
        clauses = [
            (
                f"visual distribution lock: only extracted {family_text} hue carriers keep color"
                if family_text
                else "visual distribution lock: chroma stays limited to extracted carriers"
            ),
            "lighting structure: value contrast, broad cast light, overexposed edges, ambient fill, and contact shadows carry value evidence",
            f"scene carrier map: map high-value and midtone value anchors onto {slot_text}",
        ]
        if value_key == "low-key" or "dominant" in dark_role or "major" in dark_role:
            clauses.append("color/value distribution: readable dark/midtone/bright anchors, not flat black/white wash")
        else:
            clauses.append("color/value distribution: readable grayscale value steps, not flat black/white wash")
        body = _join_limited(
            _dedupe(clauses),
            max(180, int(max_chars) - len("Bright-scene recast: ")),
            separator="; ",
        )
        return _clean_clause("Bright-scene recast: " + body)

    warm = [family for family in all_families if _hue_family_is_warm(family)]
    cool = [family for family in all_families if _hue_family_is_cool(family)]
    supporting = [
        family
        for family in all_families
        if family not in warm and family not in cool
    ]
    anchor_families = warm or light_families or all_families[:3]
    anchor_text = _human_join(anchor_families) or "reference-dominant bright chromatic"

    clauses: List[str] = [
        (
            f"visual distribution lock: {anchor_text} colored high-value light fields become lighting structure "
            f"on {slot_text}"
        ),
        (
            "lighting structure: broad cast light stays colored, overexposed edges stay colored, ambient fill, "
            "and reflective bounce carry those hues; not neutral white glare"
        ),
        "carrier arbitration: small selected/accent roles cannot shrink these bright carriers; not semantic object recolor",
        f"scene carrier map: bind those bright carriers to {slot_text} before signs, trim, and local accents",
    ]
    if cool:
        verb = "keeps" if len(cool) == 1 else "keep"
        clauses.append(f"dominant chromatic field: {_human_join(cool)} {verb} separate cool/ambient slots")
    elif len(all_families) > 1:
        clauses.append(f"dominant chromatic field: {_human_join(all_families[:3])} keep separate carrier slots")
    else:
        clauses.append("dominant chromatic field: high-value chromatic carriers stay separate from neutral fields")
    if supporting:
        verb = "stays" if len(supporting[:2]) == 1 else "stay"
        clauses.append(f"dominant chromatic field: {_human_join(supporting[:2])} {verb} supporting, not replacement")
    clauses.append(
        "color/value distribution: bright chromatic carriers remain visible before dark/base fields, not gray/white wash"
    )

    body = _join_limited(
        _dedupe(clauses),
        max(220, int(max_chars) - len("Bright-scene recast: ")),
        separator="; ",
    )
    return _clean_clause("Bright-scene recast: " + body)


def _compile_t4_5_carrier_materialization_prompt(distribution: Mapping[str, Any], *, max_chars: int = 620) -> str:
    data = _color_value_distribution_from_schema(distribution)
    if not data:
        return ""
    chroma_key = str(data.get("chroma_key") or "").lower()
    value_key = str(data.get("value_key") or "").lower()
    dark_role = str(data.get("dark_field_role") or "").lower()
    families = [
        str(item).lower()
        for item in data.get("dominant_hue_families") or []
        if str(item).strip()
    ]
    light_families = [
        str(item).lower()
        for item in data.get("dominant_light_hue_families") or []
        if str(item).strip()
    ]
    all_families = _dedupe(light_families + families)
    family_text = _human_join(all_families)
    carrier_geometry = (
        "large wall/background planes, floor/reflection bands, open-light fields, "
        "emissive/sign surfaces, large object material faces, and repeated bounded surface fields"
    )

    if chroma_key == "near-monochrome":
        hue_text = family_text or "extracted chromatic accents"
        clauses = [
            (
                f"visual distribution lock: {hue_text} materialize only on bounded visible carriers, "
                "while grayscale value fields carry the largest planes"
            ),
            f"bounded carrier geometry: {carrier_geometry}",
            (
                "material evidence: edges, seams, surface boundaries, reflection bands, and contact shadows "
                "show value steps as physical carriers"
            ),
            "anti-accent collapse: extracted hue anchors stay carrier-bound; not global tint",
        ]
        if value_key == "low-key" or "dominant" in dark_role or "major" in dark_role:
            clauses.append("color/value distribution: dark/base fields keep readable bounded midtone and bright carriers")
        else:
            clauses.append("color/value distribution: broad value carriers keep visible grayscale steps")
        body = _join_limited(
            _dedupe(clauses),
            max(220, int(max_chars) - len("Carrier materialization: ")),
            separator="; ",
        )
        return _clean_clause("Carrier materialization: " + body)

    warm = [family for family in all_families if _hue_family_is_warm(family)]
    cool = [family for family in all_families if _hue_family_is_cool(family)]
    supporting = [
        family
        for family in all_families
        if family not in warm and family not in cool
    ]
    hue_text = family_text or "reference-dominant hue families"
    clauses: List[str] = [
        (
            f"visual distribution lock: {hue_text} must materialize as the largest visible bounded carrier "
            "area before accents"
        ),
        f"bounded carrier geometry: {carrier_geometry}",
        (
            "material evidence: edges, seams, surface boundaries, painted/glass/emissive surfaces, "
            "reflection bands, and contact shadows show those hue fields as physical carriers"
        ),
        (
            "anti-accent collapse: tiny accents/sign strips/rim lines are insufficient; "
            "not global tint; not neutral white glare"
        ),
    ]
    if warm and cool:
        clauses.append("dominant chromatic field: warm and cool hue families keep separate bounded carrier groups")
    elif len(all_families) > 1:
        clauses.append(f"dominant chromatic field: {_human_join(all_families[:3])} keep separate bounded carrier groups")
    else:
        clauses.append("dominant chromatic field: high-value chromatic carriers stay separate from neutral fields")
    if supporting:
        verb = "stays" if len(supporting[:2]) == 1 else "stay"
        clauses.append(f"dominant chromatic field: {_human_join(supporting[:2])} {verb} supporting, not replacement")
    clauses.append(
        f"scene carrier map: bind those hue fields to {carrier_geometry} before signs, trim, glints, and local accents"
    )
    if value_key == "low-key" or "dominant" in dark_role or "major" in dark_role:
        clauses.append(
            "color/value distribution: dark/base fields frame but do not shrink high-value chromatic carriers"
        )
    else:
        clauses.append(
            "color/value distribution: high-value hue carriers keep chroma before white wash; broad base fields remain colored"
        )

    body = _join_limited(
        _dedupe(clauses),
        max(240, int(max_chars) - len("Carrier materialization: ")),
        separator="; ",
    )
    return _clean_clause("Carrier materialization: " + body)


def _compile_t4_6_compact_topology_prompt(distribution: Mapping[str, Any], *, max_chars: int = 620) -> str:
    data = _color_value_distribution_from_schema(distribution)
    if not data:
        return ""
    chroma_key = str(data.get("chroma_key") or "").lower()
    families = [
        str(item).lower()
        for item in data.get("dominant_hue_families") or []
        if str(item).strip()
    ]
    light_families = [
        str(item).lower()
        for item in data.get("dominant_light_hue_families") or []
        if str(item).strip()
    ]
    all_families = _dedupe(light_families + families)
    family_text = _human_join(all_families) or "reference-dominant hue families"

    if chroma_key == "near-monochrome":
        clauses = [
            (
                f"visual distribution lock: {family_text} stay on extracted bounded carriers while "
                "grayscale value fields own large carrier slots"
            ),
            (
                "scene carrier map: large carrier slots = open-light/background plane, floor/reflection band, "
                "large material face, repeated bounded surface fields"
            ),
            "anti-strip collapse: chromatic anchors cannot become only linear lamps, sign strips, rim lines, or glints",
            "dark/base carriers support/partition only and keep readable midtone/bright slots",
        ]
    else:
        clauses = [
            (
                f"visual distribution lock: {family_text} must own large carrier slots before small accents"
            ),
            (
                "scene carrier map: large carrier slots = open-light/background plane, floor/reflection band, "
                "large material face, repeated bounded surface fields"
            ),
            (
                "anti-strip collapse: linear lamps, sign strips, rim lines, glints, ceiling bars, "
                "and narrow reflection streaks do not count as main dominant-hue carriers"
            ),
            (
                "color role order: colored material-light and dominant chromatic fields before pale base and "
                "dark/base; dark/base carriers support/partition only and cannot become largest filled area "
                "unless reference dark is dominant"
            ),
        ]

    body = "; ".join(_dedupe(clauses))
    body_limit = max(320, int(max_chars) - len("Carrier topology: "))
    if len(body) > body_limit:
        compact = _compact_carrier_topology_application_clause("Carrier topology: " + body, int(max_chars))
        return compact
    return _clean_clause("Carrier topology: " + body)


def _compile_t4_7_bright_chroma_carrier_prompt(
    distribution: Mapping[str, Any],
    allocation: Sequence[Mapping[str, Any]],
    *,
    max_chars: int = 620,
) -> str:
    data = _color_value_distribution_from_schema(distribution)
    if not data:
        return ""
    chroma_key = str(data.get("chroma_key") or "").lower()
    if chroma_key == "near-monochrome":
        return _compile_t4_6_compact_topology_prompt(data, max_chars=max_chars)

    families = _bright_chroma_family_list(data)
    family_text = _human_join(families) or "reference-dominant light hue families"
    anchors = _bright_chroma_anchor_items(allocation, data, max_items=3)
    anchor_text = _bright_chroma_anchor_text(anchors, limit=150)

    clauses = [
        (
            f"visual distribution lock: {family_text} must remain saturated visible carrier color "
            "on large carrier slots before pale/base/dark"
        ),
        (
            f"priority anchors: {anchor_text} hold chroma as material-light/reflection carriers"
            if anchor_text
            else "priority anchors: extracted bright chroma anchors hold chroma as material-light/reflection carriers"
        ),
        (
            "scene carrier map: large carrier slots = open-light/background plane, floor/reflection band, large material face, "
            "repeated bounded surface fields"
        ),
        (
            "anti-strip collapse: linear lamps, sign strips, rim lines, glints, ceiling bars, "
            "and narrow reflection streaks do not count; not strip-only"
        ),
        "wash guard: not cream/white/pale wash",
        "dark/base guard: dark/base supports/partitions only and cannot become largest filled area unless reference dark is dominant",
    ]
    body = "; ".join(_dedupe(clauses))
    body_limit = max(330, int(max_chars) - len("Bright chroma carriers: "))
    if len(body) > body_limit:
        return _compact_bright_chroma_carrier_application_clause(
            "Bright chroma carriers: " + body,
            int(max_chars),
        )
    return _clean_clause("Bright chroma carriers: " + body)


def _compile_t4_7_bright_chroma_color_prompt(
    distribution: Mapping[str, Any],
    allocation: Sequence[Mapping[str, Any]],
    *,
    max_chars: int = 220,
) -> str:
    data = _color_value_distribution_from_schema(distribution)
    if not data or str(data.get("chroma_key") or "").lower() == "near-monochrome":
        return ""
    anchors = _bright_chroma_anchor_items(allocation, data, max_items=3)
    if not anchors:
        families = _bright_chroma_family_list(data)
        family_text = _human_join(families) or "reference-dominant bright chroma"
        return _clean_clause(
            "bright chroma anchors: "
            + _truncate_words(
                f"{family_text} stay saturated on large material-light/reflection carriers before pale/base/dark",
                max(96, int(max_chars) - len("bright chroma anchors: ")),
            )
        )
    anchor_text = _bright_chroma_anchor_text(anchors, limit=120)
    body = _truncate_words(
        f"{anchor_text} stay saturated on large material-light/reflection carriers before pale/base/dark",
        max(110, int(max_chars) - len("bright chroma anchors: ")),
    )
    return _clean_clause("bright chroma anchors: " + body)


def _compile_t4_8_carrier_role_metadata_prompt(
    distribution: Mapping[str, Any],
    allocation: Sequence[Mapping[str, Any]],
    *,
    max_chars: int = 660,
) -> str:
    data = _color_value_distribution_from_schema(distribution)
    if not data:
        return ""
    families = _bright_chroma_family_list(data)
    family_text = _human_join(families) or "reference-dominant chromatic hue families"
    roles = _t4_8_carrier_role_groups(allocation, data)
    clauses: List[str] = [
        (
            f"visual distribution lock: {family_text} role-separated large carrier slots before "
            "pale/base, linear accents, and dark support"
        )
    ]
    bounded = roles.get("bounded_multihue_material_fields") or []
    bright = roles.get("bright_chroma_primary") or []
    pale = roles.get("pale_base_local") or []
    accent = roles.get("linear_accent_only") or []
    dark = roles.get("dark_base_support") or []

    if bounded:
        clauses.append(
            "bounded multihue material fields: "
            + _t4_8_role_label_text(bounded, limit=160)
            + " keep saturated separate large bounded surface fields before dark partitions or wash"
        )
    if bright:
        clauses.append(
            "bright chroma primary: "
            + _t4_8_role_label_text(bright, limit=150)
            + " hold saturated visible color on large material-light/reflection carriers"
        )
    if pale:
        clauses.append(
            "pale/base local: "
            + _t4_8_role_label_text(pale, limit=120)
            + " support locally and cannot dilute bright chroma primary"
        )
    if accent:
        clauses.append(
            "linear/accent only: "
            + _t4_8_role_label_text(accent, limit=120)
            + " stay tiny or linear and cannot count as dominant carrier area"
        )
    if dark:
        clauses.append(
            "dark/base support: "
            + _t4_8_role_label_text(dark, limit=120)
            + " support/partition only and cannot become largest filled area unless reference dark is dominant"
        )
    clauses.extend(
        [
            (
                "scene carrier map: large carrier slots = open-light/background plane, floor/reflection band, "
                "large material face, repeated bounded surface fields"
            ),
            (
                "anti-collapse: strips, signs, glints, partition lines, and pale wash cannot replace "
                "main saturated carrier area"
            ),
        ]
    )
    body = "; ".join(_dedupe(clauses))
    body_limit = max(360, int(max_chars) - len("Carrier role metadata: "))
    if len(body) > body_limit:
        return _compact_carrier_role_metadata_application_clause("Carrier role metadata: " + body, int(max_chars))
    return _clean_clause("Carrier role metadata: " + body)


def _compile_t4_9_field_precedence_prompt(
    distribution: Mapping[str, Any],
    allocation: Sequence[Mapping[str, Any]],
    *,
    max_chars: int = 680,
) -> str:
    data = _color_value_distribution_from_schema(distribution)
    if not data:
        return ""
    if str(data.get("chroma_key") or "").lower() == "near-monochrome":
        return ""

    families = _bright_chroma_family_list(data)
    family_text = _human_join(families) or "reference-dominant hue families"
    roles = _t4_8_carrier_role_groups(allocation, data)
    field_anchors = _t4_9_field_anchor_items(allocation, data, max_items=4)
    bounded = [
        item
        for item in roles.get("bounded_multihue_material_fields") or []
        if _allocation_item_is_field_precedence_candidate(item, data)
    ]
    pale = roles.get("pale_base_local") or []
    accent = roles.get("linear_accent_only") or []
    dark = roles.get("dark_base_support") or []

    clauses: List[str] = [
        (
            f"visual distribution lock: {family_text} must occupy largest compatible visible carrier fields "
            "before local base, linear accents, and dark support"
        )
    ]
    if field_anchors:
        clauses.append(
            "field anchors: "
            + _t4_8_role_label_text(field_anchors, limit=150)
            + " carry saturated area on broad visible fields"
        )
    else:
        clauses.append(
            "field anchors: reference-dominant hue families carry saturated area on broad visible fields"
        )
    if bounded:
        clauses.append(
            "bounded multihue fields: "
            + _t4_8_role_label_text(bounded, limit=170)
            + " keep separate large saturated fields; not one hue or neutral wash"
        )
    clauses.append(
        "carrier field map: open light/background plane, floor/reflection band, broad material face, bounded pane/patch fields"
    )
    if accent:
        clauses.append(
            "linear/accent guard: "
            + _t4_8_role_label_text(accent, limit=105)
            + " stay secondary; strips, signs, glints, and rim lines cannot be the only evidence"
        )
    else:
        clauses.append(
            "linear/accent guard: strips, signs, glints, and rim lines stay secondary and cannot be the only evidence"
        )
    if pale:
        clauses.append(
            "pale/base guard: "
            + _t4_8_role_label_text(pale, limit=105)
            + " stays local/supportive; cannot dilute saturated field anchors"
        )
    else:
        clauses.append("pale/base guard: pale or neutral bases stay local/supportive below saturated field anchors")
    if dark:
        clauses.append(
            "dark/partition guard: "
            + _t4_8_role_label_text(dark, limit=115)
            + " separate fields only; not largest area unless reference dark is dominant"
        )
    else:
        clauses.append("dark/partition guard: dark supports or separates fields without replacing saturated area")

    body = "; ".join(_dedupe(clauses))
    body_limit = max(380, int(max_chars) - len("Field precedence: "))
    if len(body) > body_limit:
        return _compact_field_precedence_application_clause("Field precedence: " + body, int(max_chars))
    return _clean_clause("Field precedence: " + body)


def _prepare_t4_9_color_allocation_for_prompt(
    allocation: Sequence[Mapping[str, Any]],
    distribution: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    prepared = _prepare_t4_8_color_allocation_for_prompt(allocation, distribution)
    field_keys = {
        _allocation_hex_key(item)
        for item in _t4_9_field_anchor_items(prepared, distribution, max_items=5)
    }
    if not field_keys:
        return prepared
    out: List[Dict[str, Any]] = []
    for item in prepared:
        clean = dict(item)
        hex_key = _allocation_hex_key(clean)
        allocation_kind = str(clean.get("allocation") or "").lower()
        if (
            hex_key in field_keys
            and (
                "medium light" in allocation_kind
                or ("dominant field" in allocation_kind and not _allocation_item_is_shadow_like(clean))
            )
        ):
            clean["allocation"] = "large material-light"
            clean["light_role"] = "material light field"
            clean["carrier_scale"] = "broad visible field"
        out.append(clean)
    return out


def _t4_9_field_anchor_items(
    allocation: Sequence[Mapping[str, Any]],
    distribution: Mapping[str, Any],
    *,
    max_items: int,
) -> List[Mapping[str, Any]]:
    data = _color_value_distribution_from_schema(distribution)
    clean_items = _dedupe_color_light_allocation(allocation)
    ranked: List[Tuple[int, int, Mapping[str, Any]]] = []
    for index, item in enumerate(clean_items):
        if not _allocation_item_is_field_precedence_candidate(item, data):
            continue
        allocation_kind = str(item.get("allocation") or "").lower()
        if "large material-light" in allocation_kind:
            rank = 0
        elif "dominant field" in allocation_kind:
            rank = 1
        elif "medium light" in allocation_kind:
            rank = 2
        elif _allocation_item_is_bounded_multihue_candidate(item, data):
            rank = 3
        elif "large material color" in allocation_kind:
            rank = 4
        else:
            rank = 5
        ranked.append((rank, index, item))
    return _dedupe_role_items([item for _rank, _index, item in sorted(ranked, key=lambda entry: (entry[0], entry[1]))])[
        :max_items
    ]


def _allocation_item_is_field_precedence_candidate(
    item: Mapping[str, Any],
    distribution: Mapping[str, Any],
) -> bool:
    if _allocation_item_is_shadow_like(item) or _allocation_item_is_pale_base_anchor(item):
        return False
    if _allocation_item_is_linear_or_accent_only(item):
        return False
    if _allocation_max_hex_saturation(item) < 0.38:
        return False
    allocation_kind = str(item.get("allocation") or "").lower()
    if not (
        "large material color" in allocation_kind
        or "large material-light" in allocation_kind
        or "dominant field" in allocation_kind
        or "medium light" in allocation_kind
    ):
        return False
    data = _color_value_distribution_from_schema(distribution)
    families = set(_bright_chroma_family_list(data))
    text = _allocation_search_text(item)
    item_families = set(_allocation_item_hue_families(item))
    if families and not (families & item_families or _text_mentions_dominant_hue_family(text, families)):
        return False
    return _allocation_item_has_large_field_scope(item)


def _allocation_item_has_large_field_scope(item: Mapping[str, Any]) -> bool:
    text = _allocation_search_text(item)
    if any(term in text for term in ("tiny", "glint", "glints", "sparkle", "sparkles", "speck", "specular")):
        return False
    if re.search(r"\bstrip|sign strip|rim line|thin line\b", text) and not re.search(
        r"\bbroad|large|field|plane|surface|reflection|reflected|bounded|repeated|patch|pane\b",
        text,
    ):
        return False
    return bool(
        re.search(
            r"\bbroad|large|continuous|field|fields|plane|planes|surface|surfaces|floor|background|reflection|reflected|"
            r"repeated|bounded|pane|panes|patch|patches|face|faces|band|bands\b",
            text,
        )
    )


def _prepare_t4_8_color_allocation_for_prompt(
    allocation: Sequence[Mapping[str, Any]],
    distribution: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    roles = _t4_8_carrier_role_groups(allocation, distribution)
    protected_keys = {
        _allocation_hex_key(item)
        for item in (roles.get("bright_chroma_primary") or []) + (roles.get("bounded_multihue_material_fields") or [])
    }
    accent_keys = {_allocation_hex_key(item) for item in roles.get("linear_accent_only") or []}
    prepared: List[Dict[str, Any]] = []
    for item in allocation:
        clean = dict(item)
        hex_key = _allocation_hex_key(clean)
        allocation_kind = str(clean.get("allocation") or "").lower()
        if hex_key in accent_keys:
            clean["allocation"] = "tiny accent"
            clean["light_role"] = "accent only"
            clean["carrier_scale"] = "tiny local"
        elif (
            hex_key not in protected_keys
            and ("large material color" in allocation_kind or "large material-light" in allocation_kind)
            and _allocation_item_is_pale_base_anchor(clean)
        ):
            clean["allocation"] = "local material color"
            clean["light_role"] = "pale/base local material color"
            clean["carrier_scale"] = "selected carrier only"
        prepared.append(clean)
    return prepared


def _t4_8_carrier_role_groups(
    allocation: Sequence[Mapping[str, Any]],
    distribution: Mapping[str, Any],
) -> Dict[str, List[Mapping[str, Any]]]:
    data = _color_value_distribution_from_schema(distribution)
    clean_items = _dedupe_color_light_allocation(allocation)
    bright = _bright_chroma_anchor_items(clean_items, data, max_items=4)
    bounded_candidates = [
        item
        for item in clean_items
        if _allocation_item_is_bounded_multihue_candidate(item, data)
    ]
    bounded_families: List[str] = []
    for item in bounded_candidates:
        for family in _allocation_item_hue_families(item):
            _append_unique(bounded_families, family)
    bounded = bounded_candidates[:4] if len(bounded_candidates) >= 2 and len(bounded_families) >= 2 else []
    pale = [
        item
        for item in clean_items
        if _allocation_item_is_pale_base_anchor(item)
    ]
    accent = [
        item
        for item in clean_items
        if _allocation_item_is_linear_or_accent_only(item)
    ]
    dark = [
        item
        for item in clean_items
        if _allocation_item_is_shadow_like(item)
        or "structural dark" in str(item.get("allocation") or "").lower()
    ]
    return {
        "bright_chroma_primary": _dedupe_role_items(bright),
        "bounded_multihue_material_fields": _dedupe_role_items(bounded),
        "pale_base_local": _dedupe_role_items(pale),
        "linear_accent_only": _dedupe_role_items(accent),
        "dark_base_support": _dedupe_role_items(dark),
    }


def _dedupe_role_items(items: Sequence[Mapping[str, Any]]) -> List[Mapping[str, Any]]:
    out: List[Mapping[str, Any]] = []
    seen = set()
    for item in items:
        key = _allocation_hex_key(item) or _sanitize_prompt_fragment_text(item.get("label") or "").lower()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def _allocation_hex_key(item: Mapping[str, Any]) -> Tuple[str, ...]:
    return tuple(hex_value.lower() for hex_value in _dedupe_hexes(item.get("hexes") or []))


def _t4_8_role_label_text(items: Sequence[Mapping[str, Any]], *, limit: int) -> str:
    labels = [
        _sanitize_prompt_fragment_text(item.get("label") or "")
        for item in items
        if _sanitize_prompt_fragment_text(item.get("label") or "")
    ]
    return _join_limited(labels, limit, separator=", ") or "reference-derived carrier roles"


def _allocation_item_is_bounded_multihue_candidate(item: Mapping[str, Any], distribution: Mapping[str, Any]) -> bool:
    if _allocation_item_is_shadow_like(item) or _allocation_item_is_pale_base_anchor(item):
        return False
    if _allocation_item_is_linear_or_accent_only(item):
        return False
    allocation_kind = str(item.get("allocation") or "").lower()
    if not (
        "large material color" in allocation_kind
        or "large material-light" in allocation_kind
        or "dominant field" in allocation_kind
    ):
        return False
    if _allocation_max_hex_saturation(item) < 0.5:
        return False
    text = _allocation_search_text(item)
    if not any(
        term in text
        for term in (
            "bounded",
            "facet",
            "facets",
            "surface field",
            "surface fields",
            "material field",
            "material fields",
            "large surface face",
            "large surface faces",
            "translucent",
            "repeated",
        )
    ):
        return False
    families = set(_bright_chroma_family_list(distribution))
    item_families = set(_allocation_item_hue_families(item))
    return bool(item_families) and (not families or bool(families & item_families) or _text_mentions_dominant_hue_family(text, families))


def _allocation_item_is_linear_or_accent_only(item: Mapping[str, Any]) -> bool:
    allocation_kind = str(item.get("allocation") or "").lower()
    text = _allocation_search_text(item)
    scale = str(item.get("carrier_scale") or "").lower()
    if "tiny" in allocation_kind or "small" in allocation_kind:
        return True
    if any(term in text for term in ("tiny", "glint", "glints", "sparkle", "sparkles", "speck", "specular")):
        return True
    linear_terms = ("strip", "strips", "sign strip", "rim line", "rim lines", "thin line", "thin lines")
    broad_terms = ("broad", "large", "medium", "repeated", "plane", "planes", "field", "fields", "surface")
    if any(term in text for term in linear_terms) and not any(term in scale or term in text for term in broad_terms):
        return True
    return False


def _prepare_t4_7_color_allocation_for_prompt(
    allocation: Sequence[Mapping[str, Any]],
    distribution: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    bright_items = _bright_chroma_anchor_items(allocation, distribution, max_items=4)
    bright_hex_keys = {
        tuple(hex_value.lower() for hex_value in _dedupe_hexes(item.get("hexes") or []))
        for item in bright_items
    }
    prepared: List[Dict[str, Any]] = []
    for item in allocation:
        clean = dict(item)
        hex_key = tuple(hex_value.lower() for hex_value in _dedupe_hexes(clean.get("hexes") or []))
        allocation_kind = str(clean.get("allocation") or "").lower()
        if (
            hex_key not in bright_hex_keys
            and ("large material color" in allocation_kind or "large material-light" in allocation_kind)
            and _allocation_item_is_pale_base_anchor(clean)
        ):
            clean["allocation"] = "local material color"
            clean["light_role"] = "pale/base local material color"
            clean["carrier_scale"] = "selected carrier only"
        prepared.append(clean)
    return prepared


def _allocation_item_is_pale_base_anchor(item: Mapping[str, Any]) -> bool:
    text = _allocation_search_text(item)
    if not any(term in text for term in ("cream", "ivory", "pale", "white base", "soft base", "base surfaces")):
        return False
    if any(term in text for term in ("cyan", "teal", "orange", "red", "emissive", "amber")):
        return False
    return _allocation_max_hex_saturation(item) < 0.5


def _bright_chroma_family_list(distribution: Mapping[str, Any]) -> List[str]:
    light_families = [
        str(item).lower()
        for item in distribution.get("dominant_light_hue_families") or []
        if str(item).strip()
    ]
    families = [
        str(item).lower()
        for item in distribution.get("dominant_hue_families") or []
        if str(item).strip()
    ]
    return _dedupe(light_families + families)[:3]


def _bright_chroma_anchor_items(
    allocation: Sequence[Mapping[str, Any]],
    distribution: Mapping[str, Any],
    *,
    max_items: int,
) -> List[Mapping[str, Any]]:
    families = set(_bright_chroma_family_list(distribution))
    clean_items = _dedupe_color_light_allocation(allocation)
    ranked: List[Tuple[int, int, Mapping[str, Any]]] = []
    for index, item in enumerate(clean_items):
        if _allocation_item_is_shadow_like(item):
            continue
        max_sat = _allocation_max_hex_saturation(item)
        if max_sat < 0.38:
            continue
        text = _allocation_search_text(item)
        allocation_kind = str(item.get("allocation") or "").lower()
        if "tiny" in allocation_kind or "small" in allocation_kind or "local" in allocation_kind:
            continue
        if any(term in text for term in ("tiny", "glint", "glints", "sparkle", "sparkles", "speck", "specular")):
            continue
        item_families = set(_allocation_item_hue_families(item))
        hue_match = bool(families & item_families) or _text_mentions_dominant_hue_family(text, families)
        light_like = (
            "material-light" in allocation_kind
            or "medium light" in allocation_kind
            or "light" in str(item.get("light_role") or "").lower()
            or "light" in text
            or "reflection" in text
            or "emissive" in text
        )
        chroma_field = "dominant field" in allocation_kind and hue_match
        material_color = "large material color" in allocation_kind and hue_match and max_sat >= 0.5
        if not (hue_match and (light_like or chroma_field or material_color)):
            continue
        neutral_base = any(
            term in text
            for term in (
                "cream",
                "ivory",
                "pale base",
                "white base",
                "soft base",
                "base surfaces",
                "silver",
                "chrome-silver",
                "gray",
                "grey",
            )
        )
        if neutral_base and max_sat < 0.55 and not any(term in text for term in ("gold", "yellow", "orange", "cyan", "teal")):
            continue
        if "material-light" in allocation_kind:
            rank = 0
        elif "medium light" in allocation_kind:
            rank = 1
        elif chroma_field:
            rank = 2
        elif material_color:
            rank = 3
        else:
            rank = 4
        if not hue_match:
            rank += 4
        ranked.append((rank, index, item))
    return [item for _rank, _index, item in sorted(ranked, key=lambda entry: (entry[0], entry[1]))[:max_items]]


def _bright_chroma_anchor_text(items: Sequence[Mapping[str, Any]], *, limit: int) -> str:
    labels = [
        _sanitize_prompt_fragment_text(item.get("label") or "")
        for item in items
        if _sanitize_prompt_fragment_text(item.get("label") or "")
    ]
    if labels:
        return _join_limited(labels, limit, separator=", ")
    return _allocation_color_group_phrase(items, max_items=3, limit=limit) if items else ""


def _allocation_search_text(item: Mapping[str, Any]) -> str:
    return " ".join(
        str(item.get(key) or "")
        for key in (
            "allocation",
            "label",
            "carrier",
            "carrier_scale",
            "carrier_geometry",
            "light_role",
            "generator_safe_wording",
        )
    ).lower()


def _allocation_max_hex_saturation(item: Mapping[str, Any]) -> float:
    values = [_hex_color_saturation(hex_value) for hex_value in _dedupe_hexes(item.get("hexes") or [])]
    return max(values) if values else 0.0


def _hex_color_saturation(hex_value: str) -> float:
    match = re.match(r"#?([0-9a-fA-F]{6})$", str(hex_value or "").strip())
    if not match:
        return 0.0
    value = match.group(1)
    channels = [int(value[index : index + 2], 16) / 255.0 for index in (0, 2, 4)]
    high = max(channels)
    low = min(channels)
    if high <= 0:
        return 0.0
    return (high - low) / high


def _allocation_item_hue_families(item: Mapping[str, Any]) -> List[str]:
    families: List[str] = []
    for hex_value in _dedupe_hexes(item.get("hexes") or []):
        family = _hex_color_hue_family(hex_value)
        if family:
            _append_unique(families, family)
    text = _allocation_search_text(item)
    for family in ("red", "orange-red", "orange-gold", "yellow", "green", "cyan", "blue", "violet", "magenta"):
        if _text_mentions_dominant_hue_family(text, {family}):
            _append_unique(families, family)
    return families


def _hex_color_hue_family(hex_value: str) -> str:
    match = re.match(r"#?([0-9a-fA-F]{6})$", str(hex_value or "").strip())
    if not match:
        return ""
    value = match.group(1)
    r, g, b = [int(value[index : index + 2], 16) / 255.0 for index in (0, 2, 4)]
    high = max(r, g, b)
    low = min(r, g, b)
    delta = high - low
    if high <= 0 or delta <= 0.02 or delta / high < 0.12:
        return ""
    if high == r:
        hue = (60 * ((g - b) / delta) + 360) % 360
    elif high == g:
        hue = 60 * ((b - r) / delta + 2)
    else:
        hue = 60 * ((r - g) / delta + 4)
    if hue < 12 or hue >= 345:
        return "red"
    if hue < 32:
        return "orange-red"
    if hue < 55:
        return "orange-gold"
    if hue < 75:
        return "yellow"
    if hue < 155:
        return "green"
    if hue < 205:
        return "cyan"
    if hue < 255:
        return "blue"
    if hue < 285:
        return "violet"
    if hue < 345:
        return "magenta"
    return ""


def _text_mentions_dominant_hue_family(text: str, families: Iterable[str]) -> bool:
    lowered = str(text or "").lower()
    for family in families:
        family = str(family or "").lower()
        if not family:
            continue
        if family in lowered:
            return True
        if family in {"orange-red", "orange-gold"} and re.search(r"\borange|amber|gold|golden\b", lowered):
            return True
        if family == "yellow" and re.search(r"\byellow|gold|golden|lemon\b", lowered):
            return True
        if family == "cyan" and re.search(r"\bcyan|teal|aqua|turquoise\b", lowered):
            return True
        if family == "blue" and re.search(r"\bblue|azure\b", lowered):
            return True
        if family == "red" and re.search(r"\bred|ruby|crimson|scarlet\b", lowered):
            return True
        if family == "green" and re.search(r"\bgreen|mint|lime\b", lowered):
            return True
        if family == "magenta" and re.search(r"\bmagenta|pink|fuchsia\b", lowered):
            return True
        if family == "violet" and re.search(r"\bviolet|purple\b", lowered):
            return True
    return False


def _hue_family_is_warm(family: str) -> bool:
    return str(family).lower() in {"red", "orange-red", "orange-gold", "yellow"}


def _hue_family_is_cool(family: str) -> bool:
    return str(family).lower() in {"cyan", "blue", "violet", "magenta"}


def _dominant_chromatic_field_required(distribution: Mapping[str, Any]) -> bool:
    data = _color_value_distribution_from_schema(distribution)
    if not data:
        return False
    chroma_key = str(data.get("chroma_key") or "").lower()
    if chroma_key == "near-monochrome":
        return False
    chromatic_light_role = str(data.get("chromatic_light_role") or "").lower()
    chromatic_role = str(data.get("chromatic_role") or "").lower()
    high_chroma_role = str(data.get("high_chroma_role") or "").lower()
    chromatic_light_fraction = float(data.get("chromatic_light_fraction") or 0.0)
    chromatic_light_cluster = float(data.get("chromatic_light_cluster_fraction") or 0.0)
    return (
        "broad colored" in chromatic_light_role
        or "selective colored" in chromatic_light_role
        or chromatic_light_fraction >= 0.06
        or chromatic_light_cluster >= 0.04
        or "large chromatic" in chromatic_role
        or "dominant saturated" in high_chroma_role
        or "visible saturated" in high_chroma_role
    )


def _compile_dominant_chromatic_field_prompt(distribution: Mapping[str, Any], *, max_chars: int = 230) -> str:
    data = _color_value_distribution_from_schema(distribution)
    if not _dominant_chromatic_field_required(data):
        return ""
    chromatic_light_role = str(data.get("chromatic_light_role") or "").lower()
    chromatic_role = str(data.get("chromatic_role") or "").lower()
    high_chroma_role = str(data.get("high_chroma_role") or "").lower()
    light_role = str(data.get("light_field_role") or "").lower()
    families = [str(item).lower() for item in data.get("dominant_hue_families") or [] if str(item).strip()]
    light_families = [
        str(item).lower()
        for item in data.get("dominant_light_hue_families") or []
        if str(item).strip()
    ]
    hue_label = _human_join(light_families or families) or "reference-dominant hue"

    clauses: List[str] = []
    if "broad colored" in chromatic_light_role:
        clauses.append(f"{hue_label} colored light fields occupy broad visible planes")
    elif "selective colored" in chromatic_light_role:
        clauses.append(f"{hue_label} colored light islands stay visibly chromatic")
    elif "large chromatic" in chromatic_role or "dominant saturated" in high_chroma_role:
        clauses.append(f"{hue_label} chromatic fields occupy broad scene/material planes")
    else:
        clauses.append(f"{hue_label} chromatic fields remain carrier-sized, not accents")

    clauses.append("high-value color stays chromatic before white, not neutral glare or pale wash")
    if "broad" in light_role or "white base" in light_role:
        clauses.append("broad light fields must carry hue weight before becoming white")
    if "saturated" in high_chroma_role or "large chromatic" in chromatic_role:
        clauses.append("dominant color cannot collapse into tiny accents, thin lines, or local object trim")

    body = _join_limited(_dedupe(clauses), max(90, int(max_chars) - len("Dominant chromatic field: ")), separator="; ")
    return _clean_clause("Dominant chromatic field: " + body)


def _compile_scene_carrier_distribution_prompt(distribution: Mapping[str, Any], *, max_chars: int = 220) -> str:
    data = _color_value_distribution_from_schema(distribution)
    if not data:
        return ""
    value_key = str(data.get("value_key") or "").lower()
    dark_role = str(data.get("dark_field_role") or "").lower()
    light_role = str(data.get("light_field_role") or "").lower()
    chroma_key = str(data.get("chroma_key") or "").lower()
    chromatic_role = str(data.get("chromatic_role") or "").lower()
    high_chroma_role = str(data.get("high_chroma_role") or "").lower()
    value_contrast_role = str(data.get("value_contrast_role") or "").lower()
    families = [str(item).lower() for item in data.get("dominant_hue_families") or [] if str(item).strip()]
    family_text = _human_join(families)

    clauses: List[str] = []
    needs_broad_chroma_carriers = (
        chroma_key == "high-chroma"
        or "large chromatic" in chromatic_role
        or "saturated" in high_chroma_role
    )
    if needs_broad_chroma_carriers:
        hue_label = family_text or "reference-dominant hue"
        clauses.append(
            f"{hue_label} distribution maps onto broad background/environment planes, material surfaces, and repeated patches"
        )
        clauses.append("large light/shadow fields may carry chroma weight before local accents")
    elif family_text and chroma_key != "near-monochrome":
        clauses.append(f"{family_text} stays visible across repeated compatible scene/material carriers")

    if chroma_key == "near-monochrome":
        if "low" in value_key or "dark" in dark_role:
            clauses.append("dark fields, midtone planes, and bright islands share the value load")
            clauses.append("background/environment shadow carriers stay readable, not all-black")
        else:
            clauses.append("grayscale planes carry the scene before any small chromatic carrier")
        clauses.append("map unspecified content colors to value carriers instead of local object hue")
        if family_text:
            clauses.append(f"{family_text} stays visible across repeated compatible scene/material carriers")

    if value_key == "low-key" or "dominant" in dark_role or "major" in dark_role:
        clauses.append("background/environment shadow carriers stay readable, not all-black")
    if value_key == "high-key" or "broad" in light_role or "white base" in light_role:
        clauses.append("broad light planes keep edges and chroma anchors, not white wash")
    if "wide" in value_contrast_role:
        clauses.append("spread value contrast across scene planes, material fields, and highlight islands")

    if not clauses:
        clauses.append("map reference value/chroma weight onto broad compatible scene and material carriers")
    body = _join_limited(_dedupe(clauses), max(90, int(max_chars) - len("Scene carrier map: ")), separator="; ")
    return _clean_clause("Scene carrier map: " + body)


def _compile_color_value_distribution_prompt(distribution: Mapping[str, Any], *, max_chars: int = 220) -> str:
    data = _color_value_distribution_from_schema(distribution)
    if not data:
        return ""
    value_key = str(data.get("value_key") or "").lower()
    dark_role = str(data.get("dark_field_role") or "").lower()
    light_role = str(data.get("light_field_role") or "").lower()
    chroma_key = str(data.get("chroma_key") or "").lower()
    chromatic_role = str(data.get("chromatic_role") or "").lower()
    high_chroma_role = str(data.get("high_chroma_role") or "").lower()
    contrast_key = str(data.get("contrast_key") or "").lower()
    value_contrast_role = str(data.get("value_contrast_role") or "").lower()
    families = [str(item).lower() for item in data.get("dominant_hue_families") or [] if str(item).strip()]

    clauses: List[str] = []
    if "dominant" in dark_role:
        clauses.append("preserve dominant dark/black fields")
    elif "major" in dark_role or "large" in dark_role:
        clauses.append("preserve major dark/black fields")
    elif "dark" in dark_role or value_key == "low-key":
        clauses.append("preserve dark/black structure")

    if "broad" in light_role or "white base" in light_role or "base field" in light_role:
        clauses.append("retain broad light/white fields without washing out dark edges or chroma area")
    elif "selective" in light_role or "island" in light_role:
        clauses.append("keep selective light islands readable with visible area")

    if chroma_key == "near-monochrome":
        if families:
            family_text = _human_join(families)
            clauses.append(
                f"near-monochrome key; only extracted {family_text} chromatic carriers keep hue; "
                "other content colors map to grayscale value"
            )
        else:
            clauses.append(
                "near-monochrome key; map unspecified content colors to grayscale value; "
                "no local object hue unless extracted carrier permits"
            )
        if "wide" in value_contrast_role or "dark-light" in contrast_key:
            clauses.append("bright highlight islands must cut through dark fields; avoid crushed all-black fields")
        if "dominant" in dark_role or "major" in dark_role or "wide" in value_contrast_role:
            clauses.append("preserve black-white-gray steps with visible mid-gray planes, not flat black void")
        if "selective" in light_role or "island" in light_role:
            clauses.append("white highlight islands need readable area, not pinpoints lost in black")
    elif "large chromatic" in chromatic_role:
        family_text = _human_join(families) if families else "reference hue"
        clauses.append(
            f"keep {family_text} chromatic material fields as repeated carrier-area coverage, "
            "not tiny accents/brown/gray/white wash"
        )
    elif chroma_key == "high-chroma":
        family_text = _human_join(families) if families else "high-chroma"
        clauses.append(f"keep {family_text} anchors saturated as repeated carrier-area coverage on their carriers")

    if (
        chroma_key == "high-chroma"
        or "saturated" in high_chroma_role
        or "large chromatic" in chromatic_role
    ):
        if "large chromatic" not in chromatic_role:
            clauses.append("keep extracted chromatic carriers saturated/visually weighty, not brown/gray/white wash")
    if "dominant saturated" in high_chroma_role:
        clauses.append("dominant saturated fields need broad repeated coverage on compatible patches/fields before shadows/highlights")
    elif "visible saturated" in high_chroma_role:
        clauses.append("visible saturated fields need repeated coverage on compatible patches/fields before shadows/highlights")
    if "dominant saturated" in high_chroma_role or "visible saturated" in high_chroma_role:
        clauses.append("selected/accent roles may repeat on patches/bands, not whole objects")

    if "dark-light" in contrast_key or ("dark" in dark_role and ("light" in light_role or "white" in light_role)):
        clauses.append("preserve dark-light separation")
    if "wide" in value_contrast_role and chroma_key != "near-monochrome":
        clauses.append("preserve wide value range with dark anchors and bright islands")
    elif "compressed" in value_contrast_role:
        clauses.append("preserve soft compressed value range without artificial hard contrast")

    if not clauses:
        return ""
    body = _join_limited(clauses, max(80, int(max_chars) - len("Color/value distribution: ")), separator="; ")
    return _clean_clause("Color/value distribution: " + body)


def _human_join(items: Sequence[str]) -> str:
    clean = [_clean_clause(item) for item in items if _clean_clause(item)]
    if not clean:
        return ""
    if len(clean) == 1:
        return clean[0]
    if len(clean) == 2:
        return f"{clean[0]} and {clean[1]}"
    return ", ".join(clean[:-1]) + f", and {clean[-1]}"


def _tonal_balance_from_schema(value: Any) -> Dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    out: Dict[str, Any] = {}
    for key in ("value_key", "shadow_extent", "highlight_extent", "midtone_extent", "chroma_key"):
        text = _clean_clause(value.get(key) or "")
        if text:
            out[key] = text
    for key in ("luma_mean", "dark_fraction", "light_fraction", "saturation_mean"):
        raw = value.get(key)
        if isinstance(raw, (int, float)):
            out[key] = float(raw)
    notes = value.get("notes")
    if isinstance(notes, Sequence) and not isinstance(notes, (str, bytes, bytearray)):
        clean_notes = [_clean_clause(item) for item in notes if _clean_clause(item)]
        if clean_notes:
            out["notes"] = clean_notes[:4]
    return out


def _compile_tonal_balance_prompt(tonal_balance: Mapping[str, Any], *, max_chars: int = 160) -> str:
    data = _tonal_balance_from_schema(tonal_balance)
    if not data:
        return ""
    value_key = str(data.get("value_key") or "").lower()
    shadow_extent = str(data.get("shadow_extent") or "").lower()
    highlight_extent = str(data.get("highlight_extent") or "").lower()
    midtone_extent = str(data.get("midtone_extent") or "").lower()
    chroma_key = str(data.get("chroma_key") or "").lower()
    clauses: List[str] = []
    if value_key == "low-key":
        clauses.append("low-key value balance")
        if "large" in shadow_extent or "dominant" in shadow_extent:
            clauses.append("large dark fields must leave readable midtone and highlight islands")
        else:
            clauses.append("dark shapes should not erase carrier contrast")
    elif value_key == "high-key":
        clauses.append("high-key value balance")
        clauses.append("light fields must keep structural contrast and local material anchors")
    elif value_key == "mid-key":
        clauses.append("mid-key value balance")
        clauses.append("preserve readable midtones between dark and light anchors")
    if "sparse" in highlight_extent or "small" in highlight_extent or "selective" in highlight_extent:
        clauses.append("highlights stay selective")
    elif "broad" in highlight_extent or "large" in highlight_extent:
        clauses.append("broad light fields retain texture and edges")
    if "broad" in midtone_extent or "large" in midtone_extent:
        clauses.append("broad midtone carriers stay readable")
    if chroma_key == "low-chroma":
        clauses.append("keep chroma restrained")
    elif chroma_key == "high-chroma":
        clauses.append("keep saturated anchors visible on their carriers")
    if not clauses:
        return ""
    return _clean_clause("Tonal balance: " + _join_limited(clauses, max(80, max_chars - len("Tonal balance: ")), separator=", "))


def _build_color_light_allocation(
    inner: Mapping[str, Any],
    color_contract: Optional[Mapping[str, Any]],
    *,
    auxiliary_reference_text: str = "",
) -> List[Dict[str, Any]]:
    extracted = _agent_extracted_color_light_allocation(inner.get("color_light_allocation"))
    if extracted:
        return extracted[:8]
    color_text = _first_non_empty(inner, "color_carrier_binding")
    items: List[Dict[str, Any]] = []
    if color_contract:
        roles = color_contract.get("color_roles")
        if isinstance(roles, list):
            fixed_hexes = _dedupe_hexes(_HEX_COLOR_RE.findall(auxiliary_reference_text or color_text))
            for role in roles:
                if not isinstance(role, Mapping):
                    continue
                color_family = str(role.get("color_family") or "").strip()
                role_hexes = _dedupe_hexes(_HEX_COLOR_RE.findall(color_family))
                if not role_hexes:
                    continue
                fixed_hex_set = {hex_value.lower() for hex_value in fixed_hexes}
                preferred = [hex_value for hex_value in role_hexes if hex_value.lower() in fixed_hex_set]
                hexes = _dedupe_hexes(preferred or role_hexes)[:2]
                if not hexes:
                    continue
                item = _color_light_allocation_item(
                    label=_compact_color_family_label(color_family),
                    hexes=hexes,
                    context=" ".join(
                        str(role.get(key) or "")
                        for key in ("color_family", "visual_rank", "dominance_type")
                    ),
                    source="color_contract",
                )
                if item:
                    items.append(item)
    items.extend(_color_light_allocation_from_text(color_text or auxiliary_reference_text, source="color_text"))
    return _dedupe_color_light_allocation(items)[:6]


def _agent_extracted_color_light_allocation(value: Any) -> List[Dict[str, Any]]:
    if not isinstance(value, list):
        return []
    out: List[Dict[str, Any]] = []
    for item in value:
        if not isinstance(item, Mapping):
            continue
        hexes = _coerce_hex_list(item.get("hexes"))
        if not hexes:
            continue
        allocation = _clean_clause(item.get("allocation") or "")
        out.append(
            {
                "allocation": allocation,
                "label": _clean_clause(item.get("label") or "reference color"),
                "hexes": hexes[:2],
                "carrier": _clean_clause(item.get("carrier") or "local material carriers"),
                "carrier_scale": _clean_clause(item.get("carrier_scale") or ""),
                "carrier_geometry": _clean_clause(item.get("carrier_geometry") or ""),
                "light_role": _clean_clause(item.get("light_role") or "material color"),
                "material_vs_glow_boundary": _clean_clause(item.get("material_vs_glow_boundary") or ""),
                "evidence_across_references": _clean_clause(item.get("evidence_across_references") or ""),
                "generator_safe_wording": _clean_clause(item.get("generator_safe_wording") or ""),
                "source": "agent_extracted",
            }
        )
    return _dedupe_color_light_allocation(out)


def _color_light_allocation_from_text(text: str, *, source: str = "color_text") -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for clause in _split_clauses(text or ""):
        hexes = _dedupe_hexes(_HEX_COLOR_RE.findall(clause))
        if not hexes:
            continue
        clean = _clean_clause(clause)
        item = _color_light_allocation_item(
            label=_allocation_label_from_clause(clean),
            hexes=hexes[:2],
            context=clean,
            source=source,
        )
        if item:
            out.append(item)
    return _dedupe_color_light_allocation(out)


def _allocation_label_from_clause(clause: str) -> str:
    no_hex = _HEX_COLOR_RE.sub(" ", str(clause or ""))
    no_hex = re.sub(r"\([^)]*\)", " ", no_hex)
    no_hex = _clean_clause(no_hex)
    lowered = no_hex.lower()
    for marker in (" carry ", " carries ", " belongs ", " attach", " forms ", " fill ", " fills ", " for "):
        pos = lowered.find(marker)
        if pos > 0:
            no_hex = no_hex[:pos]
            break
    return _clean_clause(no_hex) or "reference color"


def _color_light_allocation_item(
    *,
    label: str,
    hexes: Sequence[str],
    context: str,
    source: str,
) -> Dict[str, Any]:
    clean_hexes = _dedupe_hexes(hexes)
    if not clean_hexes:
        return {}
    lowered = f"{label} {context}".lower()
    if any(term in lowered for term in ("structural_shadow", "structural_only", "subject_readability_mass")):
        allocation = "structural dark"
        carrier = "silhouette clarity, recesses, contact shadows, and occluded folds"
        light_role = "shadow structure"
    elif any(term in lowered for term in ("tiny", "small glint", "glint", "spark", "speck", "specular_highlight")):
        allocation = "tiny accent"
        carrier = "small glints, fine accent marks, and tiny specular points"
        light_role = "accent sparkle only"
    elif any(term in lowered for term in ("cream", "ivory", "pale", "highlight", "glow", "light ray", "light rays", "negative_space", "negative space", "bloom", "rim light", "negative_space_or_paper_base")):
        allocation = "medium light"
        carrier = "controlled backglow, light rays, reflected highlights, and negative-space light"
        light_role = "secondary light only"
    elif any(term in lowered for term in ("gold", "copper", "amber", "orange", "metallic", "glossy", "material_surface_mass", "reflective", "radiating")):
        allocation = "large material-light"
        carrier = "broad reflective/radiating material bands, reflective planes, and metallic/glossy surfaces"
        light_role = "primary material light"
    elif any(term in lowered for term in ("red", "crimson", "vermilion", "lacquer", "accent", "graphic_accent")):
        allocation = "large material color"
        carrier = "compatible subject-worn material surfaces, object material surfaces, broad scene color fields, and radiating material fields"
        light_role = "primary material color"
    elif any(term in lowered for term in ("shadow", "shadows", "recess", "recesses", "silhouette", "silhouettes", "deep", "dark", "structural_shadow", "outer structure")):
        allocation = "structural dark"
        carrier = "silhouette clarity, recesses, contact shadows, and occluded folds"
        light_role = "shadow structure"
    elif "dominant" in lowered or "base" in lowered or "atmosphere" in lowered:
        allocation = "dominant field"
        carrier = "broad environment/material fields with local subject colors preserved"
        light_role = "dominant carrier color"
    else:
        allocation = "local material color"
        carrier = "local material carriers, accents, and readable subject surfaces"
        light_role = "material color"
    return {
        "allocation": allocation,
        "label": _clean_clause(label) or "reference color",
        "hexes": clean_hexes,
        "carrier": carrier,
        "carrier_scale": "",
        "carrier_geometry": "",
        "light_role": light_role,
        "material_vs_glow_boundary": "",
        "source": source,
    }


def _coerce_hex_list(value: Any) -> List[str]:
    if isinstance(value, str):
        return _dedupe_hexes(_HEX_COLOR_RE.findall(value) or [item.strip() for item in value.split("/")])
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return _dedupe_hexes(str(item) for item in value)
    return []


def _dedupe_color_light_allocation(items: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    used_hexes = set()
    ranked = sorted(
        (item for item in items if item.get("hexes")),
        key=lambda item: (_allocation_priority(str(item.get("allocation") or "")), -len(item.get("hexes") or [])),
    )
    for item in ranked:
        hexes = _dedupe_hexes(item.get("hexes") or [])
        if not hexes:
            continue
        hex_key = {hex_value.lower() for hex_value in hexes}
        if hex_key & used_hexes:
            continue
        used_hexes.update(hex_key)
        out.append(
            {
                "allocation": _clean_clause(item.get("allocation") or "local"),
                "label": _clean_clause(item.get("label") or "reference color"),
                "hexes": hexes,
                "carrier": _clean_clause(item.get("carrier") or "local material carriers"),
                "carrier_scale": _clean_clause(item.get("carrier_scale") or ""),
                "carrier_geometry": _clean_clause(item.get("carrier_geometry") or ""),
                "light_role": _clean_clause(item.get("light_role") or "material color"),
                "source": _clean_clause(item.get("source") or "derived"),
                "material_vs_glow_boundary": _clean_clause(item.get("material_vs_glow_boundary") or ""),
                "evidence_across_references": _clean_clause(item.get("evidence_across_references") or ""),
                "generator_safe_wording": _clean_clause(item.get("generator_safe_wording") or ""),
            }
        )
    return out


def _allocation_priority(allocation: str) -> int:
    lowered = allocation.lower()
    if "large material color" in lowered:
        return 0
    if "large material-light" in lowered:
        return 1
    if "dominant" in lowered:
        return 2
    if "medium light" in lowered:
        return 3
    if "structural dark" in lowered:
        return 4
    if "tiny" in lowered or "small" in lowered:
        return 5
    return 6


def _compile_color_light_allocation_prompt(
    allocation: Sequence[Mapping[str, Any]],
    *,
    content_carrier_arbitration: Sequence[Mapping[str, str]] = (),
    content_carrier_arbitration_prompt: str = "",
    color_value_distribution_prompt: str = "",
    high_risk: bool,
    max_chars: int = 360,
) -> str:
    clean_items = _dedupe_color_light_allocation(allocation)
    if not clean_items:
        if color_value_distribution_prompt:
            return _clean_clause(
                "Color carriers: "
                + _join_limited([color_value_distribution_prompt], max(80, int(max_chars) - len("Color carriers: ")))
            )
        return ""
    material = _rank_material_anchor_items(
        _allocation_items_by_kind(clean_items, ("large material color", "large material-light"))
    )
    local_material = _allocation_items_by_kind(clean_items, ("local material color",))
    base = [
        item
        for item in _allocation_items_by_kind(clean_items, ("dominant field",))
        if not _allocation_item_is_shadow_like(item)
    ]
    light = _allocation_items_by_kind(clean_items, ("medium light",))
    dark_base = [
        item
        for item in _allocation_items_by_kind(clean_items, ("dominant field",))
        if _allocation_item_is_shadow_like(item)
    ]
    local_dark = _allocation_items_by_kind(clean_items, ("structural dark",))
    tiny = _allocation_items_by_kind(clean_items, ("tiny accent",))
    clauses: List[str] = []
    body_limit = max(80, int(max_chars) - len("Color carriers: "))
    shadow_material = [item for item in material if _allocation_item_is_shadow_like(item)]
    material_light = [item for item in material if "material-light" in str(item.get("allocation") or "").lower()]
    prioritize_light = bool((light or material_light) and (dark_base or shadow_material))
    defer_shadow_material = bool(prioritize_light and shadow_material)
    primary_material = [item for item in material if item not in shadow_material] if defer_shadow_material else material
    light_anchor_clause = ""
    shadow_material_clause = ""
    role_discipline_clause = (
        (
            "role discipline: colored material/light anchors stay carrier-bound before shadow material/base; "
            "no generic object color or whole-image tint"
        )
        if prioritize_light and shadow_material
        else (
            "role discipline: keep extracted scales; bind roles only to extracted carriers; "
            "material/light/local before dark/base; no generic object color or whole-image tint"
        )
        if prioritize_light
        else _COLOR_ROLE_DISCIPLINE_CLAUSE
    )
    color_prompt_arbitration = (
        _compact_color_prompt_arbitration_clause(content_carrier_arbitration_prompt)
        if prioritize_light
        else content_carrier_arbitration_prompt
    )
    required_clauses = [
        clause
        for clause in (
            role_discipline_clause,
            color_prompt_arbitration,
            color_value_distribution_prompt,
        )
        if clause
    ]
    required_len = len("; ".join(_clean_clause(clause) for clause in required_clauses))
    optional_anchor_budget = max(72, body_limit - required_len - 18)

    _append_unique(clauses, role_discipline_clause)
    primary_material_clause = ""
    if primary_material:
        primary_anchor_count = 4 if len(primary_material) >= 4 and optional_anchor_budget >= 220 else 3
        material_hex_tail = f" ({_allocation_hexes(primary_material, max_hexes=primary_anchor_count * 2)})"
        material_prefix_len = len("primary material/color anchors: ") + len(material_hex_tail)
        material_limit = max(96, min(260 if primary_anchor_count >= 4 else 180, optional_anchor_budget - material_prefix_len - 4))
        if prioritize_light and shadow_material:
            primary_material_clause = (
                "primary material/color anchors: "
                + _allocation_color_label_hex_group(
                    primary_material,
                    max_items=primary_anchor_count,
                    limit=360 if primary_anchor_count >= 4 else 260,
                )
            )
        else:
            primary_material_clause = (
                f"primary material/color anchors: {_allocation_color_group_phrase(primary_material, max_items=primary_anchor_count, limit=material_limit)} "
                f"{material_hex_tail.strip()}"
            )
        _append_unique(clauses, primary_material_clause)
    if prioritize_light and light:
        light_phrase = _allocation_color_group_phrase(light, max_items=1, limit=56)
        light_anchor_clause = (
            f"secondary light/highlight anchors: {light_phrase} "
            f"({_allocation_hexes(light, max_hexes=2)})"
        )
        _append_unique(clauses, light_anchor_clause)
    if defer_shadow_material:
        shadow_anchor_count = 2 if len(shadow_material) >= 2 else 1
        shadow_phrase = _allocation_color_group_phrase(
            shadow_material,
            max_items=shadow_anchor_count,
            limit=128 if shadow_anchor_count >= 2 else 58,
        )
        shadow_material_clause = (
            f"shadow material anchors: {shadow_phrase} ({_allocation_hexes(shadow_material, max_hexes=shadow_anchor_count * 2)}), "
            "support chromatic carriers without whole-scene fill"
        )
        _append_unique(clauses, shadow_material_clause)
    if prioritize_light and local_material:
        _append_unique(
            clauses,
            f"local material anchors: {_allocation_color_group_phrase(local_material, max_items=1, limit=68)} "
            f"({_allocation_hexes(local_material, max_hexes=2)})",
        )
    if prioritize_light and tiny:
        _append_unique(
            clauses,
            f"tiny accent anchors: {_allocation_color_group_phrase(tiny, max_items=1, limit=52)} "
            f"({_allocation_hexes(tiny, max_hexes=2)})",
        )
    if color_prompt_arbitration:
        _append_unique(clauses, color_prompt_arbitration)
    if color_value_distribution_prompt:
        _append_unique(clauses, color_value_distribution_prompt)
    if base:
        _append_unique(
            clauses,
            f"base fields: {_allocation_color_group_phrase(base, max_items=2, limit=96)} "
            f"({_allocation_hexes(base, max_hexes=4)})",
        )
    if not prioritize_light and local_material:
        _append_unique(
            clauses,
            f"local material anchors: {_allocation_color_group_phrase(local_material, max_items=1, limit=68)} "
            f"({_allocation_hexes(local_material, max_hexes=2)})",
        )
    if not prioritize_light and light:
        _append_unique(
            clauses,
            f"secondary light/highlight anchors: {_allocation_color_group_phrase(light, max_items=1, limit=72)} "
            f"({_allocation_hexes(light, max_hexes=2)})",
        )
    if not prioritize_light and tiny:
        _append_unique(
            clauses,
            f"tiny accent anchors: {_allocation_color_group_phrase(tiny, max_items=1, limit=52)} "
            f"({_allocation_hexes(tiny, max_hexes=2)})",
        )
    if dark_base:
        _append_unique(
            clauses,
            f"dark/base fields: {_allocation_color_group_phrase(dark_base, max_items=1, limit=72)} "
            f"({_allocation_hexes(dark_base, max_hexes=2)}), preserve anchors",
        )
    if local_dark:
        _append_unique(
            clauses,
            f"local dark structure: {_allocation_color_group_phrase(local_dark, max_items=1, limit=72)} "
            f"({_allocation_hexes(local_dark, max_hexes=2)}), preserve anchors",
        )
    if not clauses:
        return ""
    body = _join_limited_with_required_tail(
        clauses,
        body_limit,
        required_tail=[
            clause
            for clause in (
                primary_material_clause,
                light_anchor_clause,
                shadow_material_clause,
                color_prompt_arbitration,
                color_value_distribution_prompt,
            )
            if clause
        ],
        separator="; ",
    )
    return _clean_clause("Color carriers: " + body)


def _compact_color_prompt_arbitration_clause(prompt: str) -> str:
    if not _clean_clause(prompt):
        return ""
    return (
        "carrier arbitration: selected/accent/local roles stay on small surfaces, bands, patches; "
        "exclude whole subject body, whole clothing mass, whole tabletop, whole background; "
        "do not recolor entire semantic objects"
    )


def _compile_content_carrier_arbitration_prompt(
    arbitration: Sequence[Mapping[str, str]],
    *,
    max_chars: int = 220,
) -> str:
    records = _dedupe_content_carrier_arbitration(arbitration)
    protected = [
        item
        for item in records
        if _arbitration_record_needs_prompt_control(item)
    ]
    if not protected:
        return ""
    disallowed: List[str] = []
    for item in protected:
        for part in str(item.get("disallowed_content_carriers") or "").split(","):
            clean = _clean_clause(part)
            if clean:
                _append_unique(disallowed, clean)
    allowed_text = "small surfaces, bands, inserts, facets, patches"
    required_disallowed = ["whole subject body", "whole clothing mass", "whole tabletop", "whole background"]
    for carrier in required_disallowed:
        if any(carrier == item.lower() for item in disallowed):
            continue
        _append_unique(disallowed, carrier)
    disallowed_text = _join_limited(disallowed, 92, separator=", ")
    body = (
        f"selected/accent/local roles allowed only on {allowed_text}; "
        f"exclude {disallowed_text}; do not recolor entire semantic objects"
    )
    return _clean_clause("carrier arbitration: " + _truncate_words(body, max(80, int(max_chars) - len("carrier arbitration: "))))


def _arbitration_record_needs_prompt_control(item: Mapping[str, str]) -> bool:
    allocation = str(item.get("allocation") or "").lower()
    if "dominant field" in allocation:
        return False
    if "medium light" in allocation or "structural dark" in allocation:
        return False
    role_scope = str(item.get("role_scope") or "").lower()
    expansion = str(item.get("expansion_control") or "").lower()
    disallowed = str(item.get("disallowed_content_carriers") or "").lower()
    return (
        "selected carrier" in role_scope
        or "local only" in role_scope
        or "tiny/local" in role_scope
        or "tiny accent" in allocation
        or "local material color" in allocation
        or "selected/accent/local" in expansion
        or "whole subject body" in disallowed
    )


def _allocation_items_by_kind(
    items: Sequence[Mapping[str, Any]],
    kinds: Sequence[str],
) -> List[Mapping[str, Any]]:
    wanted = tuple(kind.lower() for kind in kinds)
    return [item for item in items if str(item.get("allocation") or "").lower() in wanted]


def _rank_material_anchor_items(items: Sequence[Mapping[str, Any]]) -> List[Mapping[str, Any]]:
    def rank(item: Mapping[str, Any]) -> int:
        allocation = str(item.get("allocation") or "").lower()
        shadow_like = _allocation_item_is_shadow_like(item)
        if "material-light" in allocation and not shadow_like:
            return 0
        if not shadow_like:
            return 1
        if "material-light" in allocation:
            return 2
        return 3

    return [item for _index, item in sorted(enumerate(items), key=lambda pair: (rank(pair[1]), pair[0]))]


def _allocation_group_phrase(items: Sequence[Mapping[str, Any]], *, max_items: int) -> str:
    phrases: List[str] = []
    for item in items[:max_items]:
        phrase = _allocation_item_phrase(item)
        if phrase:
            _append_unique(phrases, phrase)
    return _join_limited(phrases, 150, separator="; ") or "reference-derived carrier roles"


def _allocation_color_group_phrase(items: Sequence[Mapping[str, Any]], *, max_items: int, limit: int) -> str:
    selected_items = list(items[:max_items])
    phrases: List[str] = []
    for item in selected_items:
        phrase = _allocation_color_item_phrase(item, include_scope=True)
        if phrase:
            _append_unique(phrases, phrase)
    joined = ", ".join(phrases)
    if len(joined) <= limit and _color_group_mentions_every_label(joined, selected_items):
        return joined

    compact_phrases: List[str] = []
    for item in selected_items:
        phrase = _allocation_color_compact_item_phrase(item, include_scope=True)
        if phrase:
            _append_unique(compact_phrases, phrase)
    compact_joined = ", ".join(compact_phrases)
    if len(compact_joined) <= limit and _color_group_mentions_every_label(compact_joined, selected_items):
        return compact_joined
    label_phrases = [
        _sanitize_prompt_fragment_text(item.get("label") or "")
        for item in selected_items
        if _sanitize_prompt_fragment_text(item.get("label") or "")
    ]
    label_joined = ", ".join(label_phrases)
    label_fallback = label_joined
    if (
        label_fallback
        and len(label_fallback) <= limit
        and _color_group_mentions_every_label(label_fallback, selected_items)
    ):
        return label_fallback
    return _join_limited(compact_phrases or phrases, limit, separator=", ") or "reference-derived carrier roles"


def _allocation_color_label_hex_group(items: Sequence[Mapping[str, Any]], *, max_items: int, limit: int) -> str:
    phrases: List[str] = []
    for item in list(items[:max_items]):
        label = _sanitize_prompt_fragment_text(item.get("label") or "")
        if not label:
            label = _truncate_words(_allocation_color_compact_item_phrase(item, include_scope=False), 36)
        carrier = _compact_color_allocation_phrase(str(item.get("carrier") or ""))
        carrier_words = _truncate_words(carrier, 32) if carrier else ""
        hexes = "/".join(_dedupe_hexes(item.get("hexes") or [])[:2])
        phrase = label
        if carrier_words:
            phrase = f"{phrase}: {carrier_words}"
        if hexes:
            phrase = f"{phrase} ({hexes})"
        _append_unique(phrases, _clean_clause(phrase))
    return _join_limited(phrases, limit, separator=", ") or "reference-derived carrier roles"


def _color_group_mentions_every_label(text: str, items: Sequence[Mapping[str, Any]]) -> bool:
    lowered = text.lower()
    for item in items:
        label = _sanitize_prompt_fragment_text(item.get("label") or "")
        if label and label.lower() not in lowered:
            return False
    return True


def _allocation_color_item_phrase(item: Mapping[str, Any], *, include_scope: bool = False) -> str:
    label = _sanitize_prompt_fragment_text(item.get("label") or "")
    allocation = str(item.get("allocation") or "").lower()
    generator_wording = _compact_color_phrase_for_label(str(item.get("generator_safe_wording") or ""))
    if "large material color" in allocation and generator_wording and not generator_wording.lower().startswith("when "):
        phrase = _color_label_generator_phrase(label, generator_wording, limit=92)
        return _append_color_scope_marker(phrase, item) if include_scope else phrase
    if "material-light" in allocation and generator_wording and not generator_wording.lower().startswith("when "):
        phrase = _color_label_generator_phrase(label, generator_wording, limit=116)
        return _append_color_scope_marker(phrase, item) if include_scope else phrase
    if "local material color" in allocation and generator_wording and not generator_wording.lower().startswith("when "):
        phrase = _color_label_generator_phrase(label, generator_wording, limit=76)
        return _append_color_scope_marker(phrase, item) if include_scope else phrase
    if label:
        phrase = _truncate_words(label, 52)
        return _append_color_scope_marker(phrase, item) if include_scope else phrase
    if generator_wording:
        phrase = _truncate_words(generator_wording, 52)
        return _append_color_scope_marker(phrase, item) if include_scope else phrase
    carrier = _compact_color_allocation_phrase(str(item.get("carrier") or ""))
    phrase = _truncate_words(carrier, 52)
    return _append_color_scope_marker(phrase, item) if include_scope else phrase


def _allocation_color_compact_item_phrase(item: Mapping[str, Any], *, include_scope: bool = False) -> str:
    label = _sanitize_prompt_fragment_text(item.get("label") or "reference color")
    terms = _expected_color_carrier_terms(item)
    if terms:
        phrase = f"{label}: {', '.join(terms[:3])}"
    else:
        allocation = str(item.get("allocation") or "").lower()
        if "material-light" in allocation:
            phrase = f"{label}: material-light carriers"
        elif "large material color" in allocation:
            phrase = f"{label}: material carriers"
        else:
            phrase = label
    phrase = _truncate_words(phrase, 88)
    return _append_color_scope_marker(phrase, item) if include_scope else phrase


def _compact_color_phrase_for_label(text: str) -> str:
    clean = _compact_color_allocation_phrase(text)
    clean = re.sub(r",?\s*\bnot\b.*$", "", clean, flags=re.IGNORECASE)
    clean = _clean_clause(clean)
    return clean


def _color_label_generator_phrase(label: str, generator_wording: str, *, limit: int) -> str:
    clean_label = _sanitize_prompt_fragment_text(label)
    clean_wording = _sanitize_prompt_fragment_text(generator_wording)
    if clean_label and clean_label.lower() not in clean_wording.lower():
        return _truncate_words(f"{clean_label}: {clean_wording}", limit)
    return _truncate_words(clean_wording or clean_label, limit)


def _append_color_scope_marker(phrase: str, item: Mapping[str, Any]) -> str:
    clean = _clean_clause(phrase)
    if not clean:
        return ""
    scope = _color_scope_marker(item)
    if scope and scope.lower() not in clean.lower():
        clean = f"{clean} [{scope}]"
    boundary = _color_boundary_marker(item)
    if boundary and boundary.lower() not in clean.lower():
        clean = f"{clean}, {boundary}"
    return _truncate_words(clean, 128)


def _color_scope_marker(item: Mapping[str, Any]) -> str:
    allocation = str(item.get("allocation") or "").lower()
    scale = str(item.get("carrier_scale") or "").lower()
    geometry = str(item.get("carrier_geometry") or "").lower()
    carrier = str(item.get("carrier") or "").lower()
    light_role = str(item.get("light_role") or "").lower()
    scoped_text = " ".join((allocation, scale, geometry, light_role))
    full_text = " ".join((allocation, scale, geometry, carrier, light_role))
    if "dominant field" in allocation and "broad" in full_text:
        return "broad field"
    if _allocation_item_has_selected_carrier_scope(item):
        return "selected carrier only"
    if "broad thick continuous" in full_text or ("broad" in full_text and "continuous" in full_text):
        return "broad field"
    true_tiny_role = "tiny accent" in allocation or "accent sparkle" in light_role
    if true_tiny_role and any(term in scoped_text for term in ("tiny", "pinpoint", "fine thin", "hairline")):
        return "tiny/local only"
    if "broad" in full_text or "large" in full_text:
        return "medium/broad carrier"
    if "medium" in full_text and "local" in full_text:
        return "medium/local carrier"
    if any(term in scoped_text for term in ("small local", "medium local", "local", "dots", "short strokes")):
        return "local only"
    if any(term in scoped_text for term in ("tiny", "pinpoint", "fine thin", "hairline")):
        return "tiny/local only"
    if "medium" in full_text:
        return "medium carrier"
    return ""


def _carrier_scale_is_broad_continuous(scale: str) -> bool:
    """True when the extracted carrier scale denotes genuine broad continuous area.

    This is the strongest reference-derived signal that a role occupies a large
    field rather than scattered small carriers. It is generic: it reads only the
    extracted carrier_scale text, with no style id, hue, or failure-note input.
    """
    s = str(scale or "").lower()
    if not s:
        return False
    if "broad thick continuous" in s:
        return True
    if "broad" in s and "continuous" in s:
        return True
    return False


def _allocation_item_is_broad_light_role(item: Mapping[str, Any]) -> bool:
    """True for a broad-continuous illumination (material-light) role.

    Generic, reference-only: reads the extracted allocation kind, light role, and
    carrier scale. No style id, hue, or generated-failure input. Tangible
    material-color roles are excluded so the anti-recolor guardrail still holds.
    """
    if not _carrier_scale_is_broad_continuous(item.get("carrier_scale")):
        return False
    allocation = str(item.get("allocation") or "").lower()
    light_role = str(item.get("light_role") or "").lower()
    if "material color" in light_role:
        return False
    return "material-light" in allocation or "light" in light_role


def _allocation_item_has_selected_carrier_scope(item: Mapping[str, Any]) -> bool:
    allocation = str(item.get("allocation") or "").lower()
    if not _allocation_is_primary_material_role(allocation):
        return False
    # Broad illumination stays broad even when its description names small shapes.
    if _allocation_item_is_broad_light_role(item):
        return False
    text = " ".join(
        str(item.get(key) or "")
        for key in (
            "label",
            "carrier",
            "carrier_scale",
            "carrier_geometry",
            "light_role",
            "material_vs_glow_boundary",
            "generator_safe_wording",
        )
    ).lower()
    selected_terms = (
        "selected",
        "accent",
        "accents",
        "patch",
        "patches",
        "insert",
        "inserts",
        "stripe",
        "stripes",
        "side facet",
        "side facets",
        "side face",
        "side faces",
        "bounded",
        "slab",
        "slabs",
        "panel",
        "panels",
        "segment",
        "segments",
        "partition",
        "partitions",
        "piece",
        "pieces",
        "tile",
        "tiles",
        "fragment",
        "fragments",
        "cut shape",
        "cut shapes",
        "clustered",
        "local surface",
        "local surfaces",
        "where the content supports",
    )
    return any(term in text for term in selected_terms)


def _color_boundary_marker(item: Mapping[str, Any]) -> str:
    text = " ".join(
        str(item.get(key) or "")
        for key in ("allocation", "material_vs_glow_boundary", "light_role", "carrier")
    ).lower()
    if any(term in text for term in ("not as luminous", "not as neon", "not a glowing", "not as glow", "not as bloom", "not as luminous emission", "not as a pure neon aura")):
        return "not glow"
    if any(term in text for term in ("not as large black fills", "never as large black fills", "sparse pigment marks", "fine linework")):
        return "not broad fill"
    if any(term in text for term in ("not the main material carrier", "secondary", "medium light", "reflected highlights")):
        return "secondary light only"
    return ""


def _allocation_item_phrase(item: Mapping[str, Any]) -> str:
    wording = _compact_color_allocation_phrase(str(item.get("generator_safe_wording") or ""))
    if wording:
        return wording
    label = _sanitize_prompt_fragment_text(item.get("label") or "reference color")
    carrier = _compact_color_allocation_phrase(str(item.get("carrier") or ""))
    if carrier and label:
        return _truncate_words(f"{label} on {carrier}", 92)
    return _truncate_words(label or carrier, 92)


def _compact_color_allocation_phrase(text: str) -> str:
    clean = _sanitize_prompt_fragment_text(text)
    if not clean:
        return ""
    clean = re.sub(
        r"\bcompatible\s+subject-worn\s+and\s+object\s+material\s+surfaces\b",
        "subject/object surfaces",
        clean,
        flags=re.IGNORECASE,
    )
    clean = re.sub(
        r"\bsubject-worn\s+and\s+object\s+material\s+surfaces\b",
        "subject/object surfaces",
        clean,
        flags=re.IGNORECASE,
    )
    clean = re.sub(
        r"\bsubject-worn\s+material\s+surfaces\s+and\s+object\s+material\s+surfaces\b",
        "subject/object surfaces",
        clean,
        flags=re.IGNORECASE,
    )
    clean = re.sub(r"\bsubject-worn\s+", "", clean, flags=re.IGNORECASE)
    clean = re.sub(r"\bobject\s+material\s+surfaces\b", "object surfaces", clean, flags=re.IGNORECASE)
    clean = re.sub(r"\bcompatible\s+", "", clean, flags=re.IGNORECASE)
    clean = re.sub(r"\brather\s+than\b.*$", "", clean, flags=re.IGNORECASE)
    clean = re.sub(r"\bdo\s+not\b.*$", "", clean, flags=re.IGNORECASE)
    clean = re.sub(r"\bavoid\b.*$", "", clean, flags=re.IGNORECASE)
    clean = _sanitize_prompt_fragment_text(clean)
    return _truncate_words(clean, 96)


def _allocation_item_is_shadow_like(item: Mapping[str, Any]) -> bool:
    text = " ".join(str(item.get(key) or "") for key in ("light_role", "carrier", "label", "generator_safe_wording")).lower()
    return any(term in text for term in ("shadow", "recess", "silhouette", "occluded", "contour", "dark", "outer structure"))


def _material_structure_phrase(items: Sequence[Mapping[str, Any]]) -> str:
    text = " ".join(
        str(item.get(key) or "")
        for item in items
        for key in ("carrier", "carrier_scale", "carrier_geometry", "generator_safe_wording")
    ).lower()
    scale_parts: List[str] = []
    geometry_parts: List[str] = []
    if any(term in text for term in ("broad", "large", "wide", "thick", "substantial")):
        scale_parts.append("broad thick")
    elif any(term in text for term in ("medium", "moderate")):
        scale_parts.append("medium")
    if any(term in text for term in ("continuous", "unbroken", "long", "sweeping")):
        scale_parts.append("continuous")
    if any(term in text for term in ("layered", "stacked", "overlap", "overlapping")):
        scale_parts.append("layered")
    if re.search(r"\b(?:radiating|radial|ray|rays|sunburst)\b", text):
        geometry_parts.append("radiating material bands")
    elif any(term in text for term in ("band", "bands", "stripe", "stripes")):
        geometry_parts.append("material bands")
    if any(term in text for term in ("field", "fields", "plane", "planes", "surface", "surfaces")):
        geometry_parts.append("material fields")
    if not geometry_parts:
        geometry_parts.append("material carriers")
    phrase = " ".join(scale_parts + [_join_limited(_dedupe(geometry_parts), 120, separator=" and ")])
    return _clean_clause(phrase) or "radiating material bands"


def _material_glow_boundary_phrase(items: Sequence[Mapping[str, Any]]) -> str:
    boundary_items = [
        _clean_clause(item.get("material_vs_glow_boundary") or "")
        for item in items
        if _clean_clause(item.get("material_vs_glow_boundary") or "")
    ]
    if boundary_items:
        return _join_limited(boundary_items[:2], 240, separator="; ")
    text = " ".join(
        str(item.get(key) or "")
        for item in items
        for key in ("carrier", "carrier_scale", "carrier_geometry", "light_role")
    ).lower()
    if any(term in text for term in ("glow", "backglow", "light-only", "secondary light", "ray", "rays")):
        return "Keep tangible material-light carriers distinct from soft light-only glow and reflected highlights"
    return ""


def _allocation_hexes(items: Sequence[Mapping[str, Any]], *, max_hexes: int) -> str:
    grouped = []
    seen = set()
    for item in items:
        hexes = _dedupe_hexes(item.get("hexes") or [])
        if not hexes:
            continue
        group = "/".join(hexes[:2])
        lowered = group.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        grouped.append(group)
    if not grouped:
        return ""
    text = ", ".join(grouped)
    flat_count = sum(len(group.split("/")) for group in grouped)
    if flat_count <= max_hexes:
        return text
    selected: List[str] = []
    used = 0
    for group in grouped:
        group_count = len(group.split("/"))
        if selected and used + group_count > max_hexes:
            break
        selected.append(group)
        used += group_count
    return ", ".join(selected)


def _attach_hex_anchors_to_color_text(
    text: str,
    color_contract: Optional[Mapping[str, Any]],
    *,
    auxiliary_reference_text: str = "",
) -> str:
    roles = _hex_anchor_roles(color_contract, auxiliary_reference_text=auxiliary_reference_text)
    if not roles:
        return text
    out: List[str] = []
    used_role_indexes = set()
    used_hexes = set()
    for clause in _split_clauses(text):
        clause_clean = _clean_clause(clause)
        if not clause_clean:
            continue
        role_indexes = _matching_hex_anchor_role_indexes(clause_clean, roles)
        if not role_indexes:
            out.append(clause_clean)
            continue
        for role_index in role_indexes:
            used_role_indexes.add(role_index)
        hexes = _dedupe_hexes(
            hex_value
            for role_index in role_indexes
            for hex_value in (roles[role_index].get("hexes") or [])
        )[:2]
        used_hexes.update(hex_value.lower() for hex_value in hexes)
        out.append(_append_hex_anchor_to_clause(clause_clean, hexes))

    extra = [
        _hex_anchor_fallback_clause(role)
        for index, role in enumerate(roles)
        if index not in used_role_indexes
        and not any(str(hex_value).lower() in used_hexes for hex_value in (role.get("hexes") or []))
        and role.get("priority", 9) <= 2
    ]
    return "; ".join([*out, *[item for item in extra if item]])


def _hex_anchor_roles(
    color_contract: Optional[Mapping[str, Any]],
    *,
    auxiliary_reference_text: str = "",
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = _hex_anchor_roles_from_auxiliary_text(auxiliary_reference_text)
    if not color_contract:
        return out[:6]
    roles = color_contract.get("color_roles")
    if not isinstance(roles, list):
        return out[:6]
    fixed_hexes = _dedupe_hexes(_HEX_COLOR_RE.findall(auxiliary_reference_text or ""))
    for role in roles:
        if not isinstance(role, Mapping):
            continue
        color_family = str(role.get("color_family") or "").strip()
        role_hexes = _dedupe_hexes(_HEX_COLOR_RE.findall(color_family))
        if not role_hexes:
            continue
        preferred = [hex_value for hex_value in fixed_hexes if hex_value.lower() in {item.lower() for item in role_hexes}]
        hexes = _dedupe_hexes(preferred or role_hexes)[:2]
        tokens = _color_family_tokens(color_family)
        if not tokens or not hexes:
            continue
        visual_rank = str(role.get("visual_rank") or "").strip().lower()
        dominance_type = str(role.get("dominance_type") or "").strip().lower()
        out.append(
            {
                "tokens": tokens,
                "hexes": hexes,
                "label": _compact_color_family_label(color_family),
                "visual_rank": visual_rank,
                "dominance_type": dominance_type,
                "priority": _hex_anchor_role_priority(visual_rank, dominance_type),
            }
        )
    deduped: List[Dict[str, Any]] = []
    seen_by_hex_set: Dict[Tuple[str, ...], Dict[str, Any]] = {}
    for role in sorted(out, key=lambda item: (item.get("source") != "auxiliary_reference_text", item["priority"])):
        key = tuple(str(item).lower() for item in role.get("hexes") or [])
        if not key:
            continue
        existing = seen_by_hex_set.get(key)
        if existing is not None:
            for token in role.get("tokens") or []:
                if token not in existing["tokens"]:
                    existing["tokens"].append(token)
            existing["priority"] = min(int(existing.get("priority", 9)), int(role.get("priority", 9)))
            continue
        seen_by_hex_set[key] = role
        deduped.append(role)
    return deduped[:8]


def _hex_anchor_roles_from_auxiliary_text(auxiliary_reference_text: str) -> List[Dict[str, Any]]:
    roles: List[Dict[str, Any]] = []
    for clause in _split_clauses(auxiliary_reference_text or ""):
        for match in _HEX_COLOR_RE.finditer(clause):
            hex_value = match.group(0)
            label = _local_color_label_before_hex(clause[: match.start()])
            tokens = _color_family_tokens(label)
            if not tokens:
                continue
            lowered = label.lower()
            priority = 3
            if any(term in lowered for term in ("accent", "vivid", "orange", "gold", "crimson", "red")):
                priority = 0
            elif any(term in lowered for term in ("glow", "highlight", "bright", "cream", "cyan", "ivory")):
                priority = 1
            elif any(term in lowered for term in ("shadow", "base", "deep", "dark", "navy")):
                priority = 2
            roles.append(
                {
                    "tokens": tokens,
                    "hexes": [hex_value.upper()],
                    "label": label,
                    "visual_rank": "",
                    "dominance_type": "",
                    "priority": priority,
                    "source": "auxiliary_reference_text",
                }
            )
    return roles


def _local_color_label_before_hex(prefix: str) -> str:
    words = re.findall(r"[A-Za-z]+(?:-[A-Za-z]+)?", prefix)
    keep: List[str] = []
    for word in reversed(words):
        lowered = word.lower()
        if lowered in {"and", "as", "forms", "is", "repeating", "through", "while", "with"}:
            if keep:
                break
            continue
        keep.append(word)
        if len(keep) >= 4:
            break
    return " ".join(reversed(keep)).strip()


def _dedupe_hexes(values: Iterable[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for value in values:
        clean = str(value or "").strip()
        lowered = clean.lower()
        if not clean or lowered in seen:
            continue
        seen.add(lowered)
        out.append(clean.upper())
    return out


def _color_family_tokens(color_family: str) -> List[str]:
    clean = _HEX_COLOR_RE.sub(" ", color_family.lower())
    clean = re.sub(r"\b(?:such as|around)\b.*$", " ", clean)
    raw_tokens = re.findall(r"[a-z]+(?:-[a-z]+)?", clean)
    tokens: List[str] = []
    for token in raw_tokens:
        parts = [token]
        if "-" in token:
            parts.extend(part for part in token.split("-") if part)
        for part in parts:
            if len(part) < 3 or part in _COLOR_TOKEN_STOPWORDS:
                continue
            if part not in tokens:
                tokens.append(part)
    return tokens[:8]


def _compact_color_family_label(color_family: str) -> str:
    clean = _HEX_COLOR_RE.sub(" ", str(color_family or ""))
    clean = re.sub(r"\b(?:such as|around)\b.*$", "", clean, flags=re.IGNORECASE)
    clean = re.sub(r"\s*/\s*", "/", clean)
    clean = _clean_clause(clean)
    words = clean.split()
    return " ".join(words[:6]).strip(" ,.;") or "reference color"


def _hex_anchor_role_priority(visual_rank: str, dominance_type: str) -> int:
    lowered = f"{visual_rank} {dominance_type}"
    if "large" in lowered or "accent" in lowered or "material_surface_mass" in lowered:
        return 0
    if "medium" in lowered or "highlight" in lowered or "glow" in lowered or "negative_space" in lowered:
        return 1
    if "dominant" in lowered or "structural" in lowered or "shadow" in lowered:
        return 2
    return 3


def _matching_hex_anchor_role_indexes(clause: str, roles: Sequence[Mapping[str, Any]]) -> List[int]:
    normalized = _normalize_for_match(clause)
    scored: List[Tuple[int, int, int]] = []
    for index, role in enumerate(roles):
        tokens = role.get("tokens") or []
        score = sum(1 for token in tokens if re.search(rf"\b{re.escape(str(token))}\b", normalized))
        if score > 0:
            scored.append((-score, int(role.get("priority", 9)), index))
    scored.sort()
    if not scored:
        return []
    best_score = scored[0][0]
    return [index for score, _, index in scored if score == best_score][:2]


def _append_hex_anchor_to_clause(clause: str, hexes: Sequence[str]) -> str:
    if not hexes or _HEX_COLOR_RE.search(clause):
        return clause
    return _clean_clause(f"{clause} ({'/'.join(hexes)})")


def _hex_anchor_fallback_clause(role: Mapping[str, Any]) -> str:
    label = str(role.get("label") or "reference color")
    hexes = role.get("hexes") or []
    if not hexes:
        return ""
    return _clean_clause(f"{label} anchors carrier color {'/'.join(str(item) for item in hexes)}")


def _mechanism_role_text(
    contract: Mapping[str, Any],
    *,
    include_terms: Sequence[str],
    fallback: str,
) -> str:
    roles = contract.get("mechanism_roles")
    if not isinstance(roles, list):
        return fallback
    selected: List[str] = []
    lowered_terms = tuple(term.lower() for term in include_terms)
    for role in roles:
        if not isinstance(role, Mapping):
            continue
        haystack = " ".join(
            str(role.get(key) or "")
            for key in ("mechanism", "evidence_across_references", "generator_safe_wording")
        ).lower()
        if any(term in haystack for term in lowered_terms):
            wording = str(role.get("generator_safe_wording") or "").strip()
            if wording:
                selected.append(wording)
    return " ".join(selected[:3]) if selected else fallback


def _style_identity_text(detail_contract: Mapping[str, Any], inner: Mapping[str, Any]) -> str:
    for text in (
        _first_non_empty(inner, "generator_safe_wording"),
        _first_non_empty(detail_contract, "compiled_prompt_note"),
        _first_non_empty(detail_contract, "detail_mechanism_summary"),
    ):
        clean = _clean_clause(text)
        if clean:
            return clean
    return "Preserve the reference-derived style identity as portable rendering behavior."


def _first_non_empty(mapping: Mapping[str, Any], *keys: str, default: str = "") -> str:
    for key in keys:
        value = mapping.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return default


def _join_non_empty(*parts: str, default: str = "") -> str:
    values = [part.strip() for part in parts if isinstance(part, str) and part.strip()]
    return " ".join(values) if values else default


def _candidate_clauses(text: str, *, split_commas: bool = False) -> List[str]:
    clauses: List[str] = []
    for raw in _split_clauses(text):
        clause = _clean_clause(_compact_prompt_clause(raw))
        if not clause:
            continue
        candidates = _comma_segments(clause) if split_commas else [clause]
        for candidate in candidates:
            candidate = _clean_clause(candidate)
            if not candidate:
                continue
            if len(candidate.split()) > 34:
                candidate = _trim_dangling_connectors(" ".join(candidate.split()[:34]).rstrip(" ,.;"))
            _append_unique(clauses, candidate)
    return clauses


def _split_clauses(text: str) -> List[str]:
    return [item.strip(" ,.;") for item in _CLAUSE_SPLIT_RE.split(str(text or "")) if item.strip(" ,.;")]


def _clean_clause(text: str) -> str:
    text = _WS_RE.sub(" ", str(text or "").strip(" ,.;:-"))
    text = re.sub(r"^(?:and|or|with)\s+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+([,.;:])", r"\1", text)
    return _trim_dangling_connectors(text).strip(" ,.;")


def _compact_prompt_clause(text: str) -> str:
    clean = str(text or "")
    clean = re.sub(r"\brather than\b.*$", "", clean, flags=re.IGNORECASE)
    clean = re.sub(r"\bwhile preserving\b.*$", "", clean, flags=re.IGNORECASE)
    clean = re.sub(r"\bwithout requiring\b.*$", "", clean, flags=re.IGNORECASE)
    clean = re.sub(r"\bshould not dictate\b.*$", "", clean, flags=re.IGNORECASE)
    clean = re.sub(r"\bavoid\b.*$", "", clean, flags=re.IGNORECASE)
    clean = re.sub(r"\bbuild\s+compositions?\b", " ", clean, flags=re.IGNORECASE)
    return clean


def _comma_segments(text: str) -> List[str]:
    parts = [_clean_clause(part) for part in re.split(r",\s*", str(text or ""))]
    return [part for part in parts if part]


def _mechanism_key_limit(key: str, *, high_risk: bool) -> int:
    if not high_risk:
        return {
            "surface_rendering": 190,
            "line_rhythm": 150,
            "shape_proportion_language": 145,
            "texture_mark_system": 145,
            "detail_density": 135,
            "ornamental_contour": 120,
            "radial_visual_flow": 110,
            "color_carrier_binding": 180,
        }.get(key, 125)
    return {
        "surface_rendering": 240,
        "line_rhythm": 170,
        "shape_proportion_language": 150,
        "texture_mark_system": 170,
        "detail_density": 150,
        "ornamental_contour": 140,
        "radial_visual_flow": 120,
        "color_carrier_binding": 260,
    }.get(key, 125)


def _mechanism_clause_count(key: str, *, high_risk: bool) -> int:
    if not high_risk:
        return 2
    return {
        "surface_rendering": 3,
        "line_rhythm": 2,
        "shape_proportion_language": 2,
        "texture_mark_system": 2,
        "detail_density": 2,
        "ornamental_contour": 1,
        "radial_visual_flow": 1,
        "color_carrier_binding": 2,
    }.get(key, 2)


def _compile_color_carrier_clauses(
    clauses: Sequence[str],
    limit: int,
    *,
    high_risk: bool,
) -> str:
    compact = [
        _compact_color_carrier_clause(clause, high_risk=high_risk)
        for clause in clauses
    ]
    compact = [clause for clause in compact if clause]
    if high_risk:
        compact = _order_high_risk_color_clauses(compact)
    if not compact:
        return ""
    prefix = "Bind named/hex anchors visibly by carrier: " if high_risk else ""
    body_limit = max(24, limit - len(prefix))
    body = _join_limited(compact, body_limit, separator="; ")
    return _clean_clause(prefix + body)


def _compact_color_carrier_clause(text: str, *, high_risk: bool) -> str:
    clean = _clean_clause(text)
    if not clean:
        return ""
    clean = re.sub(r"\bbelongs?\s+to\b", "for", clean, flags=re.IGNORECASE)
    clean = re.sub(r"\battach(?:es)?\s+to\b", "for", clean, flags=re.IGNORECASE)
    clean = re.sub(r"\bare\s+only\s+for\b", "only for", clean, flags=re.IGNORECASE)
    clean = re.sub(r"\bis\s+only\s+for\b", "only for", clean, flags=re.IGNORECASE)
    clean = re.sub(r"\bis\s+the\s+dominant\b", "dominant", clean, flags=re.IGNORECASE)
    clean = re.sub(r"\bthe\s+dominant\b", "dominant", clean, flags=re.IGNORECASE)
    clean = re.sub(r"\bunder\s+warm\s+theatrical\s+light\b", "under warm light", clean, flags=re.IGNORECASE)
    if high_risk:
        if re.search(r"\b(?:shadow|shadows|recess|recesses|silhouette|silhouettes|contour|darks|dark|black|brown)\b", clean, flags=re.IGNORECASE):
            clean = re.sub(r"^Dominant\s+", "Structural ", clean, flags=re.IGNORECASE)
        clean = re.sub(
            r"\bStructural\s+[^.;()]*\((#[0-9A-Fa-f]{6}(?:/#[0-9A-Fa-f]{6})?)\)\s+carry\b.*$",
            r"structural dark \1 limited to silhouette clarity, recesses, contact shadows, occluded folds",
            clean,
            flags=re.IGNORECASE,
        )
        clean = re.sub(
            r"\bStructural\s+.*?\bcarry\b.*\((#[0-9A-Fa-f]{6}(?:/#[0-9A-Fa-f]{6})?)\)$",
            r"structural dark \1 limited to silhouette clarity, recesses, contact shadows, occluded folds",
            clean,
            flags=re.IGNORECASE,
        )
        clean = re.sub(r"\blocal\s+shadow\s+structure,\s*", "local ", clean, flags=re.IGNORECASE)
        clean = re.sub(r"\bbounded\s+decorative\s+material\s+planes\b", "primary compatible subject/material planes", clean, flags=re.IGNORECASE)
        clean = re.sub(r"\bbounded\s+subject/material\s+planes\b", "primary compatible subject/material planes", clean, flags=re.IGNORECASE)
        clean = re.sub(r"\bbounded\s+material\s+planes\b", "primary compatible subject/material planes", clean, flags=re.IGNORECASE)
        clean = re.sub(r"\bmetallic,\s+glossy,\s+or\s+local\s+contour\s+accents\s+carriers\b", "metallic, glossy, and local contour accents", clean, flags=re.IGNORECASE)
        clean = re.sub(r"\bmetallic,\s+glossy,\s+or\s+material\s+accent\s+lines\s+carriers\b", "metallic, glossy, and material accent lines", clean, flags=re.IGNORECASE)
        clean = re.sub(r"\bmetallic,\s+glossy,\s+or\s+material\s+accent\s+linework\s+carriers\b", "metallic, glossy, and material accent linework", clean, flags=re.IGNORECASE)
        clean = re.sub(r"\bmetallic,\s+glossy,\s+and\s+local\s+contour\s+accents\b", "metallic, glossy, local contour accents, and broad open material bands", clean, flags=re.IGNORECASE)
        clean = re.sub(r"\bmetallic,\s+glossy,\s+and\s+material\s+accent\s+lines\b", "metallic, glossy, material accent lines, and broad open material bands", clean, flags=re.IGNORECASE)
        clean = re.sub(r"\bmetallic,\s+glossy,\s+and\s+material\s+accent\s+linework\b", "metallic, glossy, material accent linework, and broad open material bands", clean, flags=re.IGNORECASE)
        clean = re.sub(
            r"\blarge\s+(.+?)\s+for\s+primary compatible subject/material planes\s+and\s+(?:radial|radiating)\s+fields\b",
            r"large \1 fills compatible focal subject-worn material surfaces, object material surfaces, broad scene color fields, and radiating material fields",
            clean,
            flags=re.IGNORECASE,
        )
        clean = re.sub(
            r"\blarge\s+(.+?)\s+for\s+metallic,\s+glossy,\s+(?:local\s+contour\s+accents|material\s+accent\s+lines|material\s+accent\s+linework),\s+and\s+broad\s+open\s+material\s+bands\b",
            r"large \1 forms broad open reflective/radiating material bands, reflective planes, and metallic/glossy surfaces with etched detail",
            clean,
            flags=re.IGNORECASE,
        )
        clean = re.sub(r"\blocal\s+light\s+shapes,\s*", "local light, ", clean, flags=re.IGNORECASE)
        clean = re.sub(
            r"\bcarries\s+glow,\s+local\s+light,\s+negative\s+space,\s+and\s+reflected\s+light\b",
            "for controlled luminous backglow, light rays, reflected highlights, negative-space light, not focal subject-worn/object material surfaces unless content names pale local material",
            clean,
            flags=re.IGNORECASE,
        )
        clean = re.sub(
            r"\b((?:medium\s+)?(?:pale\s+)?(?:cream|ivory|light|pale)[^.;,]*?)\s+carries\s+glow\b",
            r"\1 for glow, reflected highlights, light rays, and negative-space light, not focal subject-worn/object material surfaces unless content names pale local material",
            clean,
            flags=re.IGNORECASE,
        )
        clean = re.sub(
            r"\bfor\s+bloom,\s+rim\s+light,\s+glow,\s+and\s+specular\s+strips\b",
            "for local bloom, rim light, glow, and specular strips",
            clean,
            flags=re.IGNORECASE,
        )
        clean = re.sub(
            r"\blocal\s+subject\s+colors\s+remain\s+plausible\s+under\s+warm\s+light\b",
            "unspecified subject-worn/object material surfaces default to large/accent color roles before pale/light roles; specified local identity colors stay plausible under warm light; pale/light roles stay in glow, reflected highlights, light rays, and negative-space light",
            clean,
            flags=re.IGNORECASE,
        )
    return _clean_clause(clean)


def _order_high_risk_color_clauses(clauses: Sequence[str]) -> List[str]:
    indexed = list(enumerate(clauses))
    indexed.sort(key=lambda item: (_high_risk_color_clause_priority(item[1]), item[0]))
    return [clause for _, clause in indexed]


def _high_risk_color_clause_priority(clause: str) -> int:
    lowered = clause.lower()
    if any(term in lowered for term in ("dominant base", "base and highlight carrier", "dominant atmosphere")):
        return 0
    if any(term in lowered for term in ("large ", "broad ", "fills ", "fill ", "forms ", "occupy ", "material planes", "material bands", "scene fields", "color fields", "graphic fields", "radiating fields")):
        return 1
    if "local subject colors" in lowered or "local identity colors" in lowered or "unspecified subject-worn" in lowered or "plausible local" in lowered or "local material colors" in lowered:
        return 3
    if "tiny" in lowered or "only for glints" in lowered:
        return 4
    if any(term in lowered for term in ("medium ", "glow", "bloom", "highlight", "rim light", "reflected light", "light breaks")):
        return 2
    if any(term in lowered for term in ("shadow", "shadows", "recess", "recesses", "silhouette", "silhouettes", "structural line", "dark", "black", "brown")):
        return 3
    return 2


def _compile_high_risk_flow_clauses(clauses: Sequence[str], limit: int) -> str:
    compact = [_clean_clause(re.sub(r"^favor\s+", "", clause, flags=re.IGNORECASE)) for clause in clauses]
    compact = [clause for clause in compact if clause]
    if not compact:
        return ""
    prefix = _high_risk_flow_prefix(compact)
    body_limit = max(24, limit - len(prefix))
    body = _join_limited(compact, body_limit, separator=", ")
    return _clean_clause(prefix + body)


def _high_risk_flow_prefix(clauses: Sequence[str]) -> str:
    joined = " ".join(clauses).lower()
    has_ray_material_cue = any(
        cue in joined
        for cue in (
            "radiating",
            "reflective",
            "light ray",
            "light rays",
            "backglow",
            "metallic",
            "glow",
        )
    )
    if has_ray_material_cue:
        return (
            "Make light rays visibly radiate as broad continuous reflective/radiating material bands; "
            "keep bands thick, layered, carrier-bound across open light and material planes: "
        )
    return (
        "Carry open directional flow through material/accent color fields; preserve reflective/radiating material "
        "bands and light rays as interior surface behavior: "
    )


def _radial_flow_text_from_sources(auxiliary_reference_text: str, inner: Mapping[str, Any]) -> str:
    candidates: List[str] = []
    for source in (
        auxiliary_reference_text,
        _first_non_empty(inner, "composition_depth_prior", "composition_and_camera_prior"),
        _first_non_empty(inner, "shape_proportion_language"),
    ):
        for clause in re.split(r"[,;\n]+", str(source or "")):
            clean = _clean_clause(clause)
            lowered = clean.lower()
            if not clean or _looks_like_literal_container(clean):
                continue
            if "#" in clean:
                continue
            if lowered.startswith("with "):
                continue
            if "inward" in lowered:
                continue
            if any(term in lowered for term in _RADIAL_FLOW_TERMS):
                _append_unique(candidates, clean)
    if candidates:
        return "; ".join(candidates[:4])
    return "No reference-specific radial or directional flow mechanism extracted."


def _sanitize_positive_text(
    text: str,
    literal_items: Sequence[str],
    *,
    remove_soft: bool = False,
    high_risk: bool = False,
) -> str:
    clean = str(text or "")
    for pattern, replacement in _CONTAINER_REPLACEMENTS:
        clean = pattern.sub(replacement, clean)
    if high_risk:
        for pattern, replacement in _HIGH_RISK_MECHANISM_REPLACEMENTS:
            clean = pattern.sub(replacement, clean)
        clean = _TITLE_CASE_STYLE_LABEL_RE.sub("reference-derived", clean)
    for item in literal_items:
        item_clean = _clean_clause(item)
        if len(item_clean.split()) >= 2:
            clean = re.sub(re.escape(item_clean), "portable style behavior", clean, flags=re.IGNORECASE)
    if remove_soft:
        for pattern in _SOFT_POSITIVE_REMOVALS:
            clean = pattern.sub(" ", clean)
    clean = _WS_RE.sub(" ", clean)
    clean = re.sub(r"\s+([,.;:])", r"\1", clean)
    clean = re.sub(r"\b(?:with|and|or)\s*([,.;])", r"\1", clean, flags=re.IGNORECASE)
    clean = re.sub(r"([,;])\s*(?:and|or)\s*([,.;])", r"\1", clean, flags=re.IGNORECASE)
    clean = re.sub(r"\s+([,.;:])", r"\1", clean)
    clean = re.sub(r"\b(surface-internal contour organization|surface-internal accent linework)(?:\s+(?:and|or)\s+\1)+\b", r"\1", clean, flags=re.IGNORECASE)
    clean = re.sub(r"\bobject\s+subject\s+and\s+material\s+planes\b", "object material planes", clean, flags=re.IGNORECASE)
    clean = re.sub(r"\b(material|surface|style|color|light)\s+\1\b", r"\1", clean, flags=re.IGNORECASE)
    return clean.strip(" ,.;")


def _sanitize_prompt_fragment_text(text: str) -> str:
    clean = str(text or "")
    for pattern, replacement in _CONTAINER_REPLACEMENTS:
        clean = pattern.sub(replacement, clean)
    clean = _WS_RE.sub(" ", clean)
    clean = re.sub(r"\s+([,.;:])", r"\1", clean)
    clean = re.sub(r"\b(material|surface|style|color|light)\s+\1\b", r"\1", clean, flags=re.IGNORECASE)
    return _clean_clause(clean)


def _has_literal_container_risk(literal_items: Sequence[str]) -> bool:
    return bool(literal_items)


def _looks_like_literal_container(value: str) -> bool:
    return any(pattern.search(value) for pattern in _GENERIC_LITERAL_PATTERNS)


def _looks_like_soft_composition(value: str) -> bool:
    lowered = value.lower()
    return any(term in lowered for term in _SOFT_COMPOSITION_TERMS)


def _is_noop_mechanism_text(value: str) -> bool:
    lowered = _normalize_for_match(value)
    return lowered.startswith("no reference specific") and "mechanism extracted" in lowered


def _extract_literal_labels(text: str) -> List[str]:
    labels: List[str] = []
    for pattern, label in _LITERAL_EXTRACTORS:
        if pattern.search(text):
            _append_unique(labels, label)
    return labels


def _strip_optional_prefix(text: str) -> str:
    return re.sub(r"^(?:optional(?:ly)?|weak(?:ly)?|soft(?:ly)?)[,:]?\s*", "", text.strip(), flags=re.IGNORECASE)


def _join_limited(parts: Sequence[str], limit: int, *, separator: str = "; ") -> str:
    selected: List[str] = []
    for part in parts:
        clean = _clean_clause(part)
        if not clean:
            continue
        candidate = separator.join(selected + [clean])
        if len(candidate) <= limit:
            selected.append(clean)
            continue
        if not selected:
            selected.append(_truncate_words(clean, limit))
        break
    return separator.join(selected).strip(" ,.;")


def _join_limited_with_required_tail(
    parts: Sequence[str],
    limit: int,
    *,
    required_tail: Sequence[str],
    separator: str = "; ",
) -> str:
    """Join prompt clauses while preserving required tail clauses."""
    clean_parts = [_clean_clause(part) for part in parts if _clean_clause(part)]
    clean_required = [_clean_clause(part) for part in required_tail if _clean_clause(part)]
    selected: List[str] = []
    for part in clean_parts:
        if part in selected:
            continue
        _append_unique(selected, part)
    for part in clean_required:
        if part not in selected:
            selected.append(part)

    while clean_required and len(separator.join(selected)) > limit:
        drop_index = next(
            (
                index
                for index in range(len(selected) - 1, -1, -1)
                if selected[index] not in clean_required
            ),
            -1,
        )
        if drop_index < 0:
            break
        selected.pop(drop_index)

    if len(separator.join(selected)) <= limit:
        return separator.join(selected).strip(" ,.;")

    compressed_required = [
        _compact_required_tail_clause(part, max(72, (limit - 2 * max(0, len(clean_required) - 1)) // max(1, len(clean_required))))
        for part in clean_required
    ]
    return _join_limited(compressed_required, limit, separator=separator)


def _compact_required_tail_clause(clause: str, limit: int) -> str:
    clean = _clean_clause(clause)
    lowered = clean.lower()
    if lowered.startswith("color/value distribution:"):
        body = _strip_prompt_prefix(clean, "color/value distribution:")
        parts = [_clean_clause(part) for part in body.split(";") if _clean_clause(part)]
        priority = []
        for marker in (
            "chromatic material fields",
            "carrier-area coverage",
            "near-monochrome key",
            "black-white-gray steps",
            "dark/black",
            "light/white",
            "light islands",
            "dark-light separation",
            "wide value range",
        ):
            part = next((item for item in parts if marker in item.lower()), "")
            if part:
                _append_unique(priority, part)
        if not priority:
            priority = parts[:2]
        return _clean_clause(
            "Color/value distribution: "
            + _join_limited(priority, max(48, limit - len("Color/value distribution: ")), separator="; ")
        )
    if lowered.startswith("carrier arbitration:"):
        return _clean_clause(
            "carrier arbitration: selected/accent/local on small surfaces; exclude whole subject body, "
            "whole clothing mass, whole tabletop, whole background; do not recolor entire semantic objects"
        )
    return _truncate_words(clean, limit)


def _truncate_words(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    words = text.split()
    out: List[str] = []
    for word in words:
        candidate = " ".join(out + [word])
        if len(candidate) > max(24, limit):
            break
        out.append(word)
    while out and out[-1].strip(" ,.;").lower() in _DANGLING_PROMPT_TAIL_TOKENS:
        out.pop()
    truncated = " ".join(out).strip(" ,.;")
    return _strip_dangling_modifier_fragment(_strip_unclosed_scope_marker(truncated))


def _strip_unclosed_scope_marker(text: str) -> str:
    clean = str(text or "").strip(" ,.;")
    open_index = clean.rfind("[")
    close_index = clean.rfind("]")
    if open_index > close_index:
        clean = clean[:open_index].strip(" ,.;")
    return clean


def _trim_dangling_connectors(text: str) -> str:
    clean = str(text or "").strip(" ,.;")
    dangling = "|".join(re.escape(token) for token in sorted(_DANGLING_PROMPT_TAIL_TOKENS, key=len, reverse=True))
    clean = re.sub(rf"(?:\s+|^)(?:{dangling})$", "", clean, flags=re.IGNORECASE)
    clean = re.sub(rf"(?:,\s*)?\b(?:{dangling})\b\s*$", "", clean, flags=re.IGNORECASE)
    return _strip_dangling_modifier_fragment(clean).strip(" ,.;")


def _strip_dangling_modifier_fragment(text: str) -> str:
    clean = str(text or "").strip(" ,.;")
    head, separator, tail = clean.rpartition(",")
    if not separator:
        return clean
    tail_tokens = [token.strip(" .,:;()[]").lower() for token in tail.split()]
    if not tail_tokens or len(tail_tokens) > 4:
        return clean
    if all(_is_dangling_modifier_tail_token(token) for token in tail_tokens):
        return head.strip(" ,.;")
    return clean


def _is_dangling_modifier_tail_token(token: str) -> bool:
    if token in _DANGLING_MODIFIER_TAIL_TOKENS:
        return True
    if "-" in token:
        parts = [part for part in token.split("-") if part]
        return bool(parts) and all(part in _DANGLING_MODIFIER_TAIL_TOKENS for part in parts)
    return False


def _finish_sentence(text: str) -> str:
    clean = _clean_clause(text)
    if not clean:
        return ""
    return clean if clean.endswith(".") else clean + "."


def _append_unique(items: List[str], item: str) -> None:
    clean = _clean_clause(item)
    if not clean:
        return
    clean_norm = _normalize_for_match(clean)
    if any(_normalize_for_match(existing) == clean_norm for existing in items):
        return
    items.append(clean)


def _append_unique_nonredundant(items: List[str], item: str) -> None:
    clean = _clean_clause(item)
    if not clean:
        return
    clean_norm = _normalize_for_match(clean)
    if any(_normalize_for_match(existing) == clean_norm for existing in items):
        return
    clean_tokens = set(clean_norm.split())
    if clean_tokens:
        for existing in items:
            existing_norm = _normalize_for_match(existing)
            existing_tokens = set(existing_norm.split())
            if not existing_tokens:
                continue
            overlap = len(clean_tokens & existing_tokens) / max(1, min(len(clean_tokens), len(existing_tokens)))
            if overlap >= 0.82:
                return
    items.append(clean)


def _dedupe(items: Iterable[str]) -> List[str]:
    out: List[str] = []
    for item in items:
        _append_unique(out, item)
    return out


def _normalize_for_match(text: str) -> str:
    lowered = str(text or "").lower()
    lowered = re.sub(r"[^a-z0-9#]+", " ", lowered)
    return _WS_RE.sub(" ", lowered).strip()
