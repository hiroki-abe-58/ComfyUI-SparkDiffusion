"""Convert a ComfyUI UI-format workflow into an API prompt (for tests and integration runs)."""

from __future__ import annotations

WIDGET_TYPES = {"INT", "FLOAT", "STRING", "BOOLEAN"}
FRONTEND_ONLY = {"Note", "MarkdownNote"}


def widget_names(input_types: dict) -> list:
    """Widget input names in UI order, with a placeholder after seed-like control widgets."""
    names = []
    for section in ("required", "optional"):
        for name, spec in input_types.get(section, {}).items():
            kind = spec[0]
            opts = spec[1] if len(spec) > 1 and isinstance(spec[1], dict) else {}
            if isinstance(kind, list) or kind in WIDGET_TYPES:
                names.append(name)
                if opts.get("control_after_generate"):
                    names.append(None)  # "fixed" / "randomize" UI-only value
    return names


def ui_to_api(workflow: dict, node_classes: dict) -> dict:
    links = {link[0]: link for link in workflow.get("links", [])}
    prompt = {}
    for node in workflow["nodes"]:
        if node["type"] in FRONTEND_ONLY:
            continue
        cls = node_classes.get(node["type"])
        inputs = {}
        if cls is not None:
            names = widget_names(cls.INPUT_TYPES())
            values = node.get("widgets_values", [])
            if len(names) != len(values):
                raise ValueError(f"{node['type']}: {len(values)} widget values for {len(names)} widgets")
            inputs.update({n: v for n, v in zip(names, values) if n is not None})
        else:  # core node: only keep what the test needs
            if node["type"] == "LoadImage":
                inputs["image"] = node["widgets_values"][0]
        for inp in node.get("inputs", []):
            if inp.get("link") is not None:
                _, src, src_slot, *_ = links[inp["link"]]
                inputs[inp["name"]] = [str(src), src_slot]
        prompt[str(node["id"])] = {"class_type": node["type"], "inputs": inputs}
    return prompt
