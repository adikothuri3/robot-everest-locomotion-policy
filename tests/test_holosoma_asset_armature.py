"""The generated MJCF's armature must match the Holosoma preset's _ARMATURE list."""

import ast
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from everest_locomotion import REPO_ROOT
from everest_locomotion.robots.manifest import load_manifest


def test_xml_armature_matches_preset():
    xml_path = REPO_ROOT / "assets" / "a3_ultra" / "holosoma" / "a3_ultra_29dof.xml"
    if not xml_path.exists():
        pytest.skip("generated asset missing")
    root = ET.parse(xml_path).getroot()
    xml_armature = {
        j.get("name"): float(j.get("armature", "nan"))
        for j in root.iter("joint")
        if j.get("name", "").endswith("_joint")
    }

    src = (REPO_ROOT / "src/everest_locomotion/holosoma_ext/a3_ultra_presets.py").read_text()
    tree = ast.parse(src)
    armature_list = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id == "_ARMATURE":
                armature_list = ast.literal_eval(node.value)
    assert armature_list is not None

    manifest = load_manifest("a3_ultra")
    for name, expected in zip(manifest.control_joints, armature_list):
        assert name in xml_armature, f"{name} missing armature in XML"
        np.testing.assert_allclose(xml_armature[name], expected, atol=1e-6, err_msg=name)
