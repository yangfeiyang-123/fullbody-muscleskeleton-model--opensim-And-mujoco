# MimicMSK OpenSim Model

This directory is generated from `../MimicMSK_Model_mujoco` and is intentionally separate from the original MuJoCo model.

## Files

- `MimicMSK_OpenSim.osim`: generated OpenSim 4.5 model aligned to the MuJoCo skeleton, meshes, markers, tendon paths, wrap objects, muscles, and passive ligaments.
- `MimicMSK_OpenSim_skeleton.osim`: generated OpenSim 4.5 model without muscles. Use this first if the OpenSim GUI exits while importing the full model.
- `MimicMSK_OpenSim_bones.osim`: lighter OpenSim 4.5 model with bodies, joints, and meshes only; no muscles or marker set. Use this if the GUI still exits on the skeleton model.
- `MimicMSK_OpenSim_core.osim`: minimal OpenSim 4.5 model with bodies and joints only; no meshes, markers, or muscles. Use this to test whether the OpenSim GUI itself can load the kinematic tree.
- `Geometry/`: copied STL mesh files referenced by the model.
- `convert_mujoco_to_opensim.py`: repeatable offline converter.
- `conversion_summary.json`: conversion counts and known limitations.
- `alignment_report.json`: numeric MuJoCo-vs-OpenSim body, mesh, and marker alignment check.
- `name_mapping.csv`: mapping from MuJoCo names to OpenSim-safe names.

## Conversion Scope

The converter preserves:

- OpenSim assembly tolerance suitable for this multi-constraint converted model (`assembly_accuracy=1e-08`).
- Default OpenSim root pose is set to a Y-up standing view (`root_rx=-pi/2`, `root_ty=0.825`) while preserving the MuJoCo-derived hierarchy and local geometry.
- MuJoCo body hierarchy as OpenSim `BodySet`.
- MuJoCo hinge/slide/free joints as `CustomJoint` coordinates.
- MuJoCo `site` points as OpenSim markers.
- Mesh geometry references, mesh scale factors, and mesh local `pos`/`euler`/`quat` transforms.
- Non-identity MuJoCo mesh transforms are baked into generated STL files so the OpenSim GUI renders them in the same local pose as MuJoCo.
- Tendon site sequences as OpenSim path points.
- Tendon `geom`/`sidesite` entries as OpenSim wrap objects and `PathWrap` entries. `PathWrap` ranges are written with OpenSim's 1-based indexing so the OpenSim 4.5 GUI path visualizer can render them without an `ArrayIndexOutOfBoundsException`; cylinder wrap quadrants are inferred from radial axes rather than the cylinder axis to avoid large erroneous wrap loops.
- Wrap objects, markers, and path points are preserved in the model data but hidden by default in the GUI to match MuJoCo's cleaner visual model and avoid showing tendon helper geometry as large cyan/pink artifacts.
- MuJoCo actuators as OpenSim `Thelen2003Muscle` entries.
- Unactuated MuJoCo spatial tendons as passive OpenSim `Ligament` entries.
- MuJoCo polynomial joint equality constraints as OpenSim `CoordinateCouplerConstraint` entries where OpenSim has independent/dependent coordinate equivalents.

Known limitations:

- MuJoCo contact pairs are not converted to OpenSim contact geometry.
- MuJoCo one-coordinate equality constraints are represented as locked OpenSim coordinates.
- OpenSim and MuJoCo use different muscle dynamics implementations; corresponding force, length, springlength, and control parameters are mapped where OpenSim has equivalent fields.

To regenerate the output, run:

```powershell
python .\convert_mujoco_to_opensim.py
```

The generated model was checked with:

```powershell
opensim-cmd update-file .\MimicMSK_OpenSim.osim .\validation\full_updated.osim
opensim-cmd update-file .\MimicMSK_OpenSim_bones.osim .\validation\bones_updated.osim
opensim-cmd update-file .\MimicMSK_OpenSim_skeleton.osim .\validation\skeleton_updated.osim
```

If the OpenSim GUI previously failed while opening `MimicMSK_OpenSim.osim`, restart the GUI before reopening the regenerated file. The previous failure was caused by 0-based `PathWrap <range>` values that the OpenSim 4.5 GUI interpreted as `range_start - 1`, producing a `-1` path segment index during muscle-path visualization.

The generated `alignment_report.json` currently reports zero body position error, zero marker position error, zero mesh position error, and zero wrap-object position error against the expanded MuJoCo model.
