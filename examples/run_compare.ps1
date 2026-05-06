python -m msk_equivalence.compare `
  --osim MimicMSK_Model_opensim/MimicMSK_OpenSim.osim `
  --mjcf MimicMSK_Model_mujoco/body/myofullbody.xml `
  --mapping configs/model_mapping.yaml `
  --out results/equivalence_report
