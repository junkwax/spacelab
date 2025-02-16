Tutorial: Simulating a Black Hole with Dark Matter
==================================================

1. **Install SpaceLab**:
   .. code-block:: bash
      pip install spacelab

2. **Configure Simulation**:
   Edit `configs/simulation_config.yaml`:
   .. code-block:: yaml
      black_hole:
        mass: 1e6  # Solar masses
        spin: 0.7
      dark_matter:
        mass: 1e-22  # eV
        coupling: 1e-10

3. **Run Simulation**:
   .. code-block:: bash
      python src/simulation.py --config configs/simulation_config.yaml

4. **Analyze Results**:
   Use the `analyze_results.py` script to generate plots.