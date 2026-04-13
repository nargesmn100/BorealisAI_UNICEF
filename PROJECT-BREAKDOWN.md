# UNICEF × RBC Borealis AI Project

## AI-Powered Reconstruction of Multidimensional Child Deprivation

For the full problem statement, ML formulation, and evaluation framework,
see `docs/problem_statement.pdf`.

---

## Ethical Constraints

- Do not present outputs as official statistics
- Avoid overclaiming precision in fine-resolution maps
- Maintain transparency about uncertainty in all outputs
- Clearly state limitations and failure cases wherever results are shown

---

## Modeling Assumptions

- Proxy variables (RWI, settlement type, accessibility) encode meaningful signals about
  the spatial distribution of child deprivation
- The relationship between proxy signals and deprivation patterns is learnable from
  high-resolution spatial data
- Within-region variation in deprivation exists and can be inferred from proxy signals
  even when official statistics are only available at coarse administrative levels
