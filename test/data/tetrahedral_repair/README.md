These small neighborhoods reproduce nearly degenerate tetrahedra from
25-terminal trees grown in the pediatric-heart model with seeds 1385, 1883, and 1536.
The original meshing implementation was commit d683d30, with 100 samples per
spline and a prescribed-point tolerance of 1e-6.

Each archive contains the original double-precision `nodes`, four-node `elems`,
and a `constrained` mask. The neighborhoods include all cells incident to the
prescribed vertices of the bad elements. Coordinates have not been rounded,
translated, or rescaled. The archives contain 27 nodes / 57 cells,
17 nodes / 46 cells, and 13 nodes / 32 cells, respectively.

Three nearly collinear spline samples form a face of numerically flat
elements. Seed 1536 has six bad cells in overlapping neighborhoods.
Tests require unchanged node coordinates and
boundary faces, conserved volume, retained constraints, and positive usable
element quality after repair.
