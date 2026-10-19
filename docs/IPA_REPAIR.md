# IPA point-value repair

The canonical `InvariantPointAttention` now uses its learned 3D value points (`V_p`) in the value aggregation path. Query/key point distances remain in the attention logits; value points are independently transformed to global coordinates, attention-weighted over key residues, translated into the query frame, and rotated by `R^T` back to local coordinates.

The output retains scalar values, local point values, point norms, and CHIMERA's explicit pair-value pathway.

The implementation is intended to be globally SE(3)-invariant at the scalar output: a common rigid transform of all residue frames and translations leaves the resulting single representation unchanged, subject to floating-point tolerance.
