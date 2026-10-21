# IPA point-value repair

The canonical `InvariantPointAttention` uses learned 3D value points (`V_p`) in the value aggregation path. Query/key point distances contribute to attention logits; value points are transformed into global coordinates, aggregated over key residues with attention weights, and converted back to the query residue's local frame with `R^T(x_global - t)`.

The output concatenates four tracks: scalar values, local value-point coordinates, invariant point norms, and CHIMERA's explicit pair-to-single values. The point-logit scale is CHIMERA's variance-control choice, not a claim of bit-for-bit numerical equivalence with AlphaFold/OpenFold.

Substrate proximity is applied as a **key-residue bias**. Adding the same proximity value across every key for a query would be removed by softmax and therefore would not condition attention.

## Legacy checkpoint migration

Legacy checkpoints used the layout `[scalar values | one relative-displacement vector per head | pair values]`. The repaired IPA uses `[scalar values | all learned Vp coordinates | all Vp norms | pair values]`. Migration preserves overlapping scalar and pair projection weights, and maps the legacy 3-vector output weights to the first new value-point slot as a **lossy warm start**. The legacy feature was a relative displacement, not a learned value point, so this conversion is not functionally exact. Newly introduced point slots and norm channels retain the target module's initialization. Migration is tested both when head count changes and when it remains the same.

Rigid-transform tests cover the IPA output with and without substrate coordinates, and regression tests ensure the value-point path receives gradients and substrate conditioning changes the key distribution.
