# Experimental shrinking YOCO tail

Opt in with DS4_YOCO_CONE=1 in addition to the guarded YOCO15/25 settings.
The flag defaults off. This branch is experimental, not the serving default.
Only a whole 2668-row tail inside the final chunk is supported; other shapes
fall back to the rectangular YOCO tail. No changes to split, quantization,
expert selection, sampling, source20 history or handoff acknowledgements.

For layers 20..39 the processed suffix lengths are 2668 - 127*(layer-20).
The input h/pre suffix is correct by induction. Up to the first 127 outputs
may depend on missing local history; the next layer excludes those rows.
Layer39 leaves 128 correct outputs. Its 255 correct inputs cover current
128-row KV rings and the previous-position snapshot (129 input rows in union).
DSpark taps are taken before layers37..39 and retain the last128 inputs.
Full source20 CKV/IK is still generated for the entire prompt.

Selections are stored in original chunk-row coordinates. Suffix views shift
h/pre/topk/cand together, but compressed-key IDs inside selections are not
changed. All pointers are restored on a returned error as well as success.
The row helper is shared by computation and speculative streaming sizes.
Chunk presence/acknowledgements remain the rectangular schedule because the
admitted cone touches the same final chunk on every late layer.

CPU validation: yoco_cone_test.py compiles the real range helpers and checks
supported/fallback boundaries. A nonlinear exact-integer window oracle poisons
missing local history and checks retained outputs, inputs needed for rings and
snapshot, pre-layer taps and absolute selection coordinates at zero and nonzero
positions. The old pipeline topology test also passes. These tests do not
validate GPU kernels, numerical parity, actual checkpoint serialization or
all speculative rollback states. HIP gfx906 build passed; other backends not run.

For20160 input rows, processed late-layer rows decrease53360 to29230;
attention slabs at256 rows decrease220 to120. Indexer rows decrease13340 to8260,
slabs at32 rows decrease420 to260. These counts do not predict wall time or
unique expert transfers. GEMM shape and expert summation changes can introduce
floating-point differences even with a correct dependency graph.

GPU validation is pending. The proposed bounded test uses two rectangular
controls and one cone run, the same20160 input IDs and up to160 emitted tokens.
After timed generation, a test-only harness restores before the last prompt
token, replaces it, and teacher-forces160 fixed input tokens, recording full
logits at11 positions. This does not hold H20 fixed or compare raw GPU states;
control/control variability must be reported and the candidate must not be
called bit-exact based on equal text or one top1 result.
