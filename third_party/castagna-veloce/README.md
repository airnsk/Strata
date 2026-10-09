# Castagna Veloce reference

The experimental persistent HC schedule in `src/kernels/cuda/hc_persistent.cuh`
was inspired by [`hc-persist.cu`](https://github.com/benpeterson40/castagna-veloce/blob/3483d462715d61f21ed5877836be35a724446359/ggml/src/ggml-cuda/hc-persist.cu)
at commit `3483d462715d61f21ed5877836be35a724446359`.

Castagna Veloce is MIT licensed, copyright 2023–2026 The ggml authors and 2026
Ben Peterson. The full notice is in [LICENSE](LICENSE) and in the new kernel.

This is an adaptation of the persistent phase scheduling idea to Strata's
existing BF16 arithmetic and scratch ownership. It does not copy the donor's
Q8 HC quantization, Q4/Q5 routed experts, device-global inject stash, fixed
60-CU assembly barrier, or peer inbox protocol. No donor executable or setup
script is required. See [the experimental port](../../docs/CASTAGNA_PORT.md).
