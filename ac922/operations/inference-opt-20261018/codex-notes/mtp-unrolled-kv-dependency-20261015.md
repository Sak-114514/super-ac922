# MTP unrolled KV dependency fixture

## Plan and scope

Build a tiny D=2, four-lane CUDA graph with poisoned future K/V cache cells. Depth 0 writes K/V rows, derives hidden state and GPU argmax token1. Depth 1 consumes hidden/token1 and reads the depth-0 cache rows. Compare two separate graph executions with one unrolled graph in model node order, a deliberate read-first ordering, and an explicit versioned-cache source edge. Replay A/B/A with changed prompt values and row slots. Run first on one V100, then across the 3-GPU Meta backend on the already-authorized island B. No model, engine-source edits, broad I/O, or changes outside this fixture/notes directory.

## CPU preflight

`cpu.log` records sequential reference and model-order/explicit-edge parity. A deliberate read-first alias ordering is evaluated as a scheduler-order hazard demonstration, not treated as the expected path.
