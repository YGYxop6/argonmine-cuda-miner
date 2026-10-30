# Argonmine CUDA miner

A from-scratch CUDA implementation of the Argonmine (ARG) proof of work — Argon2id with
`t=1, m=8 MiB, p=1` — for NVIDIA GPUs.

**~55 kH/s on an RTX 4090**, against roughly 20–60 H/s for the browser miner and ~200 H/s for
the reference Go node's built-in miner. It has found real mainnet blocks.

```
RTX 4090      55,300 H/s
RTX 3090      29,100 H/s
laptop CPU       550 H/s   (hand-written C, 16 threads)
official node    198 H/s   (Go, 8 threads)
browser app    20-60 H/s   (WASM)
```

## What is proven, and what is not

Verified:

* The kernel is **bit-exact against libargon2** — same tag for the same 120-byte header, and
  the same intermediate blocks (`B2`, `B2047`, `B2048`, `B4095`, `B4096`, `B8191`) as the
  reference C implementation. Checked on both Ada (sm_89) and Ampere (sm_86).
* Real mainnet blocks were found and accepted by the network with this code.

Not provided:

* No pool protocol. This mines **solo** against an Argonmine work API: you either win the
  whole block or get nothing.
* No wallet. It takes an address and a work template; keys are never involved.

## Build

```sh
make                      # Arch=sm_89 by default
make ARCH=sm_86           # RTX 30xx
make ARCH=sm_75           # RTX 20xx
make ARCH=sm_120          # RTX 50xx
```

or directly:

```sh
nvcc -O3 -arch=sm_89 -o argon_gpu src/argon_gpu.cu
```

No other dependencies — BLAKE2b, the Argon2id block function and its address generation are
all implemented in the one `.cu` file.

## Run

### 1. One job at a time

```sh
./argon_gpu --prefix <224 hex chars> --target <target, big-endian hex> --batch 1024 --seconds 20
```

Prints a `HIT <job_id> <nonce>` line when a solution is found. Useful for testing.

### 2. Long-running daemon (recommended)

```sh
./argon_gpu --jobfile /path/to/job.txt --batch 1024
```

`job.txt` is three lines:

```
<job_id>
<header_prefix, 224 hex chars>
<target, big-endian hex>
```

The miner polls the file's mtime at every batch boundary (~20 ms) and reloads it when it
changes, so a fresh template is picked up almost instantly.

**Why this matters:** a new block every ~20 seconds invalidates the template. Restarting the
process per job wastes roughly half the machine's work on dead templates; keeping the process
resident and swapping the job in place keeps it near 100 %.

### 3. The driver

`driver/gpumine.py` does the work-API side. It fetches a template, writes `job.txt` whenever
the chain tip moves, and submits hits:

```sh
python3 driver/gpumine.py --address arg1... --batch 1024
```

Options: `--api` (default `https://explorer.argonmine.com`), `--bin`, `--jobfile`,
`--batch`, `--interval`.

## How the kernel works

One CUDA block (128 threads) computes one Argon2id hash and owns an 8 MiB pad in VRAM.
`--batch N` keeps N hashes in flight, so VRAM use is `N × 8 MiB` (1024 → 8 GB).

Per hash, three kernels:

1. **prepare** — BLAKE2b pre-hash `H0` over (parameters, 120-byte header, salt), then the first
   two blocks `B[0] = H'(H0‖0‖0)` and `B[1] = H'(H0‖1‖0)`.
2. **fill** — 8192 blocks, data-independent addressing for the first slice, data-dependent
   afterwards, with the Argon2id `G` function (`xor` → `P` → `xor`).
3. **finalize** — XOR the last block of the lane and squeeze 32 bytes with `blake2b_long`.

### Four things that are easy to get wrong

These cost hours; every one of them is silent on trivial inputs:

1. BLAKE2b's `G` uses `sigma[2i]` in the first half and **`sigma[2i+1]` in the second** — not
   the same word twice.
2. `fill_block` must run the four column `G`s and then the four diagonal `G`s in **separate
   synchronised phases**; the diagonals read words the columns just wrote.
3. Argon2's address block is compressed **twice**: `fill_block(zero, input, addr)` and then
   `fill_block(zero, addr, addr)`. With one pass, `B2` and `B3` are right and everything from
   `B4` on is wrong.
4. The pre-hash seed buffer is **72 bytes** (64-byte `H0` + LE32 block index + LE32 lane).

## Tuning

`--batch` is the only knob. Measured on an RTX 4090:

| batch | VRAM | H/s |
|---|---|---|
| 512 | 4 GB | 41,600 |
| **1024** | 8 GB | **56,800** |
| 2048 | 16 GB | 45,000 |

1024 is the sweet spot on a 24 GB card. Larger batches lose occupancy; smaller ones leave the
memory system idle.

## Caveats

* The public explorer's `/api/work` is **rate-limited per IP**. For continuous mining run your
  own node (`argonmined`) plus `argonmine-explorer` on `127.0.0.1` and point `--api` at it.
* Reward maturity is 100 blocks.
* Difficulty adjusts every block (ASERT, one-day half-life), so a fixed hashrate earns less as
  the network grows. One 4090 was ~12 % of the network at 47 kH/s total and ~0.7 % a day later
  at 7.5 MH/s total.
* This is not affiliated with the Argonmine project. Use at your own risk; solo mining means
  variance.

## License

MIT — see `LICENSE`.
