# Benchmarks

Every number here was measured with this code on rented hardware, not extrapolated.

## Cards

Same source, compiled per architecture; the tag was checked against libargon2 on each card
before timing anything.

| GPU | arch | best `--batch` | H/s | USD/h (Vast, spot price at the time) | H/s per dollar-hour |
|---|---|---|---|---|---|
| RTX 4090 (24 GB) | sm_89 | 1024 | **55,300** | 0.375 | **147,000** |
| RTX 3090 (24 GB) | sm_86 | 768 | **29,100** | 0.219 | 133,000 |

The 3090 is 42 % cheaper per hour but delivers 53 % of the hashrate, so it works out slightly
**more** expensive per hash.

## Batch curve (RTX 4090)

| batch | VRAM | H/s |
|---|---|---|
| 512 | 4 GB | 41,600 |
| 768 | 6 GB | 50,900 |
| **1024** | 8 GB | **56,800** |
| 1536 | 12 GB | 52,400 |
| 2048 | 16 GB | 45,000 |

## Where the 4090 sits

Argon2id `t=1, m=8 MiB` moves about 32 MiB of traffic per hash (8 MiB written, ~24 MiB read):

```
55,300 hash/s x 32 MiB ≈ 1.77 TB/s effective
RTX 4090 HBM bandwidth  = 1.01 TB/s
```

The card cannot do this without help from its 72 MB L2, which is why the kernel is close to the
roofline and why more tuning yields single-digit percentages, not multiples. **55 kH/s is
roughly the practical ceiling for one 4090.**

## Why cheap older cards are a trap

Scaling hashrate by memory bandwidth and dividing by the hourly price suggests that 20-series
and 30-series cards are 2.5x better value. Measurement says otherwise:

| GPU | bandwidth | measured | bandwidth-model prediction | error |
|---|---|---|---|---|
| RTX 4090 | 1008 GB/s | 55,300 | 55,300 | — |
| RTX 3090 | 936 GB/s | 29,100 | 51,400 | **-43 %** |

Argon2's fill stage is BLAKE2b — integer work — and Ampere's integer throughput plus its small
L2 (3090: 6 MB vs 4090: 72 MB) cannot keep up. Turing is expected to be worse still. For this
algorithm, buy the newest architecture with the biggest L2, not the cheapest bandwidth.

## CPU comparison

Same chain, for scale:

| Machine | H/s |
|---|---|
| RTX 4090, this miner | 55,300 |
| Hand-written C miner, 16 threads (Ryzen 7 5800H) | 550 |
| Official Go node miner, 8 threads | 198 |
| Browser (WebAssembly) | 20-60 |

## Caveats

Rented prices move daily; re-check before making a decision. Rows measured on Vast.ai
containers, one card per instance, default power limits.
