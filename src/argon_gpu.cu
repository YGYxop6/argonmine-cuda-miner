// argon_gpu.cu - Argon2id (m=8192 KiB, t=1, p=1) nonce search for Argonmine
// one CUDA block (128 threads) per hash; 8 MiB working set per hash in VRAM
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <ctime>
#include <cuda_runtime.h>
#include <chrono>
#include <sys/stat.h>
#include <unistd.h>
static unsigned char d_target_host[32];
static double now_ms(){ return std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

#define CUDA_CHECK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); exit(1);} } while(0)

__device__ __constant__ uint64_t B2IV[8] = {
    0x6a09e667f3bcc908ULL, 0xbb67ae8584caa73bULL, 0x3c6ef372fe94f82bULL, 0xa54ff53a5f1d36f1ULL,
    0x510e527fade682d1ULL, 0x9b05688c2b3e6c1fULL, 0x1f83d9abfb41bd6bULL, 0x5be0cd19137e2179ULL };

__device__ __constant__ uint8_t B2SIGMA[12][16] = {
    {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15},{14,10,4,8,9,15,13,6,1,12,0,2,11,7,5,3},
    {11,8,12,0,5,2,15,13,10,14,3,6,7,1,9,4},{7,9,3,1,13,12,11,14,2,6,5,10,4,0,15,8},
    {9,0,5,7,2,4,10,15,14,1,11,12,6,8,3,13},{2,12,6,10,0,11,8,3,4,13,7,5,15,14,1,9},
    {12,5,1,15,14,13,4,10,0,7,6,3,9,2,8,11},{13,11,7,14,12,1,3,9,5,0,15,4,8,6,2,10},
    {6,15,14,9,11,3,0,8,12,2,13,7,1,4,10,5},{10,2,8,4,7,6,1,5,15,11,9,14,3,12,13,0},
    {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15},{14,10,4,8,9,15,13,6,1,12,0,2,11,7,5,3} };

// BLAKE2 round groups: 4 column groups then 4 diagonal groups
__device__ __constant__ int GROUPS[8][4] = {
    {0,4,8,12},{1,5,9,13},{2,6,10,14},{3,7,11,15},
    {0,5,10,15},{1,6,11,12},{2,7,8,13},{3,4,9,14} };

__device__ __forceinline__ uint64_t rotr64(uint64_t x, int n) { return (x >> n) | (x << (64 - n)); }

__device__ __forceinline__ void b2_g(uint64_t &a, uint64_t &b, uint64_t &c, uint64_t &d, uint64_t x, uint64_t y) {
    a = a + b + x; d = rotr64(d ^ a, 32);
    c = c + d;     b = rotr64(b ^ c, 24);
    a = a + b + y; d = rotr64(d ^ a, 16);
    c = c + d;     b = rotr64(b ^ c, 63);
}

__device__ void b2_compress(uint64_t h[8], const uint8_t *blk, uint64_t t, int last) {
    uint64_t m[16], v[16];
    for (int i = 0; i < 16; i++) { uint64_t x; memcpy(&x, blk + i * 8, 8); m[i] = x; }
    for (int i = 0; i < 8; i++) { v[i] = h[i]; v[i + 8] = B2IV[i]; }
    v[12] ^= t;
    if (last) v[14] = ~v[14];
    for (int r = 0; r < 12; r++) {
        const uint8_t *s = B2SIGMA[r];
        b2_g(v[0], v[4], v[8],  v[12], m[s[0]],  m[s[1]]);
        b2_g(v[1], v[5], v[9],  v[13], m[s[2]],  m[s[3]]);
        b2_g(v[2], v[6], v[10], v[14], m[s[4]],  m[s[5]]);
        b2_g(v[3], v[7], v[11], v[15], m[s[6]],  m[s[7]]);
        b2_g(v[0], v[5], v[10], v[15], m[s[8]],  m[s[9]]);
        b2_g(v[1], v[6], v[11], v[12], m[s[10]], m[s[11]]);
        b2_g(v[2], v[7], v[8],  v[13], m[s[12]], m[s[13]]);
        b2_g(v[3], v[4], v[9],  v[14], m[s[14]], m[s[15]]);
    }
    for (int i = 0; i < 8; i++) h[i] ^= v[i] ^ v[i + 8];
}

struct B2 { uint64_t h[8]; uint64_t t; uint8_t buf[128]; int buflen; };

__device__ void b2_init(B2 *s, int outlen) {
    for (int i = 0; i < 8; i++) s->h[i] = B2IV[i];
    s->h[0] ^= 0x01010000ULL ^ (uint64_t)outlen;
    s->t = 0; s->buflen = 0;
}
__device__ void b2_update(B2 *s, const uint8_t *in, int len) {
    while (len > 0) {
        if (s->buflen == 128) { s->t += 128; b2_compress(s->h, s->buf, s->t, 0); s->buflen = 0; }
        int n = 128 - s->buflen; if (n > len) n = len;
        memcpy(s->buf + s->buflen, in, n);
        s->buflen += n; in += n; len -= n;
    }
}
__device__ void b2_final(B2 *s, uint8_t *out, int outlen) {
    s->t += (uint64_t)s->buflen;
    memset(s->buf + s->buflen, 0, 128 - s->buflen);
    b2_compress(s->h, s->buf, s->t, 1);
    uint8_t full[64];
    for (int i = 0; i < 8; i++) memcpy(full + i * 8, &s->h[i], 8);
    memcpy(out, full, outlen);
}

// Argon2 H' (blake2b_long)
__device__ void blake2b_long(uint8_t *out, uint32_t outlen, const uint8_t *in, int inlen) {
    uint8_t pre[4] = { (uint8_t)outlen, (uint8_t)(outlen >> 8), (uint8_t)(outlen >> 16), (uint8_t)(outlen >> 24) };
    B2 s;
    if (outlen <= 64) {
        b2_init(&s, outlen); b2_update(&s, pre, 4); b2_update(&s, in, inlen); b2_final(&s, out, outlen);
        return;
    }
    uint32_t r = (outlen + 31) / 32 - 2;
    uint8_t V[64];
    b2_init(&s, 64); b2_update(&s, pre, 4); b2_update(&s, in, inlen); b2_final(&s, V, 64);
    memcpy(out, V, 32);
    for (uint32_t i = 2; i <= r; i++) {
        b2_init(&s, 64); b2_update(&s, V, 64); b2_final(&s, V, 64);
        memcpy(out + 32 * (i - 1), V, 32);
    }
    uint32_t rem = outlen - 32 * r;
    if (rem) {
        b2_init(&s, 64); b2_update(&s, V, 64); b2_final(&s, V, 64);
        memcpy(out + 32 * r, V, rem);
    }
}

__global__ void b2test_kernel(uint8_t *out) {
    B2 s; b2_init(&s, 64); b2_update(&s, (const uint8_t *)"abc", 3); b2_final(&s, out, 64);
}

// ---------------------------------------------------------------- Argon2 core
__device__ __forceinline__ uint64_t fBlaMka(uint64_t x, uint64_t y) {
    const uint64_t m = 0xFFFFFFFFULL;
    return x + y + 2ULL * ((x & m) * (y & m));
}
__device__ __forceinline__ void aG(uint64_t *v, int a, int b, int c, int d) {
    v[a] = fBlaMka(v[a], v[b]); v[d] = rotr64(v[d] ^ v[a], 32);
    v[c] = fBlaMka(v[c], v[d]); v[b] = rotr64(v[b] ^ v[c], 24);
    v[a] = fBlaMka(v[a], v[b]); v[d] = rotr64(v[d] ^ v[a], 16);
    v[c] = fBlaMka(v[c], v[d]); v[b] = rotr64(v[b] ^ v[c], 63);
}

// curr = P(prev ^ ref) ^ (prev ^ ref); S = 128 qwords shared scratch
__device__ __forceinline__ void fill_block(const uint64_t *prev, const uint64_t *ref,
                                           uint64_t *curr, uint64_t *S, int t) {
    uint64_t R = prev[t] ^ ref[t];
    S[t] = R;
    __syncthreads();

    // 8 rounds over consecutive 16-word groups: column step, then diagonal step
    for (int phase = 0; phase < 2; phase++) {
        if (t < 32) {
            int round = t >> 2, g = (t & 3) + phase * 4;
            const int *G4 = GROUPS[g];
            int base = round * 16;
            aG(S, base + G4[0], base + G4[1], base + G4[2], base + G4[3]);
        }
        __syncthreads();
    }
    // 8 rounds over (2c+16k, 2c+1+16k) pairs: column step, then diagonal step
    for (int phase = 0; phase < 2; phase++) {
        if (t < 32) {
            int round = t >> 2, g = (t & 3) + phase * 4;
            const int *G4 = GROUPS[g];
            int i0 = 2 * round + (G4[0] & 1) + 16 * (G4[0] >> 1);
            int i1 = 2 * round + (G4[1] & 1) + 16 * (G4[1] >> 1);
            int i2 = 2 * round + (G4[2] & 1) + 16 * (G4[2] >> 1);
            int i3 = 2 * round + (G4[3] & 1) + 16 * (G4[3] >> 1);
            aG(S, i0, i1, i2, i3);
        }
        __syncthreads();
    }

    curr[t] = S[t] ^ R;
    __syncthreads();
}

__device__ __forceinline__ uint32_t index_alpha(uint32_t slice, uint32_t index, uint32_t pseudo_rand) {
    const uint32_t seg = 2048, lane_len = 8192;
    uint32_t area = (slice == 0) ? (index - 1) : (slice * seg + index - 1);
    uint64_t rel = pseudo_rand;
    rel = (rel * rel) >> 32;
    rel = (uint64_t)area - 1 - (((uint64_t)area * rel) >> 32);
    return (uint32_t)rel % lane_len;
}

// -------------------------------------------------------------- prepare kernel
__global__ void prepare_kernel(const uint8_t *headers, uint8_t *mem, int hashes, uint8_t *h0out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= hashes) return;
    uint64_t *B = (uint64_t *)(mem + (size_t)i * 8388608ULL);
    uint8_t H0[64];
    {
        B2 s; b2_init(&s, 64);
        uint32_t v;
        v = 1;      b2_update(&s, (uint8_t *)&v, 4);
        v = 32;     b2_update(&s, (uint8_t *)&v, 4);
        v = 8192;   b2_update(&s, (uint8_t *)&v, 4);
        v = 1;      b2_update(&s, (uint8_t *)&v, 4);
        v = 0x13;   b2_update(&s, (uint8_t *)&v, 4);
        v = 2;      b2_update(&s, (uint8_t *)&v, 4);
        v = 120;    b2_update(&s, (uint8_t *)&v, 4);
        b2_update(&s, headers + (size_t)i * 120, 120);
        v = 16;     b2_update(&s, (uint8_t *)&v, 4);
        b2_update(&s, (const uint8_t *)"Argonmine/pow/v1", 16);
        v = 0;      b2_update(&s, (uint8_t *)&v, 4);
        v = 0;      b2_update(&s, (uint8_t *)&v, 4);
        b2_final(&s, H0, 64);
    }
    if (h0out) memcpy(h0out + (size_t)i * 64, H0, 64);
    uint8_t seed[72], blk[1024];
    memcpy(seed, H0, 64);
    for (int idx = 0; idx < 2; idx++) {
        seed[64] = (uint8_t)idx; seed[65] = 0; seed[66] = 0; seed[67] = 0;
        seed[68] = 0; seed[69] = 0; seed[70] = 0; seed[71] = 0;
        blake2b_long(blk, 1024, seed, 72);
        uint64_t *dst = B + idx * 128;
        for (int j = 0; j < 128; j++) { uint64_t x; memcpy(&x, blk + j * 8, 8); dst[j] = x; }
    }
}

// ----------------------------------------------------------------- fill kernel
__global__ void fill_kernel(uint64_t *mem, int hashes, int dbg) {
    int hid = blockIdx.x;
    if (hid >= hashes) return;
    uint64_t *B = mem + (size_t)hid * 1048576ULL;   // 8192 blocks x 128 qwords
    __shared__ uint64_t S[128], AD[128], IN[128], Z[128];
    const int t = threadIdx.x;

    Z[t] = 0;
    IN[t] = 0;
    __syncthreads();
    if (t == 0) { IN[0] = 0; IN[1] = 0; IN[2] = 0; IN[3] = 8192; IN[4] = 1; IN[5] = 2; IN[6] = 0; }
    __syncthreads();

    for (int slice = 0; slice < 4; slice++) {
        const int indie = (slice < 2);
        const int start = (slice == 0) ? 2 : 0;
        if (t == 0) { IN[2] = (uint64_t)slice; IN[6] = 1; }
        __syncthreads();
        if (indie) {
            fill_block(Z, IN, AD, S, t);          // address block generation, step 1
            fill_block(Z, AD, AD, S, t);          // step 2 (Argon2 does it twice)
        }
        for (int i = start; i < 2048; i++) {
            uint32_t abs_idx = (uint32_t)slice * 2048u + (uint32_t)i;
            uint32_t pr;
            if (indie) {
                if (i != start && (i & 127) == 0) {
                    if (t == 0) IN[6] += 1;
                    __syncthreads();
                    fill_block(Z, IN, AD, S, t);
                    fill_block(Z, AD, AD, S, t);
                }
                pr = (uint32_t)AD[i & 127];
            } else {
                pr = (uint32_t)B[(size_t)(abs_idx - 1) * 128];
            }
            uint32_t ref = index_alpha((uint32_t)slice, (uint32_t)i, pr);
            if (dbg && t == 0 && slice == 0 && i < 6) printf("[dbg %d %u %u] ", i, pr, ref);
            fill_block(B + (size_t)(abs_idx - 1) * 128, B + (size_t)ref * 128, B + (size_t)abs_idx * 128, S, t);
        }
    }
}

// ------------------------------------------------------------- finalize kernel
__global__ void finalize_kernel(const uint8_t *mem, const uint8_t *target_le,
                                uint64_t base_nonce, int hashes, unsigned long long *out, uint8_t *tag0) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= hashes) return;
    const uint64_t *B = (const uint64_t *)(mem + (size_t)i * 8388608ULL);
    uint8_t blk[1024];
    memcpy(blk, B + 8191 * 128, 1024);
    uint8_t tag[32];
    blake2b_long(tag, 32, blk, 1024);
    if (i == 0 && tag0) memcpy(tag0, tag, 32);
    if (!target_le) return;
    int le = 1;
    for (int b = 31; b >= 0; b--) {
        if (tag[b] < target_le[b]) { le = 1; break; }
        if (tag[b] > target_le[b]) { le = 0; break; }
    }
    if (le) atomicMin(out, (unsigned long long)(base_nonce + (uint64_t)i));
}

// ----------------------------------------------------------------------- host
static int hex2bin(const char *h, uint8_t *out, int n) {
    for (int i = 0; i < n; i++) {
        unsigned v; if (sscanf(h + 2 * i, "%2x", &v) != 1) return -1; out[i] = (uint8_t)v;
    }
    return 0;
}

int main(int argc, char **argv) {
    const char *prefix_hex = NULL, *target_hex = NULL, *header_hex = NULL;
    int batch = 1024, seconds = 15, dump = 0, seconds_given = 0;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--prefix") && i + 1 < argc) prefix_hex = argv[++i];
        else if (!strcmp(argv[i], "--target") && i + 1 < argc) target_hex = argv[++i];
        else if (!strcmp(argv[i], "--header") && i + 1 < argc) { header_hex = argv[++i]; dump = 1; }
        else if (!strcmp(argv[i], "--batch") && i + 1 < argc) batch = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--seconds") && i + 1 < argc) { seconds = atoi(argv[++i]); seconds_given = 1; }
        else if (!strcmp(argv[i], "--dump")) dump = 1;
    }
    int want_jobfile = 0;
    for (int i = 1; i < argc; i++) if (!strcmp(argv[i], "--jobfile")) want_jobfile = 1;
    static char dummy_prefix[225];
    if (want_jobfile && !prefix_hex) { memset(dummy_prefix, '0', 224); prefix_hex = dummy_prefix; }
    if (want_jobfile && !target_hex) target_hex = "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff";
    if (want_jobfile && !seconds_given) seconds = 0;   // daemon: run until killed
    if (!prefix_hex && header_hex) prefix_hex = header_hex;   // 120-byte header: nonce is already inside
    if (!prefix_hex || (!target_hex && !dump)) {
        printf("usage: %s --prefix <224 hex> --target <be hex> [--batch N] [--seconds S] [--dump]\n", argv[0]);
        printf("       %s --header <240 hex> --dump        (hash one full header, print tag)\n", argv[0]);
        return 1;
    }

    uint8_t prefix[112], target_be[32], target_le[32];
    int full_header = (strlen(prefix_hex) == 240);
    if (full_header) {
        uint8_t hdr[120];
        if (hex2bin(prefix_hex, hdr, 120)) { printf("bad hex\n"); return 1; }
        memcpy(prefix, hdr, 112);
    } else if (strlen(prefix_hex) == 224) {
        if (hex2bin(prefix_hex, prefix, 112)) { printf("bad hex\n"); return 1; }
    } else { printf("prefix must be 224 or 240 hex chars, got %zu\n", strlen(prefix_hex)); return 1; }

    memset(target_be, 0, 32);
    if (target_hex) {
        uint32_t tl = (uint32_t)strlen(target_hex) / 2;
        if (tl > 32 || hex2bin(target_hex, target_be + (32 - tl), tl)) { printf("bad target\n"); return 1; }
    }
    for (int i = 0; i < 32; i++) target_le[i] = target_be[31 - i];

    size_t memsz = (size_t)batch * 8388608ULL;
    uint8_t *d_mem = NULL, *d_hdr = NULL, *d_target = NULL, *d_tag0 = NULL, *d_h0 = NULL;
    unsigned long long *d_hit = NULL;
    CUDA_CHECK(cudaMalloc(&d_mem, memsz));
    CUDA_CHECK(cudaMalloc(&d_hdr, (size_t)batch * 120));
    CUDA_CHECK(cudaMalloc(&d_target, 32));
    CUDA_CHECK(cudaMalloc(&d_tag0, 32));
    CUDA_CHECK(cudaMalloc(&d_h0, 64));
    CUDA_CHECK(cudaMalloc(&d_hit, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemcpy(d_target, target_le, 32, cudaMemcpyHostToDevice));

    uint8_t *h_hdr = (uint8_t *)malloc((size_t)batch * 120);
    uint64_t nonce_base = ((uint64_t)time(NULL) << 24);

    if (dump) {
        uint8_t hdr[120];
        memcpy(hdr, prefix, 112);
        if (full_header) { uint8_t tmp[120]; hex2bin(prefix_hex, tmp, 120); memcpy(hdr, tmp, 120); }
        else { uint64_t n = 0; memcpy(hdr + 112, &n, 8); }
        memcpy(h_hdr, hdr, 120);
        CUDA_CHECK(cudaMemcpy(d_hdr, h_hdr, 120, cudaMemcpyHostToDevice));
        prepare_kernel<<<1, 1>>>(d_hdr, d_mem, 1, d_h0);
        fill_kernel<<<1, 128>>>((uint64_t *)d_mem, 1, 1);
        CUDA_CHECK(cudaGetLastError());
        finalize_kernel<<<1, 1>>>(d_mem, NULL, 0, 1, d_hit, d_tag0);
        uint8_t tag[32];
        CUDA_CHECK(cudaMemcpy(tag, d_tag0, 32, cudaMemcpyDeviceToHost));
        printf("tag: ");
        for (int i = 0; i < 32; i++) printf("%02x", tag[i]);
        printf("\n");
        uint8_t h0[64], b0[64], b1[64], bt[64];
        uint8_t *d_bt = NULL;
        CUDA_CHECK(cudaMemcpy(h0, d_h0, 64, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(b0, d_mem, 64, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(b1, d_mem + 1024, 64, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMalloc(&d_bt, 64));
        b2test_kernel<<<1, 1>>>(d_bt);
        CUDA_CHECK(cudaMemcpy(bt, d_bt, 64, cudaMemcpyDeviceToHost));
        printf("H0 : "); for (int i = 0; i < 64; i++) printf("%02x", h0[i]); printf("\n");
        printf("B0 : "); for (int i = 0; i < 64; i++) printf("%02x", b0[i]); printf("\n");
        printf("B1 : "); for (int i = 0; i < 64; i++) printf("%02x", b1[i]); printf("\n");
        uint8_t b2x[64];
        CUDA_CHECK(cudaMemcpy(b2x, d_mem + 2 * 1024, 64, cudaMemcpyDeviceToHost));
        printf("B2 : "); for (int i = 0; i < 64; i++) printf("%02x", b2x[i]); printf("\n");
                        int cps[4] = {2047, 2048, 4095, 4096};
        for (int ci = 0; ci < 4; ci++) {
            uint8_t tmp[32];
            CUDA_CHECK(cudaMemcpy(tmp, d_mem + (size_t)cps[ci]*1024, 32, cudaMemcpyDeviceToHost));
            printf("B%d: ", cps[ci]); for (int k = 0; k < 32; k++) printf("%02x", tmp[k]); putchar(10);
        }
        printf("b2abc: "); for (int i = 0; i < 64; i++) printf("%02x", bt[i]); printf("\n");
        return 0;
    }

        // ---- long-running mode: job comes from a file, refreshed by the driver
    const char *jobfile = NULL;
    for (int i = 1; i < argc; i++) if (!strcmp(argv[i], "--jobfile") && i + 1 < argc) jobfile = argv[++i];

    char job_id[64] = "none";
    long long done = 0;
    time_t t0 = time(NULL);
    time_t last_mtime = 0;
    int have_target = 0;
    uint8_t jprefix[112];

    printf("argon_gpu: batch %d (%.1f GB VRAM), jobfile=%s\n", batch, memsz / 1073741824.0, jobfile ? jobfile : "(none)");
    fflush(stdout);

    for (;;) {
        if (jobfile) {
            struct stat st;
            if (stat(jobfile, &st) == 0 && st.st_mtime != last_mtime) {
                FILE *f = fopen(jobfile, "r");
                if (f) {
                    char l1[128] = {0}, l2[512] = {0}, l3[128] = {0};
                    if (fgets(l1, sizeof(l1), f) && fgets(l2, sizeof(l2), f) && fgets(l3, sizeof(l3), f)) {
                        l1[strcspn(l1, "\r\n")] = 0;
                        l2[strcspn(l2, "\r\n")] = 0;
                        l3[strcspn(l3, "\r\n")] = 0;
                        uint8_t pfx[112], tbe[32], tle[32];
                        uint32_t tl = (uint32_t)strlen(l3) / 2;
                        memset(tbe, 0, 32);
                        if (strlen(l2) == 224 && tl > 0 && tl <= 32 &&
                            hex2bin(l2, pfx, 112) == 0 && hex2bin(l3, tbe + (32 - tl), tl) == 0) {
                            for (int k = 0; k < 32; k++) tle[k] = tbe[31 - k];
                            snprintf(job_id, sizeof(job_id), "%s", l1);
                            memcpy(jprefix, pfx, 112);
                            memcpy((void *)d_target_host, tle, 32);
                            have_target = 1;
                            last_mtime = st.st_mtime;
                            nonce_base ^= ((uint64_t)time(NULL) << 16);
                            CUDA_CHECK(cudaMemcpy(d_target, tle, 32, cudaMemcpyHostToDevice));
                            printf("[job] %s\n", job_id);
                            fflush(stdout);
                        }
                    }
                    fclose(f);
                }
            }
        }
        if (!have_target) { usleep(200000); continue; }
        if (seconds > 0 && (int)(time(NULL) - t0) >= seconds) break;

        for (int i = 0; i < batch; i++) {
            memcpy(h_hdr + (size_t)i * 120, jprefix, 112);
            uint64_t n = nonce_base + (uint64_t)done + (uint64_t)i;
            memcpy(h_hdr + (size_t)i * 120 + 112, &n, 8);
        }
        CUDA_CHECK(cudaMemcpy(d_hdr, h_hdr, (size_t)batch * 120, cudaMemcpyHostToDevice));
        double t_start = now_ms();
        prepare_kernel<<<(batch + 127) / 128, 128>>>(d_hdr, d_mem, batch, NULL);
        fill_kernel<<<batch, 128>>>((uint64_t *)d_mem, batch, 0);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemset(d_hit, 0xff, sizeof(unsigned long long)));
        finalize_kernel<<<(batch + 127) / 128, 128>>>((const uint8_t *)d_mem, d_target, nonce_base + (uint64_t)done, batch, d_hit, NULL);
        unsigned long long hit = 0;
        CUDA_CHECK(cudaMemcpy(&hit, d_hit, 8, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaDeviceSynchronize());
        double ms = now_ms() - t_start;
        done += batch;
        printf("  %d hashes in %7.1f ms -> %8.1f H/s   (total %lld)\n", batch, ms, batch * 1000.0 / ms, done);
        if (hit != ~0ULL) { printf("HIT %s %llu\n", job_id, hit); fflush(stdout); }
        fflush(stdout);
    }
    return 0;
}