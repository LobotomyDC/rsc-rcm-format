// rsc2rcw.c - RuneCast wall/roof converter (maps63 + land63 -> RCW1)
//
// First-pass static wall/roof baker for RuneCast.
// Outputs one wall RCW1 and one roof RCW1 per original 48x48 map section:
//   <out_dir>/walls/mPXXYY.rcm
//   <out_dir>/roofs/mPXXYY.rcm
//
// Build (Linux):
//   gcc -std=c99 -O2 -Wall -Wextra rsc2rcw.c -lbz2 -lm -o rsc2rcw
//
// Usage:
//   ./rsc2rcw land63.jag maps63.jag config85.jag out_dir [options]
//
// Options:
//   --land-mem <land63.mem>     optional members landscape pack
//   --maps-mem <maps63.mem>     optional members map pack
//   --plane <0..3>              only bake one plane
//   --walls-only                do not emit roof files
//   --roofs-only                do not emit wall files
//   --thick-walls               approximate RuneCast's thick wall mode
//   --empty-files               write valid empty RCW1 files for empty chunks
//
// Notes:
// - RCW1 intentionally uses the same float-position/Q15-UV vertex layout as RCM1.
// - Coordinates are chunk-local and scaled by /100, matching RCL1 placement.
// - Wall geometry is neighbor-height aware at x/y == 48 so section edges align.
// - Roof baking is included, but should be treated as a validation prototype. Roofs
//   need neighbor roof/wall samples, so this file loads a 3x3 map neighborhood.
// - Wall RCWs also carry an RWM1 side metadata block with baked base adjacency
//   and packed tile directions. This is intentionally enough for RCM/RCW runtime
//   traversal without keeping maps63.jag/mem resident, and deliberately excludes
//   .loc scenery placement data.
// - maps63.dat uses raw wall/diagonal wall streams when VERSION_MAPS > 53, matching
//   RuneCast world_load_section_files(). Older RLE wall decoding is retained only as
//   a fallback for non-v63 inputs.

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <errno.h>
#include <math.h>

#include <sys/stat.h>
#include <sys/types.h>

#include <bzlib.h>

#define REGION_SIZE      48
#define TILE_COUNT       (REGION_SIZE * REGION_SIZE)
#define TILE_SIZE        128
#define VERTEX_SCALE_F   100.0f
#define UV_Q15_DIVISOR   32767
#define RCM_FILE_ALIGN   32u

#define PLANE_HEIGHT     80000
#define ROOF_SLOPE       16

#define COLOUR_TRANSPARENT ((int16_t)0x7fff)
#define JAGEX_TRANSPARENT  (12345678)

#define RCM_SUBMESH_TEX_NONE 0xFFFFu
#define RCM_SUBF_ALPHA      (1u << 0)
#define DIAG_NW_SE_OFFSET   12000u

#define NPC_SPRITE_COUNT 12

#define RCW1_FLAG_HAS_RWM       (1u << 0)
#define RCW1_RWM_RESERVED_MAGIC 0x314d5752u /* 'RWM1' little-endian */

#define RWM1_HEADER_SIZE 32u
#define RWM1_DIR_BYTES   (TILE_COUNT / 2)
#define RWM1_BLOCK_SIZE  (RWM1_HEADER_SIZE + TILE_COUNT + RWM1_DIR_BYTES)

#define RWM1_FLAGS_HAS_ADJACENCY   (1u << 0)
#define RWM1_FLAGS_HAS_DIRECTIONS  (1u << 1)
#define RWM1_FLAGS_HAS_TILE_CONFIG (1u << 2)
#define RWM1_FLAGS_HAS_DAT         (1u << 3)

enum {
    RCW_FLOOR_TILE_TYPE = 2
};

#if defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 201112L)
#define STATIC_ASSERT(cond, msg) _Static_assert(cond, msg)
#else
#define STATIC_ASSERT(cond, msg) typedef char static_assertion_##msg[(cond) ? 1 : -1]
#endif

/* -------------------- Texture atlas UV mapping (matches rsc2rcm) -------------------- */

typedef struct gl_atlas_position {
    float left_u, right_u;
    float top_v, bottom_v;
} gl_atlas_position;

// Copied from your src/gl/textures/model_textures.c (repomix).
// This MUST match your in-game atlas layout for model_textures.
static const gl_atlas_position gl_texture_atlas_positions[] = {
    {0.000000f, 0.125000f, 0.375000f, 0.500000f},
    {0.187500f, 0.250000f, 0.687500f, 0.750000f},
    {0.500000f, 0.625000f, 0.250000f, 0.375000f},
    {0.937500f, 1.000000f, 0.125000f, 0.187500f},
    {0.250000f, 0.375000f, 0.375000f, 0.500000f},
    {0.750000f, 0.875000f, 0.375000f, 0.500000f},
    {0.375000f, 0.500000f, 0.000000f, 0.125000f},
    {0.375000f, 0.500000f, 0.125000f, 0.250000f},
    {0.750000f, 0.875000f, 0.000000f, 0.125000f},
    {0.187500f, 0.250000f, 0.562500f, 0.625000f},
    {0.125000f, 0.187500f, 0.812500f, 0.875000f},
    {0.187500f, 0.250000f, 0.750000f, 0.812500f},
    {0.125000f, 0.250000f, 0.250000f, 0.375000f},
    {0.125000f, 0.187500f, 0.500000f, 0.562500f},
    {0.937500f, 1.000000f, 0.187500f, 0.250000f},
    {0.062500f, 0.125000f, 0.937500f, 1.000000f},
    {0.250000f, 0.375000f, 0.125000f, 0.250000f},
    {0.625000f, 0.687500f, 0.125000f, 0.250000f},
    {0.687500f, 0.812500f, 0.125000f, 0.250000f},
    {0.125000f, 0.250000f, 0.000000f, 0.125000f},
    {0.375000f, 0.500000f, 0.250000f, 0.375000f},
    {0.000000f, 0.125000f, 0.500000f, 0.625000f},
    {0.875000f, 1.000000f, 0.375000f, 0.500000f},
    {0.625000f, 0.750000f, 0.000000f, 0.125000f},
    {0.125000f, 0.187500f, 0.562500f, 0.625000f},
    {0.125000f, 0.187500f, 0.625000f, 0.687500f},
    {0.062500f, 0.125000f, 0.875000f, 0.937500f},
    {0.000000f, 0.125000f, 0.625000f, 0.750000f},
    {0.000000f, 0.125000f, 0.250000f, 0.375000f},
    {0.187500f, 0.250000f, 0.812500f, 0.875000f},
    {0.125000f, 0.187500f, 0.750000f, 0.812500f},
    {0.187500f, 0.250000f, 0.500000f, 0.562500f},
    {0.500000f, 0.625000f, 0.000000f, 0.125000f},
    {0.750000f, 0.875000f, 0.250000f, 0.375000f},
    {0.500000f, 0.625000f, 0.375000f, 0.500000f},
    {0.000000f, 0.062500f, 0.937500f, 1.000000f},
    {0.875000f, 1.000000f, 0.250000f, 0.375000f},
    {0.187500f, 0.250000f, 0.625000f, 0.687500f},
    {0.000000f, 0.062500f, 0.875000f, 0.937500f},
    {0.812500f, 0.937500f, 0.125000f, 0.250000f},
    {0.375000f, 0.500000f, 0.375000f, 0.500000f},
    {0.125000f, 0.250000f, 0.375000f, 0.500000f},
    {0.250000f, 0.375000f, 0.000000f, 0.125000f},
    {0.250000f, 0.375000f, 0.250000f, 0.375000f},
    {0.000000f, 0.125000f, 0.000000f, 0.125000f},
    {0.500000f, 0.625000f, 0.125000f, 0.250000f},
    {0.125000f, 0.187500f, 0.875000f, 0.937500f},
    {0.000000f, 0.125000f, 0.125000f, 0.250000f},
    {0.125000f, 0.187500f, 0.937500f, 1.000000f},
    {0.000000f, 0.125000f, 0.750000f, 0.875000f},
    {0.875000f, 1.000000f, 0.000000f, 0.125000f},
    {0.625000f, 0.750000f, 0.250000f, 0.375000f},
    {0.625000f, 0.750000f, 0.375000f, 0.500000f},
    {0.125000f, 0.250000f, 0.125000f, 0.250000f},
    {0.125000f, 0.187500f, 0.687500f, 0.750000f},
};

static void gl_offset_texture_uvs_atlas(gl_atlas_position tp, float *u, float *v) {
    float w = fabsf(tp.left_u - tp.right_u);
    float h = fabsf(tp.top_v  - tp.bottom_v);

    // Fountain special-case (matches your code)
    if ((w * 1024.0f) == 64.0f && (h * 1024.0f) == 128.0f) {
        h /= 2.0f;
    }

    *u *= w;
    *v *= h;
    *u += tp.left_u;
    *v += tp.top_v; // RENDER_GL path
}

static int16_t pack_uv_q15(float uv) {
    if (uv < 0.0f) uv = 0.0f;
    if (uv > 1.0f) uv = 1.0f;
    int v = (int)lrintf(uv * (float)UV_Q15_DIVISOR);
    if (v < 0) v = 0;
    if (v > UV_Q15_DIVISOR) v = UV_Q15_DIVISOR;
    return (int16_t)v;
}


static int texture_id_has_atlas(uint16_t tex_id) {
    return tex_id < (uint16_t)(sizeof(gl_texture_atlas_positions) / sizeof(gl_texture_atlas_positions[0]));
}

static void atlas_uv_to_q15(uint16_t tex_id, float u01, float v01, int16_t *out_u, int16_t *out_v) {
    float u = u01;
    float v = v01;

    if (texture_id_has_atlas(tex_id)) {
        gl_offset_texture_uvs_atlas(gl_texture_atlas_positions[tex_id], &u, &v);
    }

    if (out_u) *out_u = pack_uv_q15(u);
    if (out_v) *out_v = pack_uv_q15(v);
}

static void die(const char *msg) {
    fprintf(stderr, "fatal: %s\n", msg);
    exit(1);
}

static void *xmalloc(size_t n) {
    void *p = malloc(n ? n : 1);
    if (!p) die("out of memory");
    return p;
}

static void *xcalloc(size_t n, size_t s) {
    void *p = calloc(n ? n : 1, s ? s : 1);
    if (!p) die("out of memory");
    return p;
}

static void *xrealloc(void *p, size_t n) {
    void *q = realloc(p, n ? n : 1);
    if (!q) die("out of memory");
    return q;
}

static size_t align_up_sz(size_t v, size_t a) {
    return (v + (a - 1u)) & ~(a - 1u);
}

static uint32_t read_u24be(const uint8_t *b, size_t off, size_t len) {
    if (off + 3 > len) die("read_u24be out of range");
    return ((uint32_t)b[off] << 16) | ((uint32_t)b[off + 1] << 8) | (uint32_t)b[off + 2];
}

static uint32_t read_u32be(const uint8_t *b, size_t off, size_t len) {
    if (off + 4 > len) die("read_u32be out of range");
    return ((uint32_t)b[off] << 24) | ((uint32_t)b[off + 1] << 16) |
           ((uint32_t)b[off + 2] << 8) | (uint32_t)b[off + 3];
}

static void write_u16le(uint8_t *b, size_t off, uint16_t v) {
    b[off + 0] = (uint8_t)(v & 0xffu);
    b[off + 1] = (uint8_t)((v >> 8) & 0xffu);
}

static void write_u32le(uint8_t *b, size_t off, uint32_t v) {
    b[off + 0] = (uint8_t)(v & 0xffu);
    b[off + 1] = (uint8_t)((v >> 8) & 0xffu);
    b[off + 2] = (uint8_t)((v >> 16) & 0xffu);
    b[off + 3] = (uint8_t)((v >> 24) & 0xffu);
}

static int ensure_dir(const char *path) {
#ifdef _WIN32
    (void)path;
    return 0;
#else
    struct stat st;
    if (stat(path, &st) == 0) return S_ISDIR(st.st_mode) ? 0 : -1;
    return mkdir(path, 0755);
#endif
}

static uint8_t *read_entire_file(const char *path, size_t *out_len) {
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "open %s: %s\n", path, strerror(errno));
        return NULL;
    }
    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return NULL; }
    long sz = ftell(f);
    if (sz < 0) { fclose(f); return NULL; }
    if (fseek(f, 0, SEEK_SET) != 0) { fclose(f); return NULL; }

    uint8_t *buf = (uint8_t *)xmalloc((size_t)sz);
    if (fread(buf, 1, (size_t)sz, f) != (size_t)sz) {
        fclose(f);
        free(buf);
        return NULL;
    }
    fclose(f);
    if (out_len) *out_len = (size_t)sz;
    return buf;
}

static int write_entire_file(const char *path, const void *data, size_t len) {
    FILE *f = fopen(path, "wb");
    if (!f) {
        fprintf(stderr, "write %s: %s\n", path, strerror(errno));
        return 0;
    }
    if (fwrite(data, 1, len, f) != len) {
        fprintf(stderr, "write %s: short write\n", path);
        fclose(f);
        return 0;
    }
    if (fclose(f) != 0) {
        fprintf(stderr, "write %s: fclose failed: %s\n", path, strerror(errno));
        return 0;
    }
    return 1;
}

/* Same hash used by JAG directory lookup in RuneCast utility.c. */
static uint32_t rsc_hash_name(const char *name) {
    uint32_t hash = 0;
    for (const unsigned char *p = (const unsigned char *)name; *p; ++p) {
        unsigned char c = *p;
        if (c >= 'a' && c <= 'z') c = (unsigned char)(c - 32);
        hash = hash * 61u + (uint32_t)c - 32u;
    }
    return hash;
}

/* -------------------- JAG handling -------------------- */

typedef struct jag_entry {
    uint32_t hash;
    uint32_t unpacked_len;
    uint32_t packed_len;
    const uint8_t *payload;
} jag_entry_t;

typedef struct jag_archive {
    uint8_t *blob;
    size_t blob_len;
    jag_entry_t *entries;
    uint16_t count;
} jag_archive_t;

static uint8_t *bz2_decompress_nohdr(const uint8_t *src, uint32_t packed_len, uint32_t unpacked_len) {
    uint8_t *dst = (uint8_t *)xmalloc(unpacked_len);
    uint8_t *tmp = (uint8_t *)xmalloc((size_t)packed_len + 4u);
    memcpy(tmp, "BZh1", 4);
    memcpy(tmp + 4, src, packed_len);

    unsigned int dlen = (unsigned int)unpacked_len;
    int rc = BZ2_bzBuffToBuffDecompress((char *)dst, &dlen,
                                        (char *)tmp, (unsigned int)(packed_len + 4u), 0, 0);
    free(tmp);
    if (rc != BZ_OK || dlen != (unsigned int)unpacked_len) {
        free(dst);
        return NULL;
    }
    return dst;
}

static jag_archive_t jag_load_required(const char *path) {
    jag_archive_t a;
    memset(&a, 0, sizeof(a));

    size_t file_len = 0;
    uint8_t *file = read_entire_file(path, &file_len);
    if (!file) die("failed to read archive");
    if (file_len < 6) die("archive too small");

    uint32_t unpacked = read_u24be(file, 0, file_len);
    uint32_t packed   = read_u24be(file, 3, file_len);
    if (6u + packed > file_len) die("bad archive header");

    if (packed != unpacked) {
        uint8_t *blob = bz2_decompress_nohdr(file + 6, packed, unpacked);
        free(file);
        if (!blob) die("outer bzip decompress failed");
        a.blob = blob;
        a.blob_len = unpacked;
    } else {
        a.blob = file;
        a.blob_len = file_len;
    }

    if (a.blob_len < 2) die("archive missing directory");
    a.count = (uint16_t)((a.blob[0] << 8) | a.blob[1]);
    const size_t dir_len = 2u + (size_t)a.count * 10u;
    if (dir_len > a.blob_len) die("directory overruns blob");

    a.entries = (jag_entry_t *)xcalloc(a.count, sizeof(jag_entry_t));
    size_t data_off = dir_len;
    for (uint16_t i = 0; i < a.count; ++i) {
        const size_t eoff = 2u + (size_t)i * 10u;
        a.entries[i].hash = read_u32be(a.blob, eoff, a.blob_len);
        a.entries[i].unpacked_len = read_u24be(a.blob, eoff + 4, a.blob_len);
        a.entries[i].packed_len = read_u24be(a.blob, eoff + 7, a.blob_len);
        if (data_off + a.entries[i].packed_len > a.blob_len) die("entry payload overruns blob");
        a.entries[i].payload = a.blob + data_off;
        data_off += a.entries[i].packed_len;
    }
    return a;
}

static jag_archive_t jag_load_optional(const char *path) {
    jag_archive_t a;
    memset(&a, 0, sizeof(a));
    if (!path) return a;
    return jag_load_required(path);
}

static void jag_free(jag_archive_t *a) {
    if (!a) return;
    free(a->entries);
    free(a->blob);
    memset(a, 0, sizeof(*a));
}

static const jag_entry_t *jag_find_by_hash(const jag_archive_t *a, uint32_t hash) {
    if (!a || !a->blob) return NULL;
    for (uint16_t i = 0; i < a->count; ++i) {
        if (a->entries[i].hash == hash) return &a->entries[i];
    }
    return NULL;
}

static const jag_entry_t *jag_find_by_hash_either(const jag_archive_t *a, const jag_archive_t *b, uint32_t h) {
    const jag_entry_t *e = jag_find_by_hash(a, h);
    if (!e) e = jag_find_by_hash(b, h);
    return e;
}

static uint8_t *jag_extract_entry(const jag_entry_t *e, size_t *out_len) {
    if (!e || e->packed_len == 0 || e->unpacked_len == 0) return NULL;
    uint8_t *buf = NULL;
    if (e->packed_len == e->unpacked_len) {
        buf = (uint8_t *)xmalloc(e->unpacked_len);
        memcpy(buf, e->payload, e->unpacked_len);
    } else {
        buf = bz2_decompress_nohdr(e->payload, e->packed_len, e->unpacked_len);
        if (!buf) return NULL;
    }
    if (out_len) *out_len = (size_t)e->unpacked_len;
    return buf;
}

/* -------------------- Game data config -------------------- */

typedef struct gd_reader {
    const uint8_t *str;
    size_t str_len, str_off;
    const uint8_t *in;
    size_t in_len, in_off;
} gd_reader_t;

static int gd_u8(gd_reader_t *r) {
    if (r->in_off + 1 > r->in_len) die("gamedata: integer.dat overrun (u8)");
    return (int)r->in[r->in_off++];
}

static int gd_u16(gd_reader_t *r) {
    if (r->in_off + 2 > r->in_len) die("gamedata: integer.dat overrun (u16)");
    int v = ((int)r->in[r->in_off] << 8) | (int)r->in[r->in_off + 1];
    r->in_off += 2;
    return v;
}

static int gd_u32_fill(gd_reader_t *r) {
    if (r->in_off + 4 > r->in_len) die("gamedata: integer.dat overrun (u32)");
    int v = ((int)r->in[r->in_off] << 24) |
            ((int)r->in[r->in_off + 1] << 16) |
            ((int)r->in[r->in_off + 2] << 8) |
            ((int)r->in[r->in_off + 3]);
    r->in_off += 4;
    if (v > 99999999) v = 99999999 - v;
    if (v == JAGEX_TRANSPARENT) v = COLOUR_TRANSPARENT;
    return v;
}

static void gd_skip_string(gd_reader_t *r) {
    while (r->str_off < r->str_len && r->str[r->str_off] != 0) r->str_off++;
    if (r->str_off >= r->str_len) die("gamedata: string.dat overrun");
    r->str_off++;
}

typedef struct wallcfg {
    int height;
    int front;
    int back;
    uint8_t blocking;
    uint8_t interactive;
} wallcfg_t;

typedef struct roofcfg {
    int height;
    int fill;
} roofcfg_t;

typedef struct tilecfg {
    uint8_t type;
    uint8_t blocking;
} tilecfg_t;

typedef struct config_db {
    wallcfg_t *walls;
    int wall_count;
    roofcfg_t *roofs;
    int roof_count;
    tilecfg_t *tiles;
    int tile_count;
} config_db_t;

static void config_free(config_db_t *cfg) {
    if (!cfg) return;
    free(cfg->walls);
    free(cfg->roofs);
    free(cfg->tiles);
    memset(cfg, 0, sizeof(*cfg));
}

static int config_load(const char *config_jag_path, config_db_t *out) {
    memset(out, 0, sizeof(*out));

    jag_archive_t cfg = jag_load_required(config_jag_path);
    const jag_entry_t *e_str = jag_find_by_hash(&cfg, rsc_hash_name("string.dat"));
    const jag_entry_t *e_int = jag_find_by_hash(&cfg, rsc_hash_name("integer.dat"));
    size_t slen = 0, ilen = 0;
    uint8_t *str_dat = jag_extract_entry(e_str, &slen);
    uint8_t *int_dat = jag_extract_entry(e_int, &ilen);
    jag_free(&cfg);

    if (!str_dat || !int_dat) {
        free(str_dat);
        free(int_dat);
        return 0;
    }

    gd_reader_t r;
    memset(&r, 0, sizeof(r));
    r.str = str_dat; r.str_len = slen;
    r.in = int_dat; r.in_len = ilen;

    int item_count = gd_u16(&r);
    for (int i = 0; i < item_count; i++) gd_skip_string(&r);
    for (int i = 0; i < item_count; i++) gd_skip_string(&r);
    for (int i = 0; i < item_count; i++) gd_skip_string(&r);
    for (int i = 0; i < item_count; i++) (void)gd_u16(&r);
    for (int i = 0; i < item_count; i++) (void)gd_u32_fill(&r);
    for (int i = 0; i < item_count; i++) (void)gd_u8(&r);
    for (int i = 0; i < item_count; i++) (void)gd_u8(&r);
    for (int i = 0; i < item_count; i++) (void)gd_u16(&r);
    for (int i = 0; i < item_count; i++) (void)gd_u32_fill(&r);
    for (int i = 0; i < item_count; i++) (void)gd_u8(&r);
    for (int i = 0; i < item_count; i++) (void)gd_u8(&r);

    int npc_count = gd_u16(&r);
    for (int i = 0; i < npc_count; i++) gd_skip_string(&r);
    for (int i = 0; i < npc_count; i++) gd_skip_string(&r);
    for (int i = 0; i < npc_count; i++) (void)gd_u8(&r);
    for (int i = 0; i < npc_count; i++) (void)gd_u8(&r);
    for (int i = 0; i < npc_count; i++) (void)gd_u8(&r);
    for (int i = 0; i < npc_count; i++) (void)gd_u8(&r);
    for (int i = 0; i < npc_count; i++) (void)gd_u8(&r);
    for (int i = 0; i < npc_count; i++) for (int j = 0; j < NPC_SPRITE_COUNT; j++) (void)gd_u8(&r);
    for (int i = 0; i < npc_count; i++) (void)gd_u32_fill(&r);
    for (int i = 0; i < npc_count; i++) (void)gd_u32_fill(&r);
    for (int i = 0; i < npc_count; i++) (void)gd_u32_fill(&r);
    for (int i = 0; i < npc_count; i++) (void)gd_u32_fill(&r);
    for (int i = 0; i < npc_count; i++) (void)gd_u16(&r);
    for (int i = 0; i < npc_count; i++) (void)gd_u16(&r);
    for (int i = 0; i < npc_count; i++) (void)gd_u8(&r);
    for (int i = 0; i < npc_count; i++) (void)gd_u8(&r);
    for (int i = 0; i < npc_count; i++) (void)gd_u8(&r);
    for (int i = 0; i < npc_count; i++) gd_skip_string(&r);

    int texture_count = gd_u16(&r);
    for (int i = 0; i < texture_count; i++) gd_skip_string(&r);
    for (int i = 0; i < texture_count; i++) gd_skip_string(&r);

    int anim_count = gd_u16(&r);
    for (int i = 0; i < anim_count; i++) gd_skip_string(&r);
    for (int i = 0; i < anim_count; i++) (void)gd_u32_fill(&r);
    for (int i = 0; i < anim_count; i++) (void)gd_u8(&r);
    for (int i = 0; i < anim_count; i++) (void)gd_u8(&r);
    for (int i = 0; i < anim_count; i++) (void)gd_u8(&r);
    for (int i = 0; i < anim_count; i++) (void)gd_u8(&r);

    int object_count = gd_u16(&r);
    for (int i = 0; i < object_count; i++) gd_skip_string(&r);
    for (int i = 0; i < object_count; i++) gd_skip_string(&r);
    for (int i = 0; i < object_count; i++) gd_skip_string(&r);
    for (int i = 0; i < object_count; i++) gd_skip_string(&r);
    for (int i = 0; i < object_count; i++) gd_skip_string(&r);
    for (int i = 0; i < object_count; i++) (void)gd_u8(&r);
    for (int i = 0; i < object_count; i++) (void)gd_u8(&r);
    for (int i = 0; i < object_count; i++) (void)gd_u8(&r);
    for (int i = 0; i < object_count; i++) (void)gd_u8(&r);

    int wall_count = gd_u16(&r);
    out->wall_count = wall_count;
    out->walls = (wallcfg_t *)xcalloc((size_t)wall_count, sizeof(wallcfg_t));
    for (int i = 0; i < wall_count; i++) gd_skip_string(&r);
    for (int i = 0; i < wall_count; i++) gd_skip_string(&r);
    for (int i = 0; i < wall_count; i++) gd_skip_string(&r);
    for (int i = 0; i < wall_count; i++) gd_skip_string(&r);
    for (int i = 0; i < wall_count; i++) out->walls[i].height = gd_u16(&r);
    for (int i = 0; i < wall_count; i++) out->walls[i].front = gd_u32_fill(&r);
    for (int i = 0; i < wall_count; i++) out->walls[i].back = gd_u32_fill(&r);
    for (int i = 0; i < wall_count; i++) out->walls[i].blocking = (uint8_t)gd_u8(&r);
    for (int i = 0; i < wall_count; i++) out->walls[i].interactive = (uint8_t)gd_u8(&r);

    int roof_count = gd_u16(&r);
    out->roof_count = roof_count;
    out->roofs = (roofcfg_t *)xcalloc((size_t)roof_count, sizeof(roofcfg_t));
    for (int i = 0; i < roof_count; i++) out->roofs[i].height = gd_u8(&r);
    for (int i = 0; i < roof_count; i++) out->roofs[i].fill = gd_u8(&r);

    int tile_count = gd_u16(&r);
    out->tile_count = tile_count;
    out->tiles = (tilecfg_t *)xcalloc((size_t)tile_count, sizeof(tilecfg_t));
    for (int i = 0; i < tile_count; i++) (void)gd_u32_fill(&r);
    for (int i = 0; i < tile_count; i++) out->tiles[i].type = (uint8_t)gd_u8(&r);
    for (int i = 0; i < tile_count; i++) out->tiles[i].blocking = (uint8_t)gd_u8(&r);

    free(str_dat);
    free(int_dat);
    fprintf(stderr, "info: loaded %d wall configs, %d roof configs, %d tile configs\n",
            wall_count, roof_count, tile_count);
    return 1;
}

/* -------------------- Map/height decode -------------------- */

typedef struct chunk_data {
    uint8_t h[TILE_COUNT];
    uint8_t walls_ns[TILE_COUNT];
    uint8_t walls_ew[TILE_COUNT];
    uint16_t walls_diag[TILE_COUNT];
    uint8_t roofs[TILE_COUNT];
    uint8_t tile_decoration[TILE_COUNT];
    uint8_t tile_direction[TILE_COUNT];
    int have_h;
    int have_dat;
} chunk_data_t;

static int decode_hei(const uint8_t *data, size_t len, uint8_t out_h[TILE_COUNT]) {
    if (!data || len == 0 || !out_h) return 0;
    size_t off = 0;
    int last = 0;
    for (int t = 0; t < TILE_COUNT; ) {
        if (off >= len) return 0;
        int val = (int)data[off++];
        if (val < 128) {
            out_h[t++] = (uint8_t)val;
            last = val;
        } else {
            for (int i = 0; i < val - 128 && t < TILE_COUNT; i++) out_h[t++] = (uint8_t)last;
        }
    }

    last = 64;
    for (int y = 0; y < REGION_SIZE; y++) {
        for (int x = 0; x < REGION_SIZE; x++) {
            const int idx = x * REGION_SIZE + y;
            last = (out_h[idx] + last) & 127;
            out_h[idx] = (uint8_t)(last * 2);
        }
    }
    return 1;
}

static int dat_rle0_decode_u8(const uint8_t *data, size_t len, size_t *off, uint8_t out[TILE_COUNT]) {
    for (int t = 0; t < TILE_COUNT; ) {
        if (*off >= len) return 0;
        int val = (int)data[(*off)++];
        if (val < 128) {
            if (out) out[t] = (uint8_t)val;
            t++;
        } else {
            int run = val - 128;
            for (int i = 0; i < run && t < TILE_COUNT; i++, t++) if (out) out[t] = 0;
        }
    }
    return 1;
}

static int dat_repeat_last_decode_u8(const uint8_t *data, size_t len, size_t *off, uint8_t out[TILE_COUNT]) {
    int last = 0;
    for (int t = 0; t < TILE_COUNT; ) {
        if (*off >= len) return 0;
        int val = (int)data[(*off)++];
        if (val < 128) {
            if (out) out[t] = (uint8_t)val;
            last = val;
            t++;
        } else {
            int run = val - 128;
            for (int i = 0; i < run && t < TILE_COUNT; i++, t++) if (out) out[t] = (uint8_t)last;
        }
    }
    return 1;
}

static int decode_dat_v63_try_rle(const uint8_t *data, size_t len, chunk_data_t *out) {
    size_t off = 0;
    memset(out->walls_ns, 0, TILE_COUNT);
    memset(out->walls_ew, 0, TILE_COUNT);
    memset(out->walls_diag, 0, sizeof(out->walls_diag));
    memset(out->roofs, 0, TILE_COUNT);
    memset(out->tile_decoration, 0, TILE_COUNT);
    memset(out->tile_direction, 0, TILE_COUNT);

    if (!dat_rle0_decode_u8(data, len, &off, out->walls_ns)) return 0;
    if (!dat_rle0_decode_u8(data, len, &off, out->walls_ew)) return 0;

    uint8_t tmp[TILE_COUNT];
    if (!dat_rle0_decode_u8(data, len, &off, tmp)) return 0;
    for (int i = 0; i < TILE_COUNT; i++) if (tmp[i]) out->walls_diag[i] = tmp[i];

    memset(tmp, 0, TILE_COUNT);
    if (!dat_rle0_decode_u8(data, len, &off, tmp)) return 0;
    for (int i = 0; i < TILE_COUNT; i++) if (tmp[i]) out->walls_diag[i] = (uint16_t)(tmp[i] + DIAG_NW_SE_OFFSET);

    if (!dat_rle0_decode_u8(data, len, &off, out->roofs)) return 0;

    if (!dat_repeat_last_decode_u8(data, len, &off, out->tile_decoration)) return 0;
    if (!dat_rle0_decode_u8(data, len, &off, out->tile_direction)) return 0;
    return 1;
}

static int decode_dat_v63_try_raw_walls(const uint8_t *data, size_t len, chunk_data_t *out) {
    size_t off = 0;
    if (len < 4u * (size_t)TILE_COUNT) return 0;
    memcpy(out->walls_ns, data + off, TILE_COUNT); off += TILE_COUNT;
    memcpy(out->walls_ew, data + off, TILE_COUNT); off += TILE_COUNT;

    memset(out->walls_diag, 0, sizeof(out->walls_diag));
    const uint8_t *diag1 = data + off; off += TILE_COUNT;
    const uint8_t *diag2 = data + off; off += TILE_COUNT;
    for (int i = 0; i < TILE_COUNT; i++) if (diag1[i]) out->walls_diag[i] = diag1[i];
    for (int i = 0; i < TILE_COUNT; i++) if (diag2[i]) out->walls_diag[i] = (uint16_t)(diag2[i] + DIAG_NW_SE_OFFSET);

    memset(out->roofs, 0, TILE_COUNT);
    memset(out->tile_decoration, 0, TILE_COUNT);
    memset(out->tile_direction, 0, TILE_COUNT);
    if (!dat_rle0_decode_u8(data, len, &off, out->roofs)) return 0;
    if (!dat_repeat_last_decode_u8(data, len, &off, out->tile_decoration)) return 0;
    if (!dat_rle0_decode_u8(data, len, &off, out->tile_direction)) return 0;
    return 1;
}

static int decode_dat_v63(const uint8_t *data, size_t len, chunk_data_t *out) {
    /*
     * RuneCast is built for the maps63/layout where VERSION_MAPS > 53. In that
     * path, world_load_section_files() reads the first four wall streams as raw
     * 48*48 byte planes:
     *
     *   walls_north_south  raw u8
     *   walls_east_west    raw u8
     *   diagonal \        raw u8
     *   diagonal /         raw u8, stored as value + 12000 when non-zero
     *
     * Only the roof stream and later tile fields are RLE-style. The previous
     * prototype tried the older RLE wall layout first; it can appear to decode
     * successfully against maps63 while shifting every following stream. That is
     * exactly the kind of failure that produces incomplete fences/walls and bad
     * diagonal-roof decisions. Prefer raw-walls decoding for v63 and keep RLE as
     * a fallback only for older/nonstandard packs.
     */
    chunk_data_t raw = *out;
    const int ok_raw = decode_dat_v63_try_raw_walls(data, len, &raw);
    if (ok_raw) {
        memcpy(out->walls_ns, raw.walls_ns, TILE_COUNT);
        memcpy(out->walls_ew, raw.walls_ew, TILE_COUNT);
        memcpy(out->walls_diag, raw.walls_diag, sizeof(out->walls_diag));
        memcpy(out->roofs, raw.roofs, TILE_COUNT);
        memcpy(out->tile_decoration, raw.tile_decoration, TILE_COUNT);
        memcpy(out->tile_direction, raw.tile_direction, TILE_COUNT);
        return 1;
    }

    chunk_data_t rle = *out;
    const int ok_rle = decode_dat_v63_try_rle(data, len, &rle);
    if (ok_rle) {
        memcpy(out->walls_ns, rle.walls_ns, TILE_COUNT);
        memcpy(out->walls_ew, rle.walls_ew, TILE_COUNT);
        memcpy(out->walls_diag, rle.walls_diag, sizeof(out->walls_diag));
        memcpy(out->roofs, rle.roofs, TILE_COUNT);
        memcpy(out->tile_decoration, rle.tile_decoration, TILE_COUNT);
        memcpy(out->tile_direction, rle.tile_direction, TILE_COUNT);
        return 1;
    }

    return 0;
}

static int load_chunk_hei(const jag_archive_t *land, const jag_archive_t *land_mem,
                          int plane, int x, int y, uint8_t out_h[TILE_COUNT]) {
    if (x < 0 || y < 0 || x > 99 || y > 99) return 0;
    char name[32];
    snprintf(name, sizeof(name), "m%d%d%d%d%d.hei", plane, x / 10, x % 10, y / 10, y % 10);
    const jag_entry_t *e = jag_find_by_hash_either(land, land_mem, rsc_hash_name(name));
    if (!e) return 0;
    size_t len = 0;
    uint8_t *blob = jag_extract_entry(e, &len);
    if (!blob) return 0;
    int ok = decode_hei(blob, len, out_h);
    free(blob);
    return ok;
}

static int load_chunk_dat(const jag_archive_t *maps, const jag_archive_t *maps_mem,
                          int plane, int x, int y, chunk_data_t *out) {
    if (x < 0 || y < 0 || x > 99 || y > 99) return 0;
    char name[32];
    snprintf(name, sizeof(name), "m%d%d%d%d%d.dat", plane, x / 10, x % 10, y / 10, y % 10);
    const jag_entry_t *e = jag_find_by_hash_either(maps, maps_mem, rsc_hash_name(name));
    if (!e) return 0;
    size_t len = 0;
    uint8_t *blob = jag_extract_entry(e, &len);
    if (!blob) return 0;
    int ok = decode_dat_v63(blob, len, out);
    free(blob);
    return ok;
}

typedef struct nb_data {
    chunk_data_t c[3][3]; /* [dx+1][dy+1] */
} nb_data_t;

static void load_neighborhood(nb_data_t *nb, const jag_archive_t *land, const jag_archive_t *land_mem,
                              const jag_archive_t *maps, const jag_archive_t *maps_mem,
                              int plane, int cx, int cy) {
    memset(nb, 0, sizeof(*nb));
    for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
            chunk_data_t *c = &nb->c[dx + 1][dy + 1];
            c->have_h = load_chunk_hei(land, land_mem, plane, cx + dx, cy + dy, c->h);
            c->have_dat = load_chunk_dat(maps, maps_mem, plane, cx + dx, cy + dy, c);
        }
    }
}

static const chunk_data_t *nb_chunk_const(const nb_data_t *nb, int *x, int *y) {
    int dx = 0, dy = 0;
    while (*x < 0) { *x += REGION_SIZE; dx--; }
    while (*y < 0) { *y += REGION_SIZE; dy--; }
    while (*x >= REGION_SIZE) { *x -= REGION_SIZE; dx++; }
    while (*y >= REGION_SIZE) { *y -= REGION_SIZE; dy++; }
    if (dx < -1 || dx > 1 || dy < -1 || dy > 1) return NULL;
    return &nb->c[dx + 1][dy + 1];
}

static int nb_height_raw(const nb_data_t *nb, int x, int y) {
    int lx = x, ly = y;
    const chunk_data_t *c = nb_chunk_const(nb, &lx, &ly);
    if (c && c->have_h) return (int)c->h[lx * REGION_SIZE + ly] * 3;

    /* Conservative edge clamp fallback if a neighbor is missing. */
    if (x < 0) x = 0;
    if (y < 0) y = 0;
    if (x >= REGION_SIZE) x = REGION_SIZE - 1;
    if (y >= REGION_SIZE) y = REGION_SIZE - 1;
    c = &nb->c[1][1];
    if (!c->have_h) return 0;
    return (int)c->h[x * REGION_SIZE + y] * 3;
}

static int nb_roof_raw(const nb_data_t *nb, int x, int y) {
    int lx = x, ly = y;
    const chunk_data_t *c = nb_chunk_const(nb, &lx, &ly);
    if (!c || !c->have_dat) return 0;
    return (int)c->roofs[lx * REGION_SIZE + ly];
}

static int nb_diag_raw(const nb_data_t *nb, int x, int y) {
    int lx = x, ly = y;
    const chunk_data_t *c = nb_chunk_const(nb, &lx, &ly);
    if (!c || !c->have_dat) return 0;
    return (int)c->walls_diag[lx * REGION_SIZE + ly];
}

static int nb_wall_ew_raw(const nb_data_t *nb, int x, int y) {
    int lx = x, ly = y;
    const chunk_data_t *c = nb_chunk_const(nb, &lx, &ly);
    if (!c || !c->have_dat) return 0;
    return (int)c->walls_ew[lx * REGION_SIZE + ly];
}

static int nb_wall_ns_raw(const nb_data_t *nb, int x, int y) {
    int lx = x, ly = y;
    const chunk_data_t *c = nb_chunk_const(nb, &lx, &ly);
    if (!c || !c->have_dat) return 0;
    return (int)c->walls_ns[lx * REGION_SIZE + ly];
}

static int nb_tile_decoration_raw(const nb_data_t *nb, int x, int y) {
    int lx = x, ly = y;
    const chunk_data_t *c = nb_chunk_const(nb, &lx, &ly);
    if (!c || !c->have_dat) return 0;
    return (int)c->tile_decoration[lx * REGION_SIZE + ly];
}

static int nb_tile_direction_raw(const nb_data_t *nb, int x, int y) {
    int lx = x, ly = y;
    const chunk_data_t *c = nb_chunk_const(nb, &lx, &ly);
    if (!c || !c->have_dat) return 0;
    return (int)c->tile_direction[lx * REGION_SIZE + ly];
}

static int wall_blocks_route(const config_db_t *cfg, int wall_id) {
    if (!cfg || wall_id < 0 || wall_id >= cfg->wall_count) return 0;
    const wallcfg_t *w = &cfg->walls[wall_id];
    return w->interactive == 0 && w->blocking != 0;
}

static int tile_blocks_route(const config_db_t *cfg, int decoration) {
    if (!cfg || decoration <= 0 || decoration - 1 >= cfg->tile_count) return 0;
    return cfg->tiles[decoration - 1].blocking != 0;
}

static int tile_is_floor(const config_db_t *cfg, int decoration) {
    if (!cfg || decoration <= 0 || decoration - 1 >= cfg->tile_count) return 0;
    return cfg->tiles[decoration - 1].type == RCW_FLOOR_TILE_TYPE;
}

static int has_roof_full(const nb_data_t *nb, int x, int y) {
    return nb_roof_raw(nb, x, y) > 0 &&
           nb_roof_raw(nb, x - 1, y) > 0 &&
           nb_roof_raw(nb, x - 1, y - 1) > 0 &&
           nb_roof_raw(nb, x, y - 1) > 0;
}

static int has_neighbouring_roof(const nb_data_t *nb, int x, int y) {
    return nb_roof_raw(nb, x, y) > 0 ||
           nb_roof_raw(nb, x - 1, y) > 0 ||
           nb_roof_raw(nb, x - 1, y - 1) > 0 ||
           nb_roof_raw(nb, x, y - 1) > 0;
}

/* -------------------- RCW1 file structs -------------------- */

typedef struct rcm_header {
    char     magic[4];
    uint16_t version;
    uint16_t flags;
    uint32_t file_size;
    uint32_t vertex_count;
    uint32_t index_count;
    uint32_t submesh_count;
    uint32_t submesh_off;
    uint32_t vertex_off;
    uint32_t index_off;
    uint16_t vertex_stride;
    uint16_t index_stride;
    uint16_t uv_divisor;
    uint16_t reserved0;
    int16_t  aabb_min[3];
    int16_t  aabb_max[3];
    int16_t  bounds_pad;
    int16_t  pad_align0;
    uint32_t reserved1[9];
} rcm_header_t;

typedef struct rcm_submesh {
    uint16_t texture_id;
    uint16_t color_argb1555;
    uint16_t flags;
    uint16_t reserved0;
    uint32_t first_index;
    uint32_t index_count;
} rcm_submesh_t;

typedef struct rcm_vtxf {
    float  x, y, z;
    int8_t nx, ny, nz;
    uint8_t pad1;
    int16_t u, v;
} rcm_vtxf_t;

STATIC_ASSERT(sizeof(rcm_header_t) == 96, rcm_header_must_be_96_bytes);
STATIC_ASSERT(sizeof(rcm_submesh_t) == 16, rcm_submesh_must_be_16_bytes);
STATIC_ASSERT(sizeof(rcm_vtxf_t) == 20, rcm_vtxf_must_be_20_bytes);

typedef struct material_key {
    uint16_t texture_id;
    uint16_t color_argb1555;
    uint16_t flags;
} material_key_t;

typedef struct mesh {
    rcm_vtxf_t *v;
    uint32_t v_count, v_cap;
    uint16_t *idx;
    uint32_t idx_count, idx_cap;
    rcm_submesh_t *sub;
    uint32_t sub_count, sub_cap;
    float aabb_min[3], aabb_max[3];
    int has_aabb;
} mesh_t;

static void mesh_init(mesh_t *m) {
    memset(m, 0, sizeof(*m));
}

static void mesh_free(mesh_t *m) {
    free(m->v);
    free(m->idx);
    free(m->sub);
    memset(m, 0, sizeof(*m));
}

static void mesh_aabb_add(mesh_t *m, float x, float y, float z) {
    if (!m->has_aabb) {
        m->aabb_min[0] = m->aabb_max[0] = x;
        m->aabb_min[1] = m->aabb_max[1] = y;
        m->aabb_min[2] = m->aabb_max[2] = z;
        m->has_aabb = 1;
        return;
    }
    if (x < m->aabb_min[0]) m->aabb_min[0] = x;
    if (y < m->aabb_min[1]) m->aabb_min[1] = y;
    if (z < m->aabb_min[2]) m->aabb_min[2] = z;
    if (x > m->aabb_max[0]) m->aabb_max[0] = x;
    if (y > m->aabb_max[1]) m->aabb_max[1] = y;
    if (z > m->aabb_max[2]) m->aabb_max[2] = z;
}

static int mesh_push_vertex(mesh_t *m, const rcm_vtxf_t *v, uint16_t *out_index) {
    if (m->v_count >= 65535u) return 0;
    if (m->v_count == m->v_cap) {
        m->v_cap = m->v_cap ? m->v_cap * 2u : 1024u;
        m->v = (rcm_vtxf_t *)xrealloc(m->v, (size_t)m->v_cap * sizeof(m->v[0]));
    }
    m->v[m->v_count] = *v;
    *out_index = (uint16_t)m->v_count++;
    mesh_aabb_add(m, v->x, v->y, v->z);
    return 1;
}

static int mesh_push_index(mesh_t *m, uint16_t idx) {
    if (m->idx_count == m->idx_cap) {
        m->idx_cap = m->idx_cap ? m->idx_cap * 2u : 2048u;
        m->idx = (uint16_t *)xrealloc(m->idx, (size_t)m->idx_cap * sizeof(m->idx[0]));
    }
    m->idx[m->idx_count++] = idx;
    return 1;
}

static int mesh_begin_submesh(mesh_t *m, material_key_t mat) {
    if (m->sub_count > 0) {
        rcm_submesh_t *prev = &m->sub[m->sub_count - 1];
        if (prev->texture_id == mat.texture_id &&
            prev->color_argb1555 == mat.color_argb1555 &&
            prev->flags == mat.flags) {
            return 1;
        }
    }
    if (m->sub_count == m->sub_cap) {
        m->sub_cap = m->sub_cap ? m->sub_cap * 2u : 256u;
        m->sub = (rcm_submesh_t *)xrealloc(m->sub, (size_t)m->sub_cap * sizeof(m->sub[0]));
    }
    rcm_submesh_t *s = &m->sub[m->sub_count++];
    memset(s, 0, sizeof(*s));
    s->texture_id = mat.texture_id;
    s->color_argb1555 = mat.color_argb1555;
    s->flags = mat.flags;
    s->first_index = m->idx_count;
    s->index_count = 0;
    return 1;
}

static int mesh_append_tri_indexed(mesh_t *m, uint16_t a, uint16_t b, uint16_t c) {
    if (m->sub_count == 0) return 0;
    mesh_push_index(m, a);
    mesh_push_index(m, b);
    mesh_push_index(m, c);
    m->sub[m->sub_count - 1].index_count += 3;
    return 1;
}

static int submesh_same_material(const rcm_submesh_t *a, const rcm_submesh_t *b) {
    return a && b &&
           a->texture_id == b->texture_id &&
           a->color_argb1555 == b->color_argb1555 &&
           a->flags == b->flags;
}

static int mesh_batch_materials(mesh_t *m) {
    if (!m || m->sub_count <= 1 || m->idx_count == 0) return 1;

    rcm_submesh_t *new_sub = (rcm_submesh_t *)xcalloc(m->sub_count, sizeof(new_sub[0]));
    uint16_t *new_idx = (uint16_t *)xcalloc(m->idx_count, sizeof(new_idx[0]));
    uint8_t *consumed = (uint8_t *)xcalloc(m->sub_count, sizeof(consumed[0]));
    uint32_t new_sub_count = 0;
    uint32_t new_idx_count = 0;

    for (uint32_t seed = 0; seed < m->sub_count; seed++) {
        if (consumed[seed] || m->sub[seed].index_count == 0) continue;

        rcm_submesh_t *dst = &new_sub[new_sub_count++];
        *dst = m->sub[seed];
        dst->first_index = new_idx_count;
        dst->index_count = 0;

        for (uint32_t si = seed; si < m->sub_count; si++) {
            const rcm_submesh_t *src = &m->sub[si];
            if (consumed[si] || src->index_count == 0) continue;
            if (!submesh_same_material(dst, src)) continue;

            if (src->first_index > m->idx_count ||
                src->index_count > m->idx_count - src->first_index ||
                new_idx_count > m->idx_count - src->index_count) {
                free(consumed);
                free(new_sub);
                free(new_idx);
                return 0;
            }

            memcpy(new_idx + new_idx_count, m->idx + src->first_index,
                   (size_t)src->index_count * sizeof(new_idx[0]));
            dst->index_count += src->index_count;
            new_idx_count += src->index_count;
            consumed[si] = 1;
        }
    }

    free(consumed);
    free(m->sub);
    free(m->idx);
    m->sub = new_sub;
    m->idx = new_idx;
    m->sub_count = new_sub_count;
    m->sub_cap = m->sub_count;
    m->idx_count = new_idx_count;
    m->idx_cap = m->idx_count;
    return 1;
}

static int mesh_deduplicate_vertices(mesh_t *m) {
    if (!m || m->v_count <= 1 || m->idx_count == 0) return 1;

    rcm_vtxf_t *new_v = (rcm_vtxf_t *)xcalloc(m->v_count, sizeof(new_v[0]));
    uint16_t *remap = (uint16_t *)xcalloc(m->v_count, sizeof(remap[0]));
    uint32_t new_v_count = 0;

    for (uint32_t vi = 0; vi < m->v_count; vi++) {
        uint32_t dst = new_v_count;
        for (uint32_t i = 0; i < new_v_count; i++) {
            if (memcmp(&new_v[i], &m->v[vi], sizeof(new_v[i])) == 0) {
                dst = i;
                break;
            }
        }

        if (dst == new_v_count) {
            if (new_v_count >= 65535u) {
                free(new_v);
                free(remap);
                return 0;
            }
            new_v[new_v_count++] = m->v[vi];
        }

        remap[vi] = (uint16_t)dst;
    }

    for (uint32_t ii = 0; ii < m->idx_count; ii++) {
        uint16_t old = m->idx[ii];
        if ((uint32_t)old >= m->v_count) {
            free(new_v);
            free(remap);
            return 0;
        }
        m->idx[ii] = remap[old];
    }

    free(remap);
    free(m->v);
    m->v = new_v;
    m->v_count = new_v_count;
    m->v_cap = m->v_count;
    return 1;
}

static void normalize3(float *x, float *y, float *z) {
    float l = sqrtf((*x) * (*x) + (*y) * (*y) + (*z) * (*z));
    if (l <= 1.0e-20f) { *x = 0.0f; *y = 1.0f; *z = 0.0f; return; }
    *x /= l; *y /= l; *z /= l;
}

static int8_t snorm8(float v) {
    if (v < -1.0f) v = -1.0f;
    if (v >  1.0f) v =  1.0f;
    return (int8_t)lrintf(v * 127.0f);
}

static void compute_normal(const float p[4][3], int n, int8_t out[3]) {
    (void)n;
    float ax = p[1][0] - p[0][0];
    float ay = p[1][1] - p[0][1];
    float az = p[1][2] - p[0][2];
    float bx = p[2][0] - p[0][0];
    float by = p[2][1] - p[0][1];
    float bz = p[2][2] - p[0][2];
    float nx = ay * bz - az * by;
    float ny = az * bx - ax * bz;
    float nz = ax * by - ay * bx;
    normalize3(&nx, &ny, &nz);
    out[0] = snorm8(nx); out[1] = snorm8(ny); out[2] = snorm8(nz);
}

static uint16_t fill_to_argb1555(int fill) {
    if (fill < 0) {
        int c = -1 - fill;
        return (uint16_t)(0x8000u | (uint16_t)(c & 0x7FFF));
    }
    return 0xFFFFu;
}

static int material_from_fill(int fill, material_key_t *out) {
    if (fill == COLOUR_TRANSPARENT) return 0;
    memset(out, 0, sizeof(*out));
    if (fill >= 0) {
        const uint16_t tex = (uint16_t)fill;
        if (texture_id_has_atlas(tex)) {
            out->texture_id = tex;
            out->color_argb1555 = 0xFFFFu;
        } else {
            /* Do not let an out-of-range texture id sample the whole atlas. */
            out->texture_id = RCM_SUBMESH_TEX_NONE;
            out->color_argb1555 = 0xFFFFu;
        }
    } else {
        out->texture_id = RCM_SUBMESH_TEX_NONE;
        out->color_argb1555 = fill_to_argb1555(fill);
    }
    out->flags = 0;
    return 1;
}

static float rcw_x(int x) { return (float)x / VERTEX_SCALE_F; }
static float rcw_y(int y) { return (float)y / VERTEX_SCALE_F; }
static float rcw_z(int z) { return (float)z / VERTEX_SCALE_F; }

static int mesh_add_face(mesh_t *m, const float p_in[4][3], int n, int fill) {
    material_key_t mat;
    if (!material_from_fill(fill, &mat)) return 1;
    if (!mesh_begin_submesh(m, mat)) return 0;

    float p[4][3];
    for (int i = 0; i < n; ++i) memcpy(p[i], p_in[i], sizeof(p[i]));

    int8_t normal[3];
    compute_normal(p, n, normal);

    static const float q_uv01[4][2] = {
        {0.0f, 0.0f}, {1.0f, 0.0f}, {1.0f, 1.0f}, {0.0f, 1.0f}
    };
    static const float t_uv01[3][2] = {
        {0.0f, 0.0f}, {1.0f, 0.0f}, {0.0f, 1.0f}
    };

    uint16_t vi[4];
    for (int i = 0; i < n; ++i) {
        rcm_vtxf_t v;
        const float u01 = (n == 4) ? q_uv01[i][0] : t_uv01[i][0];
        const float v01 = (n == 4) ? q_uv01[i][1] : t_uv01[i][1];

        v.x = p[i][0]; v.y = p[i][1]; v.z = p[i][2];
        v.nx = normal[0]; v.ny = normal[1]; v.nz = normal[2]; v.pad1 = 0;

        if (mat.texture_id != RCM_SUBMESH_TEX_NONE) {
            atlas_uv_to_q15(mat.texture_id, u01, v01, &v.u, &v.v);
        } else {
            v.u = 0;
            v.v = 0;
        }

        if (!mesh_push_vertex(m, &v, &vi[i])) return 0;
    }

    if (n == 3) {
        mesh_append_tri_indexed(m, vi[0], vi[1], vi[2]);
    } else {
        mesh_append_tri_indexed(m, vi[0], vi[1], vi[2]);
        mesh_append_tri_indexed(m, vi[0], vi[2], vi[3]);
    }
    return 1;
}

static int mesh_add_face_reversed(mesh_t *m, const float p[4][3], int n, int fill) {
    float r[4][3];
    for (int i = 0; i < n; ++i) memcpy(r[i], p[n - 1 - i], sizeof(r[i]));
    return mesh_add_face(m, r, n, fill);
}

static int add_wall_segment(mesh_t *m, const nb_data_t *nb, const config_db_t *cfg,
                            int wall_object_id, int x1, int y1, int x2, int y2,
                            int thick, int coplanar_backface) {
    if (wall_object_id < 0 || wall_object_id >= cfg->wall_count) return 1;
    const wallcfg_t *w = &cfg->walls[wall_object_id];
    if (w->interactive != 0) return 1;

    int vx1 = x1 * TILE_SIZE;
    int vy1 = y1 * TILE_SIZE;
    int vx2 = x2 * TILE_SIZE;
    int vy2 = y2 * TILE_SIZE;

    if (thick) {
        int dx = (y1 - y2) * 8;
        vx1 += dx; vx2 += dx;
        int dy = (x1 - x2) * 8;
        vy1 -= dy; vy2 -= dy;
    }

    const int h1 = nb_height_raw(nb, x1, y1);
    const int h2 = nb_height_raw(nb, x2, y2);

    float face[4][3] = {
        { rcw_x(vx1), rcw_y(-h1),             rcw_z(vy1) },
        { rcw_x(vx1), rcw_y(-h1 - w->height), rcw_z(vy1) },
        { rcw_x(vx2), rcw_y(-h2 - w->height), rcw_z(vy2) },
        { rcw_x(vx2), rcw_y(-h2),             rcw_z(vy2) },
    };

    if (!mesh_add_face(m, face, 4, w->front)) return 0;

    /* Default to one two-sided plane. PVR renders with culling disabled, so the
       old coplanar reversed back face causes redundant overdraw/depth noise. */
    if (!thick && coplanar_backface &&
        !mesh_add_face_reversed(m, face, 4, w->back)) return 0;

    if (thick) {
        int pvx1 = x1 * TILE_SIZE;
        int pvy1 = y1 * TILE_SIZE;
        int pvx2 = x2 * TILE_SIZE;
        int pvy2 = y2 * TILE_SIZE;
        int dx = (y1 - y2) * 8;
        pvx1 -= dx; pvx2 -= dx;
        int dy = (x1 - x2) * 8;
        pvy1 += dy; pvy2 += dy;

        float back_face[4][3] = {
            { rcw_x(pvx1), rcw_y(-h1),             rcw_z(pvy1) },
            { rcw_x(pvx1), rcw_y(-h1 - w->height), rcw_z(pvy1) },
            { rcw_x(pvx2), rcw_y(-h2 - w->height), rcw_z(pvy2) },
            { rcw_x(pvx2), rcw_y(-h2),             rcw_z(pvy2) },
        };
        if (!mesh_add_face_reversed(m, back_face, 4, w->back)) return 0;

        float top[4][3];
        memcpy(top[0], face[1], sizeof(top[0]));
        memcpy(top[1], back_face[1], sizeof(top[1]));
        memcpy(top[2], back_face[2], sizeof(top[2]));
        memcpy(top[3], face[2], sizeof(top[3]));
        if (!mesh_add_face(m, top, 4, -7400)) return 0;
    }

    return 1;
}

static int bake_walls(mesh_t *m, const nb_data_t *nb, const config_db_t *cfg,
                      int thick, int coplanar_backface) {
    const chunk_data_t *c = &nb->c[1][1];
    if (!c->have_dat) return 0;

    for (int x = 0; x < REGION_SIZE; ++x) {
        for (int y = 0; y < REGION_SIZE; ++y) {
            const int idx = x * REGION_SIZE + y;
            int wall = c->walls_ew[idx];
            if (wall > 0) {
                if (!add_wall_segment(m, nb, cfg, wall - 1, x, y, x + 1, y, thick, coplanar_backface)) return 0;
            }
            wall = c->walls_ns[idx];
            if (wall > 0) {
                if (!add_wall_segment(m, nb, cfg, wall - 1, x, y, x, y + 1, thick, coplanar_backface)) return 0;
            }
            wall = c->walls_diag[idx];
            if (wall > 0 && wall < 12000) {
                if (!add_wall_segment(m, nb, cfg, wall - 1, x, y, x + 1, y + 1, thick, coplanar_backface)) return 0;
            } else if (wall > 12000 && wall < 24000) {
                if (!add_wall_segment(m, nb, cfg, wall - 12001, x + 1, y, x, y + 1, thick, coplanar_backface)) return 0;
            }
        }
    }
    return m->idx_count > 0;
}

static void raise_wall_object(int terrain[REGION_SIZE + 1][REGION_SIZE + 1], const config_db_t *cfg,
                              int wall_object_id, int x1, int y1, int x2, int y2) {
    if (wall_object_id < 0 || wall_object_id >= cfg->wall_count) return;
    const int h = cfg->walls[wall_object_id].height;
    if (x1 >= 0 && x1 <= REGION_SIZE && y1 >= 0 && y1 <= REGION_SIZE && terrain[x1][y1] < PLANE_HEIGHT) {
        terrain[x1][y1] += PLANE_HEIGHT + h;
    }
    if (x2 >= 0 && x2 <= REGION_SIZE && y2 >= 0 && y2 <= REGION_SIZE && terrain[x2][y2] < PLANE_HEIGHT) {
        terrain[x2][y2] += PLANE_HEIGHT + h;
    }
}

static int bake_roofs(mesh_t *m, const nb_data_t *nb, const config_db_t *cfg) {
    const chunk_data_t *c = &nb->c[1][1];
    if (!c->have_dat) return 0;

    int terrain[REGION_SIZE + 1][REGION_SIZE + 1];
    for (int x = 0; x <= REGION_SIZE; ++x) {
        for (int y = 0; y <= REGION_SIZE; ++y) terrain[x][y] = nb_height_raw(nb, x, y);
    }

    for (int x = 0; x < REGION_SIZE; ++x) {
        for (int y = 0; y < REGION_SIZE; ++y) {
            int idx = x * REGION_SIZE + y;
            int wall = c->walls_ew[idx];
            if (wall > 0) raise_wall_object(terrain, cfg, wall - 1, x, y, x + 1, y);
            wall = c->walls_ns[idx];
            if (wall > 0) raise_wall_object(terrain, cfg, wall - 1, x, y, x, y + 1);
            wall = c->walls_diag[idx];
            if (wall > 0 && wall < 12000) raise_wall_object(terrain, cfg, wall - 1, x, y, x + 1, y + 1);
            if (wall > 12000 && wall < 24000) raise_wall_object(terrain, cfg, wall - 12001, x + 1, y, x, y + 1);
        }
    }

    /* First normalization pass mirrors world_load_section(). */
    for (int x = 0; x < REGION_SIZE; ++x) {
        for (int y = 0; y < REGION_SIZE; ++y) {
            int roof_id = c->roofs[x * REGION_SIZE + y];
            if (roof_id <= 0) continue;
            int ex = x + 1, sy = y + 1;
            int h0 = terrain[x][y];
            int h1 = terrain[ex][y];
            int h2 = terrain[ex][sy];
            int h3 = terrain[x][sy];
            if (h0 > PLANE_HEIGHT) h0 -= PLANE_HEIGHT;
            if (h1 > PLANE_HEIGHT) h1 -= PLANE_HEIGHT;
            if (h2 > PLANE_HEIGHT) h2 -= PLANE_HEIGHT;
            if (h3 > PLANE_HEIGHT) h3 -= PLANE_HEIGHT;
            int h = h0;
            if (h1 > h) h = h1;
            if (h2 > h) h = h2;
            if (h3 > h) h = h3;
            if (h >= PLANE_HEIGHT) h -= PLANE_HEIGHT;
            if (h0 < PLANE_HEIGHT) terrain[x][y] = h; else terrain[x][y] -= PLANE_HEIGHT;
            if (h1 < PLANE_HEIGHT) terrain[ex][y] = h; else terrain[ex][y] -= PLANE_HEIGHT;
            if (h2 < PLANE_HEIGHT) terrain[ex][sy] = h; else terrain[ex][sy] -= PLANE_HEIGHT;
            if (h3 < PLANE_HEIGHT) terrain[x][sy] = h; else terrain[x][sy] -= PLANE_HEIGHT;
        }
    }

    for (int x = 0; x < REGION_SIZE; ++x) {
        for (int y = 0; y < REGION_SIZE; ++y) {
            int roof_raw = c->roofs[x * REGION_SIZE + y];
            if (roof_raw <= 0) continue;
            if (roof_raw - 1 < 0 || roof_raw - 1 >= cfg->roof_count) continue;

            int ex = x + 1, sy = y + 1;
            int v1x = x * TILE_SIZE, v1z = y * TILE_SIZE;
            int v2x = v1x + TILE_SIZE, v2z = v1z;
            int v4x = v2x, v4z = v1z + TILE_SIZE;
            int v3x = v1x, v3z = v4z;

            int h0 = terrain[x][y];
            int h1 = terrain[ex][y];
            int h2 = terrain[ex][sy];
            int h3 = terrain[x][sy];
            int rh = cfg->roofs[roof_raw - 1].height;

            if (has_roof_full(nb, x, y) && h0 < PLANE_HEIGHT) { h0 += rh + PLANE_HEIGHT; terrain[x][y] = h0; }
            if (has_roof_full(nb, ex, y) && h1 < PLANE_HEIGHT) { h1 += rh + PLANE_HEIGHT; terrain[ex][y] = h1; }
            if (has_roof_full(nb, ex, sy) && h2 < PLANE_HEIGHT) { h2 += rh + PLANE_HEIGHT; terrain[ex][sy] = h2; }
            if (has_roof_full(nb, x, sy) && h3 < PLANE_HEIGHT) { h3 += rh + PLANE_HEIGHT; terrain[x][sy] = h3; }

            if (h0 >= PLANE_HEIGHT) h0 -= PLANE_HEIGHT;
            if (h1 >= PLANE_HEIGHT) h1 -= PLANE_HEIGHT;
            if (h2 >= PLANE_HEIGHT) h2 -= PLANE_HEIGHT;
            if (h3 >= PLANE_HEIGHT) h3 -= PLANE_HEIGHT;

            if (!has_neighbouring_roof(nb, x - 1, y)) v1x -= ROOF_SLOPE;
            if (!has_neighbouring_roof(nb, x + 1, y)) v1x += ROOF_SLOPE;
            if (!has_neighbouring_roof(nb, x, y - 1)) v1z -= ROOF_SLOPE;
            if (!has_neighbouring_roof(nb, x, y + 1)) v1z += ROOF_SLOPE;

            if (!has_neighbouring_roof(nb, ex - 1, y)) v2x -= ROOF_SLOPE;
            if (!has_neighbouring_roof(nb, ex + 1, y)) v2x += ROOF_SLOPE;
            if (!has_neighbouring_roof(nb, ex, y - 1)) v2z -= ROOF_SLOPE;
            if (!has_neighbouring_roof(nb, ex, y + 1)) v2z += ROOF_SLOPE;

            if (!has_neighbouring_roof(nb, ex - 1, sy)) v4x -= ROOF_SLOPE;
            if (!has_neighbouring_roof(nb, ex + 1, sy)) v4x += ROOF_SLOPE;
            if (!has_neighbouring_roof(nb, ex, sy - 1)) v4z -= ROOF_SLOPE;
            if (!has_neighbouring_roof(nb, ex, sy + 1)) v4z += ROOF_SLOPE;

            if (!has_neighbouring_roof(nb, x - 1, sy)) v3x -= ROOF_SLOPE;
            if (!has_neighbouring_roof(nb, x + 1, sy)) v3x += ROOF_SLOPE;
            if (!has_neighbouring_roof(nb, x, sy - 1)) v3z -= ROOF_SLOPE;
            if (!has_neighbouring_roof(nb, x, sy + 1)) v3z += ROOF_SLOPE;

            const int fill = cfg->roofs[roof_raw - 1].fill;
            h0 = -h0; h1 = -h1; h2 = -h2; h3 = -h3;
            const int diag = nb_diag_raw(nb, x, y);

            float p1[3] = { rcw_x(v1x), rcw_y(h0), rcw_z(v1z) };
            float p2[3] = { rcw_x(v2x), rcw_y(h1), rcw_z(v2z) };
            float p4[3] = { rcw_x(v4x), rcw_y(h2), rcw_z(v4z) };
            float p3[3] = { rcw_x(v3x), rcw_y(h3), rcw_z(v3z) };
            float tri[4][3];

#define SET3(A,B,C) do { memcpy(tri[0], A, sizeof(tri[0])); memcpy(tri[1], B, sizeof(tri[1])); memcpy(tri[2], C, sizeof(tri[2])); } while (0)
#define SET4(A,B,C,D) do { memcpy(tri[0], A, sizeof(tri[0])); memcpy(tri[1], B, sizeof(tri[1])); memcpy(tri[2], C, sizeof(tri[2])); memcpy(tri[3], D, sizeof(tri[3])); } while (0)

            if (diag > 12000 && diag < 24000 && nb_roof_raw(nb, x - 1, y - 1) == 0) {
                SET3(p4, p3, p2); if (!mesh_add_face(m, tri, 3, fill)) return 0;
            } else if (diag > 12000 && diag < 24000 && nb_roof_raw(nb, x + 1, y + 1) == 0) {
                SET3(p1, p2, p3); if (!mesh_add_face(m, tri, 3, fill)) return 0;
            } else if (diag > 0 && diag < 12000 && nb_roof_raw(nb, x + 1, y - 1) == 0) {
                SET3(p3, p1, p4); if (!mesh_add_face(m, tri, 3, fill)) return 0;
            } else if (diag > 0 && diag < 12000 && nb_roof_raw(nb, x - 1, y + 1) == 0) {
                SET3(p2, p4, p1); if (!mesh_add_face(m, tri, 3, fill)) return 0;
            } else if (h0 == h1 && h2 == h3) {
                SET4(p1, p2, p4, p3); if (!mesh_add_face(m, tri, 4, fill)) return 0;
            } else if (h0 == h3 && h1 == h2) {
                SET4(p3, p1, p2, p4); if (!mesh_add_face(m, tri, 4, fill)) return 0;
            } else {
                int direction = 1;
                if (nb_roof_raw(nb, x - 1, y - 1) > 0) direction = 0;
                if (nb_roof_raw(nb, x + 1, y + 1) > 0) direction = 0;
                if (!direction) {
                    SET3(p2, p4, p1); if (!mesh_add_face(m, tri, 3, fill)) return 0;
                    SET3(p3, p1, p4); if (!mesh_add_face(m, tri, 3, fill)) return 0;
                } else {
                    SET3(p1, p2, p3); if (!mesh_add_face(m, tri, 3, fill)) return 0;
                    SET3(p4, p3, p2); if (!mesh_add_face(m, tri, 3, fill)) return 0;
                }
            }
#undef SET3
#undef SET4
        }
    }

    return m->idx_count > 0;
}

static void build_rwm1_block(uint8_t out[RWM1_BLOCK_SIZE], const nb_data_t *nb,
                             const config_db_t *cfg, int plane, int sx, int sy) {
    memset(out, 0, RWM1_BLOCK_SIZE);

    uint8_t *adj = out + RWM1_HEADER_SIZE;
    uint8_t *dir = adj + TILE_COUNT;

    for (int x = 0; x < REGION_SIZE; x++) {
        for (int y = 0; y < REGION_SIZE; y++) {
            const int idx = x * REGION_SIZE + y;
            uint8_t a = 0;

            const int decoration = nb_tile_decoration_raw(nb, x, y);
            if (tile_blocks_route(cfg, decoration)) a |= 0x40u;
            if (tile_is_floor(cfg, decoration)) a |= 0x80u;

            int wall = nb_wall_ew_raw(nb, x, y);
            if (wall > 0 && wall_blocks_route(cfg, wall - 1)) a |= 0x01u;

            wall = nb_wall_ew_raw(nb, x, y + 1);
            if (wall > 0 && wall_blocks_route(cfg, wall - 1)) a |= 0x04u;

            wall = nb_wall_ns_raw(nb, x, y);
            if (wall > 0 && wall_blocks_route(cfg, wall - 1)) a |= 0x02u;

            wall = nb_wall_ns_raw(nb, x + 1, y);
            if (wall > 0 && wall_blocks_route(cfg, wall - 1)) a |= 0x08u;

            wall = nb_diag_raw(nb, x, y);
            if (wall > 0 && wall < 12000 && wall_blocks_route(cfg, wall - 1)) {
                a |= 0x20u;
            } else if (wall > 12000 && wall < 24000 && wall_blocks_route(cfg, wall - 12001)) {
                a |= 0x10u;
            }

            adj[idx] = a;

            const uint8_t d = (uint8_t)(nb_tile_direction_raw(nb, x, y) & 0x0f);
            if (idx & 1) {
                dir[idx >> 1] |= (uint8_t)(d << 4);
            } else {
                dir[idx >> 1] |= d;
            }
        }
    }

    memcpy(out, "RWM1", 4);
    write_u16le(out, 4, (uint16_t)RWM1_HEADER_SIZE);
    write_u16le(out, 6, 1);
    out[8] = (uint8_t)plane;
    out[9] = (uint8_t)sx;
    out[10] = (uint8_t)sy;
    write_u16le(out, 12, REGION_SIZE);
    write_u16le(out, 14, REGION_SIZE);

    uint32_t flags = RWM1_FLAGS_HAS_ADJACENCY | RWM1_FLAGS_HAS_DIRECTIONS;
    if (cfg && cfg->tile_count > 0) flags |= RWM1_FLAGS_HAS_TILE_CONFIG;
    if (nb && nb->c[1][1].have_dat) flags |= RWM1_FLAGS_HAS_DAT;
    write_u32le(out, 16, flags);
    write_u32le(out, 20, RWM1_HEADER_SIZE);
    write_u32le(out, 24, RWM1_HEADER_SIZE + TILE_COUNT);
    write_u32le(out, 28, RWM1_BLOCK_SIZE);
}

static int write_rcw1_file(const char *path, const mesh_t *m, int allow_empty,
                           const nb_data_t *rwm_nb, const config_db_t *rwm_cfg,
                           int plane, int sx, int sy) {
    if (!m) return 0;
    if (!allow_empty && (m->v_count == 0 || m->idx_count == 0 || m->sub_count == 0)) return 0;

    rcm_header_t hdr;
    memset(&hdr, 0, sizeof(hdr));
    memcpy(hdr.magic, "RCW1", 4);
    hdr.version = 1;
    const int has_rwm = (rwm_nb && rwm_cfg);
    if (has_rwm) hdr.flags |= RCW1_FLAG_HAS_RWM;
    hdr.vertex_count = m->v_count;
    hdr.index_count = m->idx_count;
    hdr.submesh_count = m->sub_count;
    hdr.vertex_stride = (uint16_t)sizeof(rcm_vtxf_t);
    hdr.index_stride = 2;
    hdr.uv_divisor = UV_Q15_DIVISOR;

    hdr.submesh_off = (uint32_t)align_up_sz(sizeof(hdr), RCM_FILE_ALIGN);
    hdr.vertex_off = (uint32_t)align_up_sz((size_t)hdr.submesh_off + (size_t)hdr.submesh_count * sizeof(rcm_submesh_t), RCM_FILE_ALIGN);
    hdr.index_off = (uint32_t)align_up_sz((size_t)hdr.vertex_off + (size_t)hdr.vertex_count * sizeof(rcm_vtxf_t), RCM_FILE_ALIGN);
    size_t idx_bytes = (size_t)hdr.index_count * sizeof(uint16_t);
    const size_t mesh_end = (size_t)hdr.index_off + idx_bytes;
    const size_t rwm_off = has_rwm ? align_up_sz(mesh_end, RCM_FILE_ALIGN) : 0u;
    const size_t end_off = has_rwm ? (rwm_off + RWM1_BLOCK_SIZE) : mesh_end;
    hdr.file_size = (uint32_t)align_up_sz(end_off, RCM_FILE_ALIGN);
    if (has_rwm) {
        hdr.reserved1[0] = RCW1_RWM_RESERVED_MAGIC;
        hdr.reserved1[1] = (uint32_t)rwm_off;
        hdr.reserved1[2] = (uint32_t)RWM1_BLOCK_SIZE;
    }

    if (m->has_aabb) {
        hdr.aabb_min[0] = (int16_t)lrintf(m->aabb_min[0] * VERTEX_SCALE_F);
        hdr.aabb_min[1] = (int16_t)lrintf(m->aabb_min[1] * VERTEX_SCALE_F);
        hdr.aabb_min[2] = (int16_t)lrintf(m->aabb_min[2] * VERTEX_SCALE_F);
        hdr.aabb_max[0] = (int16_t)lrintf(m->aabb_max[0] * VERTEX_SCALE_F);
        hdr.aabb_max[1] = (int16_t)lrintf(m->aabb_max[1] * VERTEX_SCALE_F);
        hdr.aabb_max[2] = (int16_t)lrintf(m->aabb_max[2] * VERTEX_SCALE_F);
    }

    uint8_t *blob = (uint8_t *)xcalloc(1, hdr.file_size);
    memcpy(blob, &hdr, sizeof(hdr));
    if (hdr.submesh_count) memcpy(blob + hdr.submesh_off, m->sub, (size_t)hdr.submesh_count * sizeof(rcm_submesh_t));
    if (hdr.vertex_count) memcpy(blob + hdr.vertex_off, m->v, (size_t)hdr.vertex_count * sizeof(rcm_vtxf_t));
    if (hdr.index_count) memcpy(blob + hdr.index_off, m->idx, idx_bytes);
    if (has_rwm) {
        uint8_t rwm[RWM1_BLOCK_SIZE];
        build_rwm1_block(rwm, rwm_nb, rwm_cfg, plane, sx, sy);
        memcpy(blob + rwm_off, rwm, RWM1_BLOCK_SIZE);
    }

    int ok = write_entire_file(path, blob, hdr.file_size);
    free(blob);
    return ok;
}

static void section_name(char out[32], int plane, int x, int y) {
    snprintf(out, 32, "m%d%d%d%d%d", plane, x / 10, x % 10, y / 10, y % 10);
}

static void usage(const char *exe) {
    fprintf(stderr,
        "Usage: %s land63.jag maps63.jag config85.jag out_dir [options]\n"
        "Options:\n"
        "  --land-mem <land63.mem>\n"
        "  --maps-mem <maps63.mem>\n"
        "  --plane <0..3>\n"
        "  --walls-only\n"
        "  --roofs-only\n"
        "  --thick-walls\n"
        "  --coplanar-backface\n"
        "  --deduplicate-vertices\n"
        "  --empty-files\n",
        exe);
}

int main(int argc, char **argv) {
    if (argc < 5) { usage(argv[0]); return 1; }

    const char *land_path = argv[1];
    const char *maps_path = argv[2];
    const char *config_path = argv[3];
    const char *out_dir = argv[4];
    const char *land_mem_path = NULL;
    const char *maps_mem_path = NULL;
    int plane_filter = -1;
    int do_walls = 1;
    int do_roofs = 1;
    int thick_walls = 0;
    int coplanar_backface = 0;
    int deduplicate_vertices = 0;
    int empty_files = 0;

    for (int i = 5; i < argc; ++i) {
        if (!strcmp(argv[i], "--land-mem") && i + 1 < argc) {
            land_mem_path = argv[++i];
        } else if (!strcmp(argv[i], "--maps-mem") && i + 1 < argc) {
            maps_mem_path = argv[++i];
        } else if (!strcmp(argv[i], "--plane") && i + 1 < argc) {
            plane_filter = atoi(argv[++i]);
            if (plane_filter < 0 || plane_filter > 3) die("--plane must be 0..3");
        } else if (!strcmp(argv[i], "--walls-only")) {
            do_roofs = 0;
        } else if (!strcmp(argv[i], "--roofs-only")) {
            do_walls = 0;
        } else if (!strcmp(argv[i], "--thick-walls")) {
            thick_walls = 1;
        } else if (!strcmp(argv[i], "--coplanar-backface")) {
            coplanar_backface = 1;
        } else if (!strcmp(argv[i], "--deduplicate-vertices")) {
            deduplicate_vertices = 1;
        } else if (!strcmp(argv[i], "--empty-files")) {
            empty_files = 1;
        } else {
            usage(argv[0]);
            return 1;
        }
    }

    if (!do_walls && !do_roofs) die("nothing to do");

    if (ensure_dir(out_dir) != 0) die("failed to create output directory");
    char walls_dir[1024], roofs_dir[1024];
    snprintf(walls_dir, sizeof(walls_dir), "%s/walls", out_dir);
    snprintf(roofs_dir, sizeof(roofs_dir), "%s/roofs", out_dir);
    if (do_walls && ensure_dir(walls_dir) != 0) die("failed to create walls directory");
    if (do_roofs && ensure_dir(roofs_dir) != 0) die("failed to create roofs directory");

    config_db_t cfg;
    if (!config_load(config_path, &cfg)) die("failed to load config jag");

    jag_archive_t land = jag_load_required(land_path);
    jag_archive_t maps = jag_load_required(maps_path);
    jag_archive_t land_mem = jag_load_optional(land_mem_path);
    jag_archive_t maps_mem = jag_load_optional(maps_mem_path);

    int written_walls = 0, written_roofs = 0, seen = 0;
    for (int plane = 0; plane <= 3; ++plane) {
        if (plane_filter >= 0 && plane != plane_filter) continue;
        for (int x = 0; x <= 99; ++x) {
            for (int y = 0; y <= 99; ++y) {
                char hei_name[32];
                snprintf(hei_name, sizeof(hei_name), "m%d%d%d%d%d.hei", plane, x / 10, x % 10, y / 10, y % 10);
                char dat_name[32];
                snprintf(dat_name, sizeof(dat_name), "m%d%d%d%d%d.dat", plane, x / 10, x % 10, y / 10, y % 10);
                /* RCW is driven by map .dat. Upper floors commonly have .dat but no .hei;
                   the client treats missing .hei as zero terrain height/colour and still
                   loads .dat walls/roofs. */
                (void)hei_name;
                if (!jag_find_by_hash_either(&maps, &maps_mem, rsc_hash_name(dat_name))) continue;

                nb_data_t nb;
                load_neighborhood(&nb, &land, &land_mem, &maps, &maps_mem, plane, x, y);
                if (!nb.c[1][1].have_dat) continue;
                seen++;

                char base[32];
                section_name(base, plane, x, y);

                if (do_walls) {
                    mesh_t mesh;
                    mesh_init(&mesh);
                    int ok = bake_walls(&mesh, &nb, &cfg, thick_walls, coplanar_backface);
                    if (ok && !mesh_batch_materials(&mesh)) {
                        fprintf(stderr, "warn: failed batching wall mesh %s\n", base);
                        ok = 0;
                    }
                    if (ok && deduplicate_vertices && !mesh_deduplicate_vertices(&mesh)) {
                        fprintf(stderr, "warn: failed deduplicating wall mesh %s\n", base);
                        ok = 0;
                    }
                    if (ok || empty_files || nb.c[1][1].have_dat) {
                        char out_path[1200];
                        snprintf(out_path, sizeof(out_path), "%s/%s.rcm", walls_dir, base);
                        if (!write_rcw1_file(out_path, &mesh, empty_files || nb.c[1][1].have_dat,
                                             &nb, &cfg, plane, x, y)) {
                            fprintf(stderr, "warn: failed writing %s\n", out_path);
                        } else {
                            written_walls++;
                        }
                    }
                    mesh_free(&mesh);
                }

                if (do_roofs) {
                    mesh_t mesh;
                    mesh_init(&mesh);
                    int ok = bake_roofs(&mesh, &nb, &cfg);
                    if (ok && !mesh_batch_materials(&mesh)) {
                        fprintf(stderr, "warn: failed batching roof mesh %s\n", base);
                        ok = 0;
                    }
                    if (ok && deduplicate_vertices && !mesh_deduplicate_vertices(&mesh)) {
                        fprintf(stderr, "warn: failed deduplicating roof mesh %s\n", base);
                        ok = 0;
                    }
                    if (ok || empty_files) {
                        char out_path[1200];
                        snprintf(out_path, sizeof(out_path), "%s/%s.rcm", roofs_dir, base);
                        if (!write_rcw1_file(out_path, &mesh, empty_files, NULL, NULL, plane, x, y)) {
                            fprintf(stderr, "warn: failed writing %s\n", out_path);
                        } else {
                            written_roofs++;
                        }
                    }
                    mesh_free(&mesh);
                }
            }
        }
    }

    fprintf(stderr, "info: considered %d sections; wrote %d wall RCW and %d roof RCW files\n",
            seen, written_walls, written_roofs);

    jag_free(&land);
    jag_free(&maps);
    jag_free(&land_mem);
    jag_free(&maps_mem);
    config_free(&cfg);
    return 0;
}
