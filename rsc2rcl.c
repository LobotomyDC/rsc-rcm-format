// rsc2rcl.c - RuneCast landscape converter (HEI/DAT -> RCL1)
// Based on the rsc2rcm.c utility (JAG + bz2 handling), but outputs RCL1 Landscape meshes.
//
// Build (Linux): gcc -std=c99 -O2 -Wall -Wextra rsc2rcl.c -lbz2 -lm -o rsc2rcl
//
// Usage:
//   ./rsc2rcl land63.jag out_dir [options]

// Options:
//   --deduplicate-vertices
//   --no-uvs
//   --merge-faces
//   --darken
//   --land-mem <land63.mem>
//   --maps <maps63.jag>
//   --maps-mem <maps63.mem>
//   --no-trim-dat-only
//
// Notes (v1):
// - This converter emits one RCL1 mesh for each landscape/map section that exists as either
//   .hei terrain data or .dat map data. DAT-only upper/interior floors are emitted as
//   trimmed flat zero-height chunks by default, preserving enclosed overlay==0 base-floor
//   cells while discarding exterior void. Transparent HOLE tiles remain holes; use
//   --no-trim-dat-only for full slabs.
// - Per-vertex colour is baked from the same terrain colour ramp used by the client.
// - Planes 1 and 2 treat undecorated/base terrain as transparent, matching the runtime.
// - UVs, when enabled, are the per-tile 0..1 corners in Q0.15 (matching RCM). They are
//   currently unused because the output is untextured.
// - Geometry uses 47x47 tiles from a 48x48 vertex lattice (matching the client's 2x2 chunk
//   assembly into 96x96 vertices -> 95x95 tiles rendered).

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <errno.h>
#include <math.h>

#include <sys/stat.h>
#include <sys/types.h>

#include <bzlib.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define REGION_SIZE      48
#define TILE_COUNT       (REGION_SIZE * REGION_SIZE)   // 2304
#define TILE_SIZE        128
#define VERTEX_SCALE_F   100.0f
#define UV_Q15_DIVISOR   32767
#define RCM_FILE_ALIGN   32u

#define RLM_PICK_SIZE    47
#define RLM_PICK_COUNT   (RLM_PICK_SIZE * RLM_PICK_SIZE)
#define RLM_PICK_BYTES   ((RLM_PICK_COUNT + 7) / 8)
#define RLM1_HEADER_SIZE 32u
#define RLM1_BLOCK_SIZE  (RLM1_HEADER_SIZE + RLM_PICK_BYTES + TILE_COUNT)

#define RCL1_FLAG_HAS_RLM       (1u << 0)
#define RCL1_RLM_RESERVED_MAGIC 0x314d4c52u /* 'RLM1' little-endian */

#define RLM1_FLAGS_HAS_PICK_MASK  (1u << 0)
#define RLM1_FLAGS_HAS_HEIGHTS    (1u << 1)
#define RLM1_FLAGS_HAS_CONFIG     (1u << 2)
#define RLM1_FLAGS_DAT_ONLY       (1u << 3)
#define RLM1_FLAGS_HAS_DAT        (1u << 4)
#define RLM1_FLAGS_HAS_HEI        (1u << 5)

#define COLOUR_TRANSPARENT ((int16_t)0x7fff)
#define RCM_SUBMESH_TEX_NONE 0xFFFFu

enum {
    RCL_FLOOR_TILE_TYPE = 2,
    RCL_LIQUID_TILE_TYPE = 3,
    RCL_BRIDGE_TILE_TYPE = 4,
    RCL_HOLE_TILE_TYPE = 5
};

/* RCL1 terrain origin bias (tile units).
   RuneScape's stitched terrain effectively behaves as if the visible mesh origin
   is biased by one tile relative to the chunk-local (0,0) sample. */
#ifndef RCL_TERRAIN_BIAS_X_TILES
#define RCL_TERRAIN_BIAS_X_TILES (0)
#endif
#ifndef RCL_TERRAIN_BIAS_Z_TILES
#define RCL_TERRAIN_BIAS_Z_TILES (0)
#endif

enum {
    RCM_SM_ALPHA   = 1u << 0,
    RCM_SM_GOURAUD = 1u << 1,
};

/* -------------------- Helpers -------------------- */

static inline float rcl_world_x(int vx) {
    return (float)((vx + RCL_TERRAIN_BIAS_X_TILES) * TILE_SIZE) / VERTEX_SCALE_F;
}
static inline float rcl_world_z(int vy) {
    return (float)((vy + RCL_TERRAIN_BIAS_Z_TILES) * TILE_SIZE) / VERTEX_SCALE_F;
}

static void die(const char *msg) {
    fprintf(stderr, "fatal: %s\n", msg);
    exit(1);
}

static size_t align_up_sz(size_t v, size_t a) {
    return (v + (a - 1u)) & ~(a - 1u);
}

static uint32_t read_u32be(const uint8_t *b, size_t off, size_t len) {
    if (off + 4 > len) die("read_u32be out of range");
    return ((uint32_t)b[off] << 24) | ((uint32_t)b[off+1] << 16) | ((uint32_t)b[off+2] << 8) | (uint32_t)b[off+3];
}

static uint32_t read_u24be(const uint8_t *b, size_t off, size_t len) {
    if (off + 3 > len) die("read_u24be out of range");
    return ((uint32_t)b[off] << 16) | ((uint32_t)b[off+1] << 8) | (uint32_t)b[off+2];
}

static void snorm8_from_float3(int8_t out[3], float x, float y, float z);

static int ensure_dir(const char *path) {
#ifdef _WIN32
    (void)path;
    return 0;
#else
    struct stat st;
    if (stat(path, &st) == 0) {
        return S_ISDIR(st.st_mode) ? 0 : -1;
    }
    if (mkdir(path, 0755) != 0) return -1;
    return 0;
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

    uint8_t *buf = (uint8_t*)malloc((size_t)sz);
    if (!buf) { fclose(f); return NULL; }

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

    const size_t wrote = fwrite(data, 1, len, f);
    if (wrote != len) {
        fprintf(stderr, "write %s: short write (%zu/%zu)\n", path, wrote, len);
        fclose(f);
        return 0;
    }

    if (fclose(f) != 0) {
        fprintf(stderr, "write %s: fclose failed (%s)\n", path, strerror(errno));
        return 0;
    }

    return 1;
}

/* Same hash used by JAG directory lookup in rsc-c's utility.c */
static uint32_t rsc_hash_name(const char *name) {
    uint32_t hash = 0;
    for (const unsigned char *p = (const unsigned char *)name; *p; p++) {
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
    const uint8_t *payload;   /* points into archive blob */
} jag_entry_t;

typedef struct jag_archive {
    uint8_t *blob;
    size_t blob_len;
    jag_entry_t *entries;
    uint16_t count;
} jag_archive_t;

static uint8_t *bz2_decompress_nohdr(const uint8_t *src, uint32_t packed_len, uint32_t unpacked_len) {
    uint8_t *dst = (uint8_t*)malloc(unpacked_len);
    if (!dst) return NULL;

    /* Jagex uses raw bzip blocks without the "BZh1" header */
    uint8_t *tmp = (uint8_t*)malloc(packed_len + 4u);
    if (!tmp) { free(dst); return NULL; }
    memcpy(tmp, "BZh1", 4);
    memcpy(tmp + 4, src, packed_len);

    unsigned int dlen = (unsigned int)unpacked_len;
    int rc = BZ2_bzBuffToBuffDecompress((char*)dst, &dlen, (char*)tmp, (unsigned int)(packed_len + 4u), 0, 0);

    free(tmp);
    if (rc != BZ_OK || dlen != (unsigned int)unpacked_len) {
        free(dst);
        return NULL;
    }
    return dst;
}

static jag_archive_t jag_load(const char *path) {
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

    a.entries = (jag_entry_t*)calloc(a.count, sizeof(jag_entry_t));
    if (!a.entries) die("oom entries");

    size_t data_off = dir_len;
    for (uint16_t i = 0; i < a.count; i++) {
        const size_t eoff = 2u + (size_t)i * 10u;
        uint32_t hash = read_u32be(a.blob, eoff, a.blob_len);
        uint32_t ulen = read_u24be(a.blob, eoff + 4, a.blob_len);
        uint32_t plen = read_u24be(a.blob, eoff + 7, a.blob_len);
        if (data_off + plen > a.blob_len) die("entry payload overruns blob");

        a.entries[i].hash = hash;
        a.entries[i].unpacked_len = ulen;
        a.entries[i].packed_len = plen;
        a.entries[i].payload = a.blob + data_off;

        data_off += plen;
    }

    return a;
}

static void jag_free(jag_archive_t *a) {
    if (!a) return;
    free(a->entries);
    free(a->blob);
    memset(a, 0, sizeof(*a));
}

static const jag_entry_t *jag_find_by_hash(const jag_archive_t *a, uint32_t hash) {
    if (!a) return NULL;
    for (uint16_t i = 0; i < a->count; i++) {
        if (a->entries[i].hash == hash) return &a->entries[i];
    }
    return NULL;
}

static uint8_t *jag_extract_entry(const jag_entry_t *e, size_t *out_len) {
    if (!e) return NULL;
    if (e->packed_len == 0 || e->unpacked_len == 0) return NULL;

    uint8_t *buf = NULL;
    if (e->packed_len == e->unpacked_len) {
        buf = (uint8_t*)malloc(e->unpacked_len);
        if (!buf) return NULL;
        memcpy(buf, e->payload, e->unpacked_len);
    } else {
        buf = bz2_decompress_nohdr(e->payload, e->packed_len, e->unpacked_len);
        if (!buf) return NULL;
    }
    if (out_len) *out_len = (size_t)e->unpacked_len;
    return buf;
}

/* -------------------- Game-data tile config ------------------------------------
 *
 * We only need the TileConfig section:
 *   tile_count
 *   tiles[i].decoration (aka "fill" in the renderer; texture id if >=0, RGB if <0)
 *   tiles[i].type
 *   tiles[i].blocking
 *
 * This is parsed from config*.jag -> integer.dat + string.dat, using the same decode
 * rules as rsc-c's game-data.c.
 */

#define JAGEX_TRANSPARENT (12345678)
#define NPC_SPRITE_COUNT 12

typedef struct gd_reader {
    const uint8_t *str;
    size_t str_len;
    size_t str_off;

    const uint8_t *in;
    size_t in_len;
    size_t in_off;
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
    int v = ((int)r->in[r->in_off]     << 24) |
            ((int)r->in[r->in_off + 1] << 16) |
            ((int)r->in[r->in_off + 2] <<  8) |
            ((int)r->in[r->in_off + 3]);
    r->in_off += 4;

    /* rsc-c behaviour: values > 99999999 encode signed colours. */
    if (v > 99999999) v = 99999999 - v;
    return v;
}

static void gd_skip_string(gd_reader_t *r) {
    if (!r->str) die("gamedata: missing string.dat");
    while (r->str_off < r->str_len && r->str[r->str_off] != 0) r->str_off++;
    if (r->str_off >= r->str_len) die("gamedata: string.dat overrun (string)");
    r->str_off++; /* consume NUL */
}

typedef struct tilecfg_db {
    int loaded;
    uint16_t tile_count;         /* actual count in config */
    int16_t  fill[256];          /* tiles[i].decoration (fill) */
    uint8_t  type[256];          /* tiles[i].type */
    uint8_t  blocking[256];      /* tiles[i].blocking */
} tilecfg_db_t;

static tilecfg_db_t g_tilecfg;

static void tilecfg_reset(void) {
    memset(&g_tilecfg, 0, sizeof(g_tilecfg));
    for (int i = 0; i < 256; i++) {
        g_tilecfg.fill[i] = COLOUR_TRANSPARENT;
        g_tilecfg.type[i] = 0;
        g_tilecfg.blocking[i] = 0;
    }
}

static int tilecfg_load_from_config_jag(const char *config_jag_path) {
    tilecfg_reset();

    jag_archive_t cfg = jag_load(config_jag_path);

    size_t slen = 0, ilen = 0;
    const jag_entry_t *e_str = jag_find_by_hash(&cfg, rsc_hash_name("string.dat"));
    const jag_entry_t *e_int = jag_find_by_hash(&cfg, rsc_hash_name("integer.dat"));

    uint8_t *str_dat = jag_extract_entry(e_str, &slen);
    uint8_t *int_dat = jag_extract_entry(e_int, &ilen);

    jag_free(&cfg);

    if (!str_dat || !int_dat) {
        free(str_dat);
        free(int_dat);
        fprintf(stderr, "warn: --config provided but missing string.dat/integer.dat\n");
        return 0;
    }

    gd_reader_t r;
    memset(&r, 0, sizeof(r));
    r.str = str_dat; r.str_len = slen; r.str_off = 0;
    r.in  = int_dat; r.in_len  = ilen; r.in_off  = 0;

    /* --- Mirror rsc-c's game_data_load_data() up to TileConfig --- */

    int item_count = gd_u16(&r);
    for (int i = 0; i < item_count; i++) gd_skip_string(&r); /* name */
    for (int i = 0; i < item_count; i++) gd_skip_string(&r); /* desc */
    for (int i = 0; i < item_count; i++) gd_skip_string(&r); /* command */
    for (int i = 0; i < item_count; i++) (void)gd_u16(&r);   /* sprite */
    for (int i = 0; i < item_count; i++) (void)gd_u32_fill(&r); /* base_price */
    for (int i = 0; i < item_count; i++) (void)gd_u8(&r);    /* stackable */
    for (int i = 0; i < item_count; i++) (void)gd_u8(&r);    /* unused */
    for (int i = 0; i < item_count; i++) (void)gd_u16(&r);   /* wearable */
    for (int i = 0; i < item_count; i++) (void)gd_u32_fill(&r); /* mask */
    for (int i = 0; i < item_count; i++) (void)gd_u8(&r);    /* special */
    for (int i = 0; i < item_count; i++) (void)gd_u8(&r);    /* members */

    int npc_count = gd_u16(&r);
    for (int i = 0; i < npc_count; i++) gd_skip_string(&r);  /* name */
    for (int i = 0; i < npc_count; i++) gd_skip_string(&r);  /* desc */
    for (int i = 0; i < npc_count; i++) (void)gd_u8(&r);     /* attack */
    for (int i = 0; i < npc_count; i++) (void)gd_u8(&r);     /* strength */
    for (int i = 0; i < npc_count; i++) (void)gd_u8(&r);     /* hits */
    for (int i = 0; i < npc_count; i++) (void)gd_u8(&r);     /* defense */
    for (int i = 0; i < npc_count; i++) (void)gd_u8(&r);     /* attackable */
    for (int i = 0; i < npc_count; i++) {
        for (int j = 0; j < NPC_SPRITE_COUNT; j++) (void)gd_u8(&r); /* sprites */
    }
    for (int i = 0; i < npc_count; i++) (void)gd_u32_fill(&r); /* hair */
    for (int i = 0; i < npc_count; i++) (void)gd_u32_fill(&r); /* top */
    for (int i = 0; i < npc_count; i++) (void)gd_u32_fill(&r); /* bottom */
    for (int i = 0; i < npc_count; i++) (void)gd_u32_fill(&r); /* skin */
    for (int i = 0; i < npc_count; i++) (void)gd_u16(&r);      /* width */
    for (int i = 0; i < npc_count; i++) (void)gd_u16(&r);      /* height */
    for (int i = 0; i < npc_count; i++) (void)gd_u8(&r);       /* walk_speed */
    for (int i = 0; i < npc_count; i++) (void)gd_u8(&r);       /* combat_speed */
    for (int i = 0; i < npc_count; i++) (void)gd_u8(&r);       /* combat_width */
    for (int i = 0; i < npc_count; i++) gd_skip_string(&r);    /* command */

    int texture_count = gd_u16(&r);
    for (int i = 0; i < texture_count; i++) gd_skip_string(&r); /* name */
    for (int i = 0; i < texture_count; i++) gd_skip_string(&r); /* subtype */

    int anim_count = gd_u16(&r);
    for (int i = 0; i < anim_count; i++) gd_skip_string(&r);    /* name */
    for (int i = 0; i < anim_count; i++) (void)gd_u32_fill(&r); /* colour */
    for (int i = 0; i < anim_count; i++) (void)gd_u8(&r);       /* gender */
    for (int i = 0; i < anim_count; i++) (void)gd_u8(&r);       /* has_a */
    for (int i = 0; i < anim_count; i++) (void)gd_u8(&r);       /* has_f */
    for (int i = 0; i < anim_count; i++) (void)gd_u8(&r);       /* file_id */

    int object_count = gd_u16(&r);
    for (int i = 0; i < object_count; i++) gd_skip_string(&r);  /* name */
    for (int i = 0; i < object_count; i++) gd_skip_string(&r);  /* desc */
    for (int i = 0; i < object_count; i++) gd_skip_string(&r);  /* cmd1 */
    for (int i = 0; i < object_count; i++) gd_skip_string(&r);  /* cmd2 */
    for (int i = 0; i < object_count; i++) gd_skip_string(&r);  /* model_name */
    for (int i = 0; i < object_count; i++) (void)gd_u8(&r);     /* width */
    for (int i = 0; i < object_count; i++) (void)gd_u8(&r);     /* height */
    for (int i = 0; i < object_count; i++) (void)gd_u8(&r);     /* type */
    for (int i = 0; i < object_count; i++) (void)gd_u8(&r);     /* elevation */

    int wall_object_count = gd_u16(&r);
    for (int i = 0; i < wall_object_count; i++) gd_skip_string(&r); /* name */
    for (int i = 0; i < wall_object_count; i++) gd_skip_string(&r); /* desc */
    for (int i = 0; i < wall_object_count; i++) gd_skip_string(&r); /* cmd1 */
    for (int i = 0; i < wall_object_count; i++) gd_skip_string(&r); /* cmd2 */
    for (int i = 0; i < wall_object_count; i++) (void)gd_u16(&r);   /* height */
    for (int i = 0; i < wall_object_count; i++) {                  /* tex_front fill */
        int fill = gd_u32_fill(&r);
        if (fill == JAGEX_TRANSPARENT) fill = COLOUR_TRANSPARENT;
        (void)fill;
    }
    for (int i = 0; i < wall_object_count; i++) {                  /* tex_back fill */
        int fill = gd_u32_fill(&r);
        if (fill == JAGEX_TRANSPARENT) fill = COLOUR_TRANSPARENT;
        (void)fill;
    }
    for (int i = 0; i < wall_object_count; i++) (void)gd_u8(&r);   /* blocking */
    for (int i = 0; i < wall_object_count; i++) (void)gd_u8(&r);   /* interactive */

    int roof_count = gd_u16(&r);
    for (int i = 0; i < roof_count; i++) (void)gd_u8(&r);          /* height */
    for (int i = 0; i < roof_count; i++) (void)gd_u8(&r);          /* fill */

    int tile_count = gd_u16(&r);
    g_tilecfg.tile_count = (uint16_t)tile_count;

    /* tiles[i].decoration (fill) */
    for (int i = 0; i < tile_count; i++) {
        int fill = gd_u32_fill(&r);
        if (fill == JAGEX_TRANSPARENT) fill = COLOUR_TRANSPARENT;

        if (i < 256) g_tilecfg.fill[i] = (int16_t)fill;
    }

    /* tiles[i].type */
    for (int i = 0; i < tile_count; i++) {
        int v = gd_u8(&r);
        if (i < 256) g_tilecfg.type[i] = (uint8_t)v;
    }

    /* tiles[i].blocking */
    for (int i = 0; i < tile_count; i++) {
        int v = gd_u8(&r);
        if (i < 256) g_tilecfg.blocking[i] = (uint8_t)v;
    }

    g_tilecfg.loaded = 1;

    free(str_dat);
    free(int_dat);

    fprintf(stderr, "info: loaded %u TileConfig entries from %s\n",
            (unsigned)g_tilecfg.tile_count, config_jag_path);

    if (g_tilecfg.tile_count > 256) {
        fprintf(stderr, "warn: tile_count=%u > 256; only first 256 entries used\n",
                (unsigned)g_tilecfg.tile_count);
    }

    return 1;
}

static int16_t tilecfg_fill_from_overlay(uint8_t overlay) {
    if (!g_tilecfg.loaded) return COLOUR_TRANSPARENT;
    if (overlay == 0) return COLOUR_TRANSPARENT;
    const unsigned idx = (unsigned)overlay - 1u;
    if (idx >= (unsigned)g_tilecfg.tile_count) return COLOUR_TRANSPARENT;
    return g_tilecfg.fill[idx];
}


/* -------------------- RCL1 output format -------------------- */

/*
 * Header and submesh layout identical to RCM (96-byte header, 16-byte submesh).
 * Magic differs: "RCL1".
 */
typedef struct rcm_header {
    char     magic[4];          // "RCL1"
    uint16_t version;           // 1
    uint16_t flags;             // reserved

    uint32_t file_size;         // bytes incl padding to RCM_FILE_ALIGN

    uint32_t vertex_count;
    uint32_t index_count;       // number of uint16 indices
    uint32_t submesh_count;

    uint32_t submesh_off;       // aligned
    uint32_t vertex_off;        // aligned
    uint32_t index_off;         // aligned

    uint16_t vertex_stride;     // format-dependent
    uint16_t index_stride;      // 2

    uint16_t uv_divisor;        // usually 32767 (or 0 if --no-uvs)
    uint16_t reserved0;

    int16_t  aabb_min[3];
    int16_t  aabb_max[3];
    int16_t  bounds_pad;

    int16_t  pad_align0;

    uint32_t reserved1[9];
} rcm_header_t;

typedef struct rcm_submesh {
    uint16_t texture_id;       // 0xFFFF = untextured
    uint16_t color_argb1555;   // tint (unused for RCL1 v1; set 0xFFFF)
    uint16_t flags;            // RCM_SM_ALPHA, RCM_SM_GOURAUD
    uint16_t reserved0;
    uint32_t first_index;
    uint32_t index_count;
} rcm_submesh_t;

/* Vertex with UVs: 24 bytes */
typedef struct rcl_vtxf_cuv {
    float    x, y, z;          // world units (RSC / 100)
    uint32_t rgba;             // ABGR or RGBA? We'll store RGBA (0xRRGGBBAA)
    int8_t   nx, ny, nz;       // snorm8
    uint8_t  pad0;
    int16_t  u, v;             // Q0.15 in [0..uv_divisor]
} rcl_vtxf_cuv_t;

/* Vertex without UVs: 20 bytes */
typedef struct rcl_vtxf_c {
    float    x, y, z;
    uint32_t rgba;
    int8_t   nx, ny, nz;
    uint8_t  pad0;
} rcl_vtxf_c_t;


/* -------------------- Texture atlas UV mapping (matches rsc2rcm) -------------------- */

typedef struct gl_atlas_position {
    float left_u, right_u;
    float top_v, bottom_v;
} gl_atlas_position;

// Copied from rsc-c's src/gl/textures/model_textures.c.
// This MUST match the in-game atlas layout for model_textures.
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

    // Fountain special-case (which probably isn't necessary here, oh well!)
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

/* -------------------- Terrain colour ramp -------------------- */

typedef struct rgb8 { uint8_t r, g, b; } rgb8_t;
static rgb8_t g_terrain_rgb[256];


/* 50% darken in HSV "value" space (matches rsc-landscape's behaviour).
   Implemented with integer HSV to avoid any platform-dependent float quirks. */
static void darken_rgb50_hsv_u8(uint8_t *r, uint8_t *g, uint8_t *b) {
    const uint8_t R = *r, G = *g, B = *b;

    uint8_t rgb_min = R < G ? (R < B ? R : B) : (G < B ? G : B);
    uint8_t rgb_max = R > G ? (R > B ? R : B) : (G > B ? G : B);
    const uint8_t v = rgb_max;

    if (rgb_max == 0 || rgb_max == rgb_min) {
        /* Gray or black: just half v */
        const uint8_t vv = (uint8_t)(v >> 1);
        *r = vv; *g = vv; *b = vv;
        return;
    }

    const uint8_t delta = (uint8_t)(rgb_max - rgb_min);

    /* s in [0..255] */
    const uint8_t s = (uint8_t)((uint16_t)delta * 255u / rgb_max);

    /* h in [0..255] (0..6 sectors) */
    int16_t h;
    if (rgb_max == R) {
        h = (int16_t)(43 * (int16_t)(G - B) / (int16_t)delta);
    } else if (rgb_max == G) {
        h = (int16_t)(85 + 43 * (int16_t)(B - R) / (int16_t)delta);
    } else {
        h = (int16_t)(171 + 43 * (int16_t)(R - G) / (int16_t)delta);
    }
    if (h < 0) h += 256;
    const uint8_t hu = (uint8_t)h;

    /* Darken: halve V (floor) */
    const uint8_t vd = (uint8_t)(v >> 1);

    /* HSV -> RGB (integer) */
    const uint8_t region = (uint8_t)(hu / 43u);          /* 0..5 */
    const uint8_t rem    = (uint8_t)((hu - region * 43u) * 6u); /* 0..255 */

    const uint8_t p = (uint8_t)((uint16_t)vd * (255u - s) / 255u);
    const uint8_t q = (uint8_t)((uint16_t)vd * (255u - (uint16_t)s * rem / 255u) / 255u);
    const uint8_t t = (uint8_t)((uint16_t)vd * (255u - (uint16_t)s * (255u - rem) / 255u) / 255u);

    switch (region) {
        default:
        case 0: *r = vd; *g = t;  *b = p;  break;
        case 1: *r = q;  *g = vd; *b = p;  break;
        case 2: *r = p;  *g = vd; *b = t;  break;
        case 3: *r = p;  *g = q;  *b = vd; break;
        case 4: *r = t;  *g = p;  *b = vd; break;
        case 5: *r = vd; *g = p;  *b = q;  break;
    }
}
static void init_terrain_rgb(int darken) {
    for (int i = 0; i < 64; i++) {
        int r = 255 - i * 4;
        int g = 255 - (int)((double)i * 1.75);
        int b = 255 - i * 4;
        if (r < 0) r = 0;
        if (g < 0) g = 0;
        if (b < 0) b = 0;
        g_terrain_rgb[i] = (rgb8_t){ (uint8_t)r, (uint8_t)g, (uint8_t)b };
    }
    for (int i = 0; i < 64; i++) {
        int r = i * 3;
        int g = 144;
        int b = 0;
        if (r > 255) r = 255;
        g_terrain_rgb[i + 64] = (rgb8_t){ (uint8_t)r, (uint8_t)g, (uint8_t)b };
    }
    for (int i = 0; i < 64; i++) {
        int r = 192 - (int)((double)i * 1.5);
        int g = 144 - (int)((double)i * 1.5);
        int b = 0;
        if (r < 0) r = 0;
        if (g < 0) g = 0;
        g_terrain_rgb[i + 128] = (rgb8_t){ (uint8_t)r, (uint8_t)g, (uint8_t)b };
    }
    for (int i = 0; i < 64; i++) {
        int r = 96 - (int)((double)i * 1.5);
        int g = 48 + (int)((double)i * 1.5);
        int b = 0;
        if (r < 0) r = 0;
        if (g > 255) g = 255;
        g_terrain_rgb[i + 192] = (rgb8_t){ (uint8_t)r, (uint8_t)g, (uint8_t)b };
    }
    if (darken) {
        for (int i = 0; i < 256; ++i) {
            darken_rgb50_hsv_u8(&g_terrain_rgb[i].r, &g_terrain_rgb[i].g, &g_terrain_rgb[i].b);
        }
    }

}

/* -------------------- HEI decode -------------------- */

static int decode_hei(const uint8_t *data, size_t len, uint8_t out_h[TILE_COUNT], uint8_t out_c[TILE_COUNT]) {
    if (!data) return 0;
    size_t off = 0;

    /* Heights: RLE then delta integrate, matching rsc-c's world_load_section_files */
    int last = 0;
    for (int t = 0; t < TILE_COUNT; ) {
        if (off >= len) return 0;
        int v = (int)data[off++];
        if (v < 128) {
            out_h[t++] = (uint8_t)v;
            last = v;
        } else {
            int run = v - 128;
            for (int i = 0; i < run && t < TILE_COUNT; i++) {
                out_h[t++] = (uint8_t)last;
            }
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

    /* Colours */
    last = 0;
    for (int t = 0; t < TILE_COUNT; ) {
        if (off >= len) return 0;
        int v = (int)data[off++];
        if (v < 128) {
            out_c[t++] = (uint8_t)v;
            last = v;
        } else {
            int run = v - 128;
            for (int i = 0; i < run && t < TILE_COUNT; i++) {
                out_c[t++] = (uint8_t)last;
            }
        }
    }

    last = 35;
    for (int y = 0; y < REGION_SIZE; y++) {
        for (int x = 0; x < REGION_SIZE; x++) {
            const int idx = x * REGION_SIZE + y;
            last = (out_c[idx] + last) & 127;
            out_c[idx] = (uint8_t)(last * 2);
        }
    }

    return 1;
}


/* -------------------- MAP .dat decode (tile overlays + diagonal walls) -------------------- */

/*
 * maps63 .dat contains several 48x48 fields back-to-back (walls, roofs, overlays, etc).
 *
 * rsc-landscape's parser treats maps63 as "version 2":
 *  - wall streams (vertical/horizontal/diagonal) use a simple RLE for ZERO runs:
 *      byte < 128 => literal value for one tile
 *      byte >=128 => run of (byte-128) zeros
 *  - roof stream uses the same "run of zeros" scheme
 *  - tileDecoration uses "repeat last": byte < 128 => literal and becomes "last",
 *      byte >=128 => repeat last (byte-128) times
 *  - tileDirection again uses the "run of zeros" scheme
 *
 * For RCL1 we care about tileDecoration and (now) diagonal walls. We decode the diagonal
 * stream into a uint16 array where:
 *  - 0      => no diagonal wall
 *  - 1..255 => '/' diagonal (NE-SW) with raw wall id
 *  - 12000+ => '\' diagonal (NW-SE) with raw wall id (matches rsc-landscape's offset)
 *
 * Some caches/archives may store the 4 wall fields as raw TILE_COUNT bytes instead of
 * emitting zero-runs. We auto-detect by trying the RLE decoder first and falling back to
 * the raw-skip layout if it fails.
 */
#define DIAG_NW_SE_OFFSET 12000u

static int dat_rle0_skip_u8(const uint8_t *data, size_t len, size_t *off) {
    for (int t = 0; t < TILE_COUNT; ) {
        if (*off >= len) return 0;
        const int v = (int)data[(*off)++];
        if (v < 128) {
            t++;
        } else {
            t += (v - 128);
        }
    }
    return 1;
}

static int dat_rle0_decode_u8(const uint8_t *data, size_t len, size_t *off,
                             uint8_t out[TILE_COUNT]) {
    if (!out) return dat_rle0_skip_u8(data, len, off);

    for (int t = 0; t < TILE_COUNT; ) {
        if (*off >= len) return 0;
        const int v = (int)data[(*off)++];
        if (v < 128) {
            out[t++] = (uint8_t)v;
        } else {
            const int run = v - 128;
            for (int i = 0; i < run && t < TILE_COUNT; ++i) {
                out[t++] = 0;
            }
        }
    }
    return 1;
}


static int dat_rle0_decode_diag1(const uint8_t *data, size_t len, size_t *off,
                                uint16_t *out_diag) {
    for (int t = 0; t < TILE_COUNT; ) {
        if (*off >= len) return 0;
        const int v = (int)data[(*off)++];
        if (v < 128) {
            if (out_diag) out_diag[t] = (uint16_t)v;
            t++;
        } else {
            const int run = v - 128;
            if (out_diag) memset(&out_diag[t], 0, (size_t)run * sizeof(uint16_t));
            t += run;
        }
    }
    return 1;
}

static int dat_rle0_decode_diag2(const uint8_t *data, size_t len, size_t *off,
                                uint16_t *out_diag) {
    for (int t = 0; t < TILE_COUNT; ) {
        if (*off >= len) return 0;
        const int v = (int)data[(*off)++];
        if (v < 128) {
            if (out_diag) out_diag[t] = (uint16_t)(v + (int)DIAG_NW_SE_OFFSET);
            t++;
        } else {
            /* run of zeros: leave previous values intact (should already be 0 unless diag1 wrote it) */
            t += (v - 128);
        }
    }
    return 1;
}

/* Debug: dump .dat decode stats per chunk */
static int g_debug_dat = 0;
static int g_last_dat_used_raw = 0; /* 1=raw, 0=rle */

static int decode_dat_v63_try_rle(const uint8_t *data, size_t len,
                                 uint8_t out_dec[TILE_COUNT],
                                 uint16_t out_wdiag[TILE_COUNT],
                                 uint8_t out_dir[TILE_COUNT]) {
    if (!data || len == 0 || !out_dec) return 0;

    size_t off = 0;
    if (out_wdiag) memset(out_wdiag, 0, (size_t)TILE_COUNT * sizeof(uint16_t));

    /* wallsVertical, wallsHorizontal */
    if (!dat_rle0_skip_u8(data, len, &off)) return 0;
    if (!dat_rle0_skip_u8(data, len, &off)) return 0;

    /* wallsDiagonal (/) and wallsDiagonal (\) */
    if (!dat_rle0_decode_diag1(data, len, &off, out_wdiag)) return 0;
    if (!dat_rle0_decode_diag2(data, len, &off, out_wdiag)) return 0;

    /* roofs: RLE with "runs of 0" (byte >= 128 => skip that many tiles as 0) */
    if (!dat_rle0_skip_u8(data, len, &off)) return 0;

    /* tile_decoration: RLE repeat last */
    int last = 0;
    for (int t = 0; t < TILE_COUNT; ) {
        if (off >= len) return 0;
        const int v = (int)data[off++];
        if (v < 128) {
            out_dec[t++] = (uint8_t)v;
            last = v;
        } else {
            const int run = v - 128;
            for (int i = 0; i < run && t < TILE_COUNT; i++) {
                out_dec[t++] = (uint8_t)last;
            }
        }
    }

    /* tile_direction: RLE with "runs of 0" */
    if (!dat_rle0_decode_u8(data, len, &off, out_dir)) return 0;

    return 1;
}

static int decode_dat_v63_try_raw(const uint8_t *data, size_t len,
                                 uint8_t out_dec[TILE_COUNT],
                                 uint16_t out_wdiag[TILE_COUNT],
                                 uint8_t out_dir[TILE_COUNT]) {
    if (!data || len == 0 || !out_dec) return 0;

    size_t off = 0;

    /* 4 wall fields as raw TILE_COUNT bytes each */
    if (off + 4u * (size_t)TILE_COUNT > len) return 0;

    if (out_wdiag) {
        memset(out_wdiag, 0, (size_t)TILE_COUNT * sizeof(uint16_t));

        const uint8_t *diag1 = data + off + 2u * (size_t)TILE_COUNT;
        const uint8_t *diag2 = data + off + 3u * (size_t)TILE_COUNT;

        for (int i = 0; i < TILE_COUNT; i++) {
            const uint8_t v = diag1[i];
            if (v) out_wdiag[i] = (uint16_t)v;
        }
        for (int i = 0; i < TILE_COUNT; i++) {
            const uint8_t v = diag2[i];
            if (v) out_wdiag[i] = (uint16_t)(v + DIAG_NW_SE_OFFSET);
        }
    }

    off += 4u * (size_t)TILE_COUNT;

    /* roofs: RLE with "runs of 0" */
    if (!dat_rle0_skip_u8(data, len, &off)) return 0;

    /* tile_decoration: RLE repeat last */
    int last = 0;
    for (int t = 0; t < TILE_COUNT; ) {
        if (off >= len) return 0;
        const int v = (int)data[off++];
        if (v < 128) {
            out_dec[t++] = (uint8_t)v;
            last = v;
        } else {
            const int run = v - 128;
            for (int i = 0; i < run && t < TILE_COUNT; i++) {
                out_dec[t++] = (uint8_t)last;
            }
        }
    }

    /* tile_direction: RLE with "runs of 0" */
    if (!dat_rle0_decode_u8(data, len, &off, out_dir)) return 0;

    return 1;
}


static int decode_dat_v63(const uint8_t *data, size_t len,
                          uint8_t out_dec[TILE_COUNT],
                          uint16_t out_wdiag[TILE_COUNT],
                          uint8_t out_dir[TILE_COUNT]) {
    uint8_t dec_rle[TILE_COUNT];
    uint8_t dec_raw[TILE_COUNT];
    uint8_t dir_rle[TILE_COUNT];
    uint8_t dir_raw[TILE_COUNT];
    uint16_t diag_rle[TILE_COUNT];
    uint16_t diag_raw[TILE_COUNT];

    const int ok_rle = decode_dat_v63_try_rle(data, len,
                                             dec_rle,
                                             out_wdiag ? diag_rle : NULL,
                                             out_dir ? dir_rle : NULL);
    const int ok_raw = decode_dat_v63_try_raw(data, len,
                                             dec_raw,
                                             out_wdiag ? diag_raw : NULL,
                                             out_dir ? dir_raw : NULL);

    if (!ok_rle && !ok_raw) return 0;

    if (ok_rle && !ok_raw) {
        memcpy(out_dec, dec_rle, sizeof(dec_rle));
        if (out_wdiag) memcpy(out_wdiag, diag_rle, sizeof(diag_rle));
        if (out_dir)  memcpy(out_dir,  dir_rle,  sizeof(dir_rle));
        g_last_dat_used_raw = 0;
        return 1;
    }
    if (!ok_rle && ok_raw) {
        memcpy(out_dec, dec_raw, sizeof(dec_raw));
        if (out_wdiag) memcpy(out_wdiag, diag_raw, sizeof(diag_raw));
        if (out_dir)  memcpy(out_dir,  dir_raw,  sizeof(dir_raw));
        g_last_dat_used_raw = 1;
        return 1;
    }

    /* Both parsed; choose the one with fewer "impossible" overlay ids.
       tile-overlays.json in this pipeline defines ids 1..25 (0 = none).

       IMPORTANT (v63 / VERSION_MAPS > 53):
       rsc-c reads the four wall fields (including both diagonal streams) as RAW bytes
       (no RLE) for maps63. Older caches used RLE-zero runs for walls/diagonals.
       When both decoders appear plausible, prefer RAW unless it looks clearly wrong. */
    int bad_rle = 0, bad_raw = 0;
    for (int i = 0; i < TILE_COUNT; i++) {
        const int a = (int)dec_rle[i];
        const int b = (int)dec_raw[i];
        if (a != 0 && (a < 0 || a > 25)) bad_rle++;
        if (b != 0 && (b < 0 || b > 25)) bad_raw++;
    }

    int diag_nz_rle = 0, diag_nz_raw = 0;
    int diag_hi_rle = 0, diag_hi_raw = 0;
    if (out_wdiag) {
        for (int i = 0; i < TILE_COUNT; i++) {
            const uint16_t vr = diag_raw[i];
            if ((vr >= 1u && vr <= 255u) ||
                (vr > (uint16_t)DIAG_NW_SE_OFFSET && vr <= (uint16_t)(DIAG_NW_SE_OFFSET + 255u))) {
                diag_nz_raw++;
                if ((vr >= 128u && vr <= 255u) ||
                    (vr > (uint16_t)(DIAG_NW_SE_OFFSET + 127u) && vr <= (uint16_t)(DIAG_NW_SE_OFFSET + 255u))) {
                    diag_hi_raw++;
                }
            }

            const uint16_t vl = diag_rle[i];
            if ((vl >= 1u && vl <= 255u) ||
                (vl > (uint16_t)DIAG_NW_SE_OFFSET && vl <= (uint16_t)(DIAG_NW_SE_OFFSET + 255u))) {
                diag_nz_rle++;
                if ((vl >= 128u && vl <= 255u) ||
                    (vl > (uint16_t)(DIAG_NW_SE_OFFSET + 127u) && vl <= (uint16_t)(DIAG_NW_SE_OFFSET + 255u))) {
                    diag_hi_rle++;
                }
            }
        }
    }

    /* RAW mis-parse of an RLE wall stream tends to produce an implausibly large number of
       diagonal "walls" because run markers (>=128/192) are mistaken for wall ids. */
    const int raw_diag_suspicious = (out_wdiag && diag_nz_raw > (TILE_COUNT / 4));

    int use_raw = 0;

    /* If RAW exposes high-id diagonal walls (>=128) and doesn't look suspicious, that's a
       strong signal we're looking at VERSION_MAPS>53 RAW streams. */
    if (out_wdiag && !raw_diag_suspicious && (diag_hi_raw > 0) && (diag_hi_rle == 0)) {
        use_raw = 1;
    } else if (bad_raw < bad_rle) {
        use_raw = 1;
    } else if (bad_raw > bad_rle) {
        use_raw = 0;
    } else {
        /* Tie: prefer RAW for modern caches, unless RAW looks suspicious. */
        use_raw = raw_diag_suspicious ? 0 : 1;
    }

    if (use_raw) {
        memcpy(out_dec, dec_raw, sizeof(dec_raw));
        if (out_wdiag) memcpy(out_wdiag, diag_raw, sizeof(diag_raw));
        if (out_dir)  memcpy(out_dir,  dir_raw,  sizeof(dir_raw));
    } else {
        memcpy(out_dec, dec_rle, sizeof(dec_rle));
        if (out_wdiag) memcpy(out_wdiag, diag_rle, sizeof(diag_rle));
        if (out_dir)  memcpy(out_dir,  dir_rle,  sizeof(dir_rle));
    }

    g_last_dat_used_raw = use_raw ? 1 : 0;
    return 1;
}

/* Decode the first two wall streams using the same layout selected by
 * decode_dat_v63().  For VERSION_MAPS > 53 / maps63 this will normally be
 * raw TILE_COUNT-byte planes: walls_north_south followed by walls_east_west.
 * Older/RLE caches are kept functional by replaying the zero-run decoder. */
static int decode_dat_v63_walls_selected(const uint8_t *data, size_t len,
                                         uint8_t out_walls_ns[TILE_COUNT],
                                         uint8_t out_walls_ew[TILE_COUNT]) {
    if (!data || len == 0 || !out_walls_ns || !out_walls_ew) return 0;

    if (g_last_dat_used_raw) {
        if (2u * (size_t)TILE_COUNT > len) return 0;
        memcpy(out_walls_ns, data, TILE_COUNT);
        memcpy(out_walls_ew, data + TILE_COUNT, TILE_COUNT);
        return 1;
    }

    size_t off = 0;
    if (!dat_rle0_decode_u8(data, len, &off, out_walls_ns)) return 0;
    if (!dat_rle0_decode_u8(data, len, &off, out_walls_ew)) return 0;
    return 1;
}


static void dat_debug_print(const char *name,
                            const uint8_t dec[TILE_COUNT],
                            const uint16_t wdiag[TILE_COUNT],
                            const uint8_t dir[TILE_COUNT]) {
    if (!g_debug_dat) return;

    int ov_nonzero = 0, ov_bad = 0;
    for (int i = 0; i < TILE_COUNT; i++) {
        const int v = (int)dec[i];
        if (v != 0) ov_nonzero++;
        if (v != 0 && (v < 0 || v > 25)) ov_bad++;
    }

    int diag_slash = 0, diag_back = 0;
    int diag_slash_hi = 0, diag_back_hi = 0;
    int diag_other = 0;

    for (int i = 0; i < TILE_COUNT; i++) {
        const uint16_t v = wdiag ? wdiag[i] : 0;
        if (v >= 1u && v <= 255u) {
            diag_slash++;
            if (v >= 128u) diag_slash_hi++;
        } else if (v > (uint16_t)DIAG_NW_SE_OFFSET && v <= (uint16_t)(DIAG_NW_SE_OFFSET + 255u)) {
            diag_back++;
            if (v > (uint16_t)(DIAG_NW_SE_OFFSET + 127u)) diag_back_hi++;
        } else if (v != 0) {
            diag_other++;
        }
    }

    int dir_nz = 0, dir_max = 0;
    if (dir) {
        for (int i = 0; i < TILE_COUNT; i++) {
            const int v = (int)dir[i];
            if (v) dir_nz++;
            if (v > dir_max) dir_max = v;
        }
    }

    fprintf(stderr,
            "dat %s mode=%s ov_nonzero=%d ov_bad=%d diag/=%d (hi=%d) diag\\\\=%d (hi=%d) diag_other=%d dir_nz=%d dir_max=%d\n",
            name ? name : "(null)",
            g_last_dat_used_raw ? "RAW" : "RLE",
            ov_nonzero, ov_bad,
            diag_slash, diag_slash_hi,
            diag_back, diag_back_hi,
            diag_other,
            dir_nz, dir_max);
}


/* -------------------- 3x3 neighborhood sampling (Option A) -------------------- */

typedef struct rcl_nb {
    const uint8_t *h[3][3];   /* heights */
    const uint8_t *c[3][3];   /* colours (unused for now, but handy later) */
    const uint8_t *dec[3][3]; /* decorations / overlays */
} rcl_nb_t;

typedef struct rcl_nb_storage {
    uint8_t h[3][3][TILE_COUNT];
    uint8_t c[3][3][TILE_COUNT];
    uint8_t dec[3][3][TILE_COUNT];
    uint8_t have_hei[3][3];
    rcl_nb_t nb; /* points into the arrays above */
} rcl_nb_storage_t;

static const jag_entry_t *jag_find_by_hash_either(const jag_archive_t *a,
                                                  const jag_archive_t *b,
                                                  uint32_t h) {
    const jag_entry_t *e = jag_find_by_hash(a, h);
    if (!e && b && b->blob) e = jag_find_by_hash(b, h);
    return e;
}

static int load_chunk_hei(const jag_archive_t *land,
                          const jag_archive_t *land_mem,
                          int plane, int x, int y,
                          uint8_t out_h[TILE_COUNT],
                          uint8_t out_c[TILE_COUNT]) {
    char name[16];
    snprintf(name, sizeof(name), "m%d%d%d%d%d.hei", plane, x / 10, x % 10, y / 10, y % 10);
    const uint32_t hh = rsc_hash_name(name);
    const jag_entry_t *e = jag_find_by_hash_either(land, land_mem, hh);
    if (!e) return 0;

    size_t len = 0;
    uint8_t *blob = jag_extract_entry(e, &len);
    if (!blob) return 0;

    const int ok = decode_hei(blob, len, out_h, out_c);
    free(blob);
    return ok;
}

static int load_chunk_dec(const jag_archive_t *maps,
                          const jag_archive_t *maps_mem,
                          int plane, int x, int y,
                          uint8_t out_dec[TILE_COUNT]) {
    char name[16];
    snprintf(name, sizeof(name), "m%d%d%d%d%d.dat", plane, x / 10, x % 10, y / 10, y % 10);
    const uint32_t hh = rsc_hash_name(name);
    const jag_entry_t *e = jag_find_by_hash_either(maps, maps_mem, hh);
    if (!e) return 0;

    size_t len = 0;
    uint8_t *blob = jag_extract_entry(e, &len);
    if (!blob) return 0;

    const int ok = decode_dat_v63(blob, len, out_dec, NULL, NULL);
    free(blob);
    return ok;
}

static int load_chunk_dat_fields(const jag_archive_t *maps,
                                 const jag_archive_t *maps_mem,
                                 int plane, int x, int y,
                                 uint8_t out_dec[TILE_COUNT],
                                 uint16_t out_wdiag[TILE_COUNT],
                                 uint8_t out_dir[TILE_COUNT],
                                 uint8_t out_walls_ns[TILE_COUNT],
                                 uint8_t out_walls_ew[TILE_COUNT]) {
    if (!maps || !maps->blob) return 0;

    char name[16];
    snprintf(name, sizeof(name), "m%d%d%d%d%d.dat", plane, x / 10, x % 10, y / 10, y % 10);
    const uint32_t hh = rsc_hash_name(name);
    const jag_entry_t *e = jag_find_by_hash_either(maps, maps_mem, hh);
    if (!e) return 0;

    size_t len = 0;
    uint8_t *blob = jag_extract_entry(e, &len);
    if (!blob) return 0;

    uint8_t tmp_dec[TILE_COUNT];
    uint16_t tmp_diag[TILE_COUNT];
    uint8_t tmp_dir[TILE_COUNT];

    const int ok = decode_dat_v63(blob, len,
                                  out_dec ? out_dec : tmp_dec,
                                  out_wdiag ? out_wdiag : tmp_diag,
                                  out_dir ? out_dir : tmp_dir);
    if (ok && (out_walls_ns || out_walls_ew)) {
        uint8_t tmp_ns[TILE_COUNT];
        uint8_t tmp_ew[TILE_COUNT];
        if (!decode_dat_v63_walls_selected(blob, len,
                                           out_walls_ns ? out_walls_ns : tmp_ns,
                                           out_walls_ew ? out_walls_ew : tmp_ew)) {
            if (out_walls_ns) memset(out_walls_ns, 0, TILE_COUNT);
            if (out_walls_ew) memset(out_walls_ew, 0, TILE_COUNT);
        }
    }

    free(blob);
    return ok;
}

static int chunk_dat_exists(const jag_archive_t *maps,
                            const jag_archive_t *maps_mem,
                            int plane, int x, int y) {
    if (!maps || !maps->blob) return 0;
    char name[16];
    snprintf(name, sizeof(name), "m%d%d%d%d%d.dat", plane, x / 10, x % 10, y / 10, y % 10);
    const uint32_t hh = rsc_hash_name(name);
    return jag_find_by_hash_either(maps, maps_mem, hh) != NULL;
}

static void build_nb_3x3(rcl_nb_storage_t *st,
                         const jag_archive_t *land,
                         const jag_archive_t *land_mem,
                         const jag_archive_t *maps,
                         const jag_archive_t *maps_mem,
                         int with_uv,
                         int plane, int cx, int cy,
                         const uint8_t center_h[TILE_COUNT],
                         const uint8_t center_c[TILE_COUNT],
                         const uint8_t center_dec[TILE_COUNT]) {
    memset(st, 0, sizeof(*st));

    for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
            const int ix = dx + 1;
            const int iy = dy + 1;

            const int x = cx + dx;
            const int y = cy + dy;

            if (x < 0 || x > 99 || y < 0 || y > 99) {
                continue;
            }

            if (dx == 0 && dy == 0) {
                memcpy(st->h[ix][iy], center_h, TILE_COUNT);
                memcpy(st->c[ix][iy], center_c, TILE_COUNT);
                if (with_uv && center_dec) memcpy(st->dec[ix][iy], center_dec, TILE_COUNT);
                else memset(st->dec[ix][iy], 0, TILE_COUNT);
                st->have_hei[ix][iy] = 1;
                continue;
            }

            const int have_hei = load_chunk_hei(land, land_mem, plane, x, y,
                                                  st->h[ix][iy], st->c[ix][iy]);
            int have_dat = 0;

            if (maps && maps->blob) {
                if (with_uv) {
                    have_dat = load_chunk_dec(maps, maps_mem, plane, x, y, st->dec[ix][iy]);
                    if (!have_dat) memset(st->dec[ix][iy], 0, TILE_COUNT);
                } else {
                    have_dat = chunk_dat_exists(maps, maps_mem, plane, x, y);
                    memset(st->dec[ix][iy], 0, TILE_COUNT);
                }
            } else {
                memset(st->dec[ix][iy], 0, TILE_COUNT);
            }

            if (have_hei || have_dat) {
                /* DAT-only sections are valid on upper/interior planes. The original
                 * runtime loader zeroes height/colour when .hei is absent, then still
                 * decodes the .dat stream. Treat such neighbours as flat chunks so
                 * edge normals/decoration sampling do not see them as missing. */
                if (!have_hei) {
                    memset(st->h[ix][iy], 0, TILE_COUNT);
                    memset(st->c[ix][iy], 0, TILE_COUNT);
                }
                st->have_hei[ix][iy] = 1;
            }
        }
    }

    /* bind pointers */
    for (int iy = 0; iy < 3; ++iy) {
        for (int ix = 0; ix < 3; ++ix) {
            if (st->have_hei[ix][iy]) {
                st->nb.h[ix][iy]   = st->h[ix][iy];
                st->nb.c[ix][iy]   = st->c[ix][iy];
                st->nb.dec[ix][iy] = with_uv ? st->dec[ix][iy] : NULL;
            } else {
                st->nb.h[ix][iy] = NULL;
                st->nb.c[ix][iy] = NULL;
                st->nb.dec[ix][iy] = NULL;
            }
        }
    }
}

static inline uint8_t nb_h_get(const rcl_nb_t *nb, int x, int y) {
    if (!nb || !nb->h[1][1]) return 0;

    const int ox = x, oy = y;
    int dx = 0, dy = 0;

    if (x < 0) { dx = -1; x += REGION_SIZE; }
    else if (x >= REGION_SIZE) { dx = 1; x -= REGION_SIZE; }

    if (y < 0) { dy = -1; y += REGION_SIZE; }
    else if (y >= REGION_SIZE) { dy = 1; y -= REGION_SIZE; }

    const uint8_t *h = nb->h[dx + 1][dy + 1];
    if (h) return h[x * REGION_SIZE + y];

    /* missing neighbor: clamp to edge of center chunk (preserves “flat derivative” behavior) */
    const uint8_t *hc = nb->h[1][1];
    int sx = ox, sy = oy;
    if (sx < 0) sx = 0; else if (sx >= REGION_SIZE) sx = REGION_SIZE - 1;
    if (sy < 0) sy = 0; else if (sy >= REGION_SIZE) sy = REGION_SIZE - 1;
    return hc[sx * REGION_SIZE + sy];
}

static inline uint8_t nb_c_get(const rcl_nb_t *nb, int x, int y) {
    if (!nb || !nb->c[1][1]) return 0;

    const int ox = x, oy = y;
    int dx = 0, dy = 0;

    if (x < 0) { dx = -1; x += REGION_SIZE; }
    else if (x >= REGION_SIZE) { dx = 1; x -= REGION_SIZE; }

    if (y < 0) { dy = -1; y += REGION_SIZE; }
    else if (y >= REGION_SIZE) { dy = 1; y -= REGION_SIZE; }

    const uint8_t *c = nb->c[dx + 1][dy + 1];
    if (c) return c[x * REGION_SIZE + y];

    /* missing neighbor: clamp to edge of center chunk */
    const uint8_t *cc = nb->c[1][1];
    int sx = ox, sy = oy;
    if (sx < 0) sx = 0; else if (sx >= REGION_SIZE) sx = REGION_SIZE - 1;
    if (sy < 0) sy = 0; else if (sy >= REGION_SIZE) sy = REGION_SIZE - 1;
    return cc[sx * REGION_SIZE + sy];
}

static inline uint8_t nb_dec_get(const rcl_nb_t *nb, int x, int y) {
    if (!nb || !nb->dec[1][1]) return 0;

    int dx = 0, dy = 0;

    if (x < 0) { dx = -1; x += REGION_SIZE; }
    else if (x >= REGION_SIZE) { dx = 1; x -= REGION_SIZE; }

    if (y < 0) { dy = -1; y += REGION_SIZE; }
    else if (y >= REGION_SIZE) { dy = 1; y -= REGION_SIZE; }

    const uint8_t *d = nb->dec[dx + 1][dy + 1];
    if (!d) return 0;
    return d[x * REGION_SIZE + y];
}

static void compute_normals_lattice_nb(const rcl_nb_t *nb, int8_t out_n[TILE_COUNT][3]) {
    for (int y = 0; y < REGION_SIZE; y++) {
        for (int x = 0; x < REGION_SIZE; x++) {
            const int idx = x * REGION_SIZE + y;

            const float hL = (float)nb_h_get(nb, x - 1, y) * 3.0f;
            const float hR = (float)nb_h_get(nb, x + 1, y) * 3.0f;
            const float hD = (float)nb_h_get(nb, x, y - 1) * 3.0f;
            const float hU = (float)nb_h_get(nb, x, y + 1) * 3.0f;

            const float dx = (hR - hL);
            const float dz = (hU - hD);

            const float nx = -dx;
            const float ny = (float)(2 * TILE_SIZE);
            const float nz = -dz;

            snorm8_from_float3(out_n[idx], nx, ny, nz);
        }
    }
}

static inline int8_t snorm8_from_f32(float v) {
    int iv = (int)lrintf(v * 127.0f);
    if (iv < -127) iv = -127;
    if (iv > 127) iv = 127;
    return (int8_t)iv;
}

static inline void nb_normal_get(const rcl_nb_t *nb, int x, int y, int8_t out3[3]) {
    const float hL = (float)nb_h_get(nb, x - 1, y) * 3.0f;
    const float hR = (float)nb_h_get(nb, x + 1, y) * 3.0f;
    const float hD = (float)nb_h_get(nb, x, y - 1) * 3.0f;
    const float hU = (float)nb_h_get(nb, x, y + 1) * 3.0f;

    const float dx = (hR - hL);
    const float dz = (hU - hD);

    /* same shape as compute_normals_lattice_nb() */
    float nx = -dx;
    float ny = 256.0f;
    float nz = -dz;

    const float len = sqrtf(nx*nx + ny*ny + nz*nz);
    if (len > 0.0f) {
        nx /= len; ny /= len; nz /= len;
    }

    out3[0] = snorm8_from_f32(nx);
    out3[1] = snorm8_from_f32(ny);
    out3[2] = snorm8_from_f32(nz);
}

/* -------------------- Tile overlay semantics (from rsc-landscape tile-overlays.json) --------------------
 *
 * rsc-landscape stores overlay metadata + RGB swatches for the world map/minimap.
 * Those RGB values are intentionally dark (the map darkens colours ~50%), so we
 * "undim" them by scaling 2x (clamped) to better match in-game brightness.
 *
 * Flags here are used ONLY for:
 *  - deciding which overlays are textured vs solid-colour
 *  - deciding which overlays get the antialias/corner-cut treatment
 *  - special-case adjacency rules (water/lava next to bridges/logs, etc.)
 */

#define OVF_VALID      0x01
#define OVF_BLOCKED    0x02
#define OVF_BRIDGE     0x04  /* also "bridge-like" for liquid adjacency */
#define OVF_INDOORS    0x08
#define OVF_ANTIALIAS  0x10
#define OVF_LIQUID     0x20
#define OVF_FLOOR      0x40
#define OVF_TEXTURED   0x80

typedef struct overlay_def {
    uint8_t r, g, b; /* map RGB (darkened) from tile-overlays.json */
    uint8_t flags;
} overlay_def_t;

/* Overlay IDs are the raw tileDecoration bytes from the cache. */
static const overlay_def_t g_overlay_defs[256] = {
    [  1] = {  64,  64,  64, 0x11 }, /* road */
    [  2] = {  36,  64, 127, 0xB3 }, /* water */
    [  3] = { 100,  48,   2, 0xC9 }, /* brown_floor */
    [  4] = { 100,  48,   2, 0x85 }, /* bridge */
    [  5] = {  64,  64,  64, 0xC9 }, /* stone_floor */
    [  6] = { 107,   3,  15, 0xC9 }, /* maroon_floor */
    [  7] = {  34,  48,  58, 0xB3 }, /* swamp_water */
    [  8] = {   0,   0,   0, 0x03 }, /* hole */
    [  9] = {  97,  97,  97, 0x13 }, /* mountain */
    [ 10] = { 255,   0, 255, 0x03 }, /* black */
    [ 11] = {  96,  48,   1, 0xB3 }, /* lava */
    [ 12] = { 100,  48,   2, 0x85 }, /* bridge_2 */
    [ 13] = {  31,  63, 125, 0xC9 }, /* blue_floor */
    [ 14] = {  64,  64,  64, 0xC9 }, /* pentagram */
    [ 15] = {  36,  22,   9, 0xC9 }, /* purple_floor */
    [ 16] = {   0,   0,   0, 0xC9 }, /* black_floor */
    [ 17] = { 121, 123, 120, 0xC9 }, /* light_stone_floor */
    [ 18] = {   0,   0,   0, 0x01 }, /* object_platform */
    [ 19] = {   0,   0,   0, 0x03 }, /* black_2 */
    [ 20] = {   0,   0,   0, 0x85 }, /* object_platform_2 */
    [ 21] = {   0,   0,   0, 0x85 }, /* log */
    [ 23] = {  63,  33,   0, 0xC9 }, /* sand_floor */
    [ 24] = {  63,  33,   0, 0xC9 }, /* mud_floor */
    [ 25] = {  63,  33,   0, 0xC1 }, /* water_floor */
};

static inline uint8_t overlay_rgb_undim(uint8_t c) {
    unsigned v = (unsigned)c * 2u;
    return (v > 255u) ? 255u : (uint8_t)v;
}

static inline int overlay_get_rgb(uint8_t ov, uint8_t *r, uint8_t *g, uint8_t *b) {
    const overlay_def_t *d = &g_overlay_defs[ov];
    if ((d->flags & OVF_VALID) == 0) return 0;
    if (r) *r = overlay_rgb_undim(d->r);
    if (g) *g = overlay_rgb_undim(d->g);
    if (b) *b = overlay_rgb_undim(d->b);
    return 1;
}

static inline int overlay_is_indoors(uint8_t ov) {
    return (g_overlay_defs[ov].flags & OVF_INDOORS) != 0;
}

static inline int overlay_floor_class(uint8_t ov) {
    if (!ov) return -1;
    return overlay_is_indoors(ov) ? 1 : 0;
}


static inline int overlay_is_bridge_like(uint8_t ov) {
    return (g_overlay_defs[ov].flags & OVF_BRIDGE) != 0;
}

static inline uint8_t tilecfg_type_from_overlay(uint8_t overlay) {
    if (!g_tilecfg.loaded || overlay == 0) return 0;
    const unsigned idx = (unsigned)overlay - 1u;
    if (idx >= (unsigned)g_tilecfg.tile_count) return 0;
    return g_tilecfg.type[idx];
}

static inline int overlay_is_liquid_like(uint8_t ov) {
    if (ov == 0) return 0;
    if (g_tilecfg.loaded && tilecfg_type_from_overlay(ov) == RCL_LIQUID_TILE_TYPE) return 1;
    return (g_overlay_defs[ov].flags & OVF_LIQUID) != 0;
}

static inline int overlay_has_runtime_surface(uint8_t ov) {
    if (ov == 0) return 0;

    if (g_tilecfg.loaded) {
        const int16_t fill = tilecfg_fill_from_overlay(ov);
        return fill != COLOUR_TRANSPARENT;
    }

    const overlay_def_t *d = &g_overlay_defs[ov];
    return (d->flags & OVF_VALID) != 0;
}

static inline int overlay_is_hole(uint8_t ov) {
    /* rsc-c's TileConfig is authoritative here.  Several overlays can be
     * HOLE_TILE_TYPE; id 8 is the common transparent ladder/stair hole, but
     * DAT-only upper floors can use other HOLE ids as well.  Treat all hole
     * types as geometry cutouts, not black/grey quads. */
    if (ov == 0) return 0;
    if (g_tilecfg.loaded && tilecfg_type_from_overlay(ov) == 5) return 1;
    return ov == 8;
}

/* Map overlay byte -> RCM texture id.
 *
 * In practice, tileDecoration overlays share the same texture index space as model textures.
 * The in-game client uses overlay values as 1-based indices; we store 0-based atlas indices.
 */
/* Map overlay byte -> RCM texture id (atlas index).
 *
 * IMPORTANT:
 *  - The raw tileDecoration byte is NOT the atlas index.
 *  - The atlas index comes from TileConfig.decoration ("fill") in config*.jag.
 *  - RuneCast treats fill >= 0 as a texture id, fill < 0 as a packed RGB colour.
 */
static uint16_t overlay_to_texture_id(uint8_t overlay) {
    const overlay_def_t *d = &g_overlay_defs[overlay];
    if (overlay == 0) return RCM_SUBMESH_TEX_NONE;
    if ((d->flags & OVF_VALID) == 0) return RCM_SUBMESH_TEX_NONE;

    /* Prefer rsc-accurate mapping if we have it. */
    if (g_tilecfg.loaded) {
        const int16_t fill = tilecfg_fill_from_overlay(overlay);

        /* fill==COLOUR_TRANSPARENT => treat as untextured */
        if (fill >= 0 && fill != COLOUR_TRANSPARENT) {
            const uint16_t tex = (uint16_t)fill;
            return texture_id_has_atlas(tex) ? tex : RCM_SUBMESH_TEX_NONE;
        }
        return RCM_SUBMESH_TEX_NONE;
    }

    /* Fallback: legacy heuristic. */
    if ((d->flags & OVF_TEXTURED) == 0) return RCM_SUBMESH_TEX_NONE;

    const uint16_t tex = (uint16_t)(overlay - 1u);
    return texture_id_has_atlas(tex) ? tex : RCM_SUBMESH_TEX_NONE;
}

/* -------------------- Normal computation -------------------- */

static void snorm8_from_float3(int8_t out[3], float x, float y, float z) {
    /* Normalize then quantize to [-127..127] */
    const float len = sqrtf(x*x + y*y + z*z);
    float nx = x, ny = y, nz = z;
    if (len > 1e-20f) {
        nx /= len; ny /= len; nz /= len;
    } else {
        nx = 0.0f; ny = 1.0f; nz = 0.0f;
    }
    int ix = (int)lrintf(nx * 127.0f);
    int iy = (int)lrintf(ny * 127.0f);
    int iz = (int)lrintf(nz * 127.0f);
    if (ix < -127) ix = -127;
    if (ix > 127) ix = 127;
    if (iy < -127) iy = -127;
    if (iy > 127) iy = 127;
    if (iz < -127) iz = -127;
    if (iz > 127) iz = 127;
    out[0] = (int8_t)ix;
    out[1] = (int8_t)iy;
    out[2] = (int8_t)iz;
}

static void compute_normals_lattice(const uint8_t h[TILE_COUNT], int8_t out_n[TILE_COUNT][3]) {
    /* height in RSC units: (byte * 3). note: y in mesh is -height. */
    for (int y = 0; y < REGION_SIZE; y++) {
        for (int x = 0; x < REGION_SIZE; x++) {
            const int idx = x * REGION_SIZE + y;

            const int xm1 = (x > 0) ? (x - 1) : x;
            const int xp1 = (x + 1 < REGION_SIZE) ? (x + 1) : x;
            const int ym1 = (y > 0) ? (y - 1) : y;
            const int yp1 = (y + 1 < REGION_SIZE) ? (y + 1) : y;

            const int idxL = xm1 * REGION_SIZE + y;
            const int idxR = xp1 * REGION_SIZE + y;
            const int idxD = x * REGION_SIZE + ym1;
            const int idxU = x * REGION_SIZE + yp1;

            const float hL = (float)h[idxL] * 3.0f;
            const float hR = (float)h[idxR] * 3.0f;
            const float hD = (float)h[idxD] * 3.0f;
            const float hU = (float)h[idxU] * 3.0f;

            /* Derivatives in y=-height space are negated, but the normal formula below already
               matches the cross-product result for y=-h. */
            const float dx = (hR - hL);
            const float dz = (hU - hD);

            float nx = -dx;
            float ny = (float)(2 * TILE_SIZE);
            float nz = -dz;

            snorm8_from_float3(out_n[idx], nx, ny, nz);
        }
    }
}

/* -------------------- Vertex dedup map (optional) -------------------- */

typedef struct vtx_key {
    /* Packed exact bytes of whichever vertex struct we use. */
    uint32_t h;
    uint32_t off;
} vtx_key_t;

static uint32_t fnv1a32(const void *data, size_t len) {
    const uint8_t *p = (const uint8_t*)data;
    uint32_t h = 2166136261u;
    for (size_t i = 0; i < len; i++) {
        h ^= (uint32_t)p[i];
        h *= 16777619u;
    }
    return h;
}

/* A very simple open-addressing hash table for dedup; sized to next power of two. */
typedef struct vtx_table {
    uint8_t  *pool;      /* vertex bytes pool */
    size_t    stride;
    size_t    count;
    size_t    cap;       /* max vertices in pool */

    vtx_key_t *slots;
    size_t     slot_mask;
} vtx_table_t;

static size_t next_pow2(size_t v) {
    size_t p = 1;
    while (p < v) p <<= 1;
    return p;
}

static int vtx_table_init(vtx_table_t *t, size_t stride, size_t expected_vertices) {
    memset(t, 0, sizeof(*t));
    t->stride = stride;
    t->cap = expected_vertices;

    t->pool = (uint8_t*)malloc(stride * expected_vertices);
    if (!t->pool) return 0;

    const size_t slots = next_pow2(expected_vertices * 2u + 1u);
    t->slots = (vtx_key_t*)calloc(slots, sizeof(vtx_key_t));
    if (!t->slots) { free(t->pool); return 0; }
    t->slot_mask = slots - 1u;
    return 1;
}

static void vtx_table_free(vtx_table_t *t) {
    if (!t) return;
    free(t->pool);
    free(t->slots);
    memset(t, 0, sizeof(*t));
}

static int vtx_table_find_or_add(vtx_table_t *t, const void *vbytes, uint16_t *out_index) {
    if (!t || !vbytes || !out_index) return 0;

    const uint32_t h = fnv1a32(vbytes, t->stride);
    size_t pos = (size_t)h & t->slot_mask;

    for (;;) {
        vtx_key_t *s = &t->slots[pos];
        if (s->h == 0) {
            /* empty: add */
            if (t->count >= t->cap) return 0;
            const size_t off = t->count * t->stride;
            memcpy(t->pool + off, vbytes, t->stride);
            t->count++;
            s->h = h ? h : 1u;
            s->off = (uint32_t)off;
            *out_index = (uint16_t)(t->count - 1);
            return 1;
        }
        if (s->h == (h ? h : 1u)) {
            /* compare bytes */
            if (memcmp(t->pool + s->off, vbytes, t->stride) == 0) {
                *out_index = (uint16_t)(s->off / t->stride);
                return 1;
            }
        }
        pos = (pos + 1u) & t->slot_mask;
    }
}

/* -------------------- Mesh build -------------------- */

typedef struct mesh_out {
    uint8_t *vtx_bytes;
    size_t   vtx_stride;
    uint32_t vtx_count;

    uint16_t *idx;
    uint32_t  idx_count;

    rcm_submesh_t *sub;
    uint32_t       sub_count;

    float aabb_min[3];
    float aabb_max[3];
} mesh_out_t;

static void aabb_init(mesh_out_t *m) {
    m->aabb_min[0] = m->aabb_min[1] = m->aabb_min[2] =  1e30f;
    m->aabb_max[0] = m->aabb_max[1] = m->aabb_max[2] = -1e30f;
}

static void aabb_add(mesh_out_t *m, float x, float y, float z) {
    if (x < m->aabb_min[0]) m->aabb_min[0] = x;
    if (y < m->aabb_min[1]) m->aabb_min[1] = y;
    if (z < m->aabb_min[2]) m->aabb_min[2] = z;
    if (x > m->aabb_max[0]) m->aabb_max[0] = x;
    if (y > m->aabb_max[1]) m->aabb_max[1] = y;
    if (z > m->aabb_max[2]) m->aabb_max[2] = z;
}

static int host_is_little_endian(void) {
    const uint16_t x = 1;
    return *((const uint8_t *)&x) != 0;
}

static uint32_t pack_rgba_u32(uint8_t r, uint8_t g, uint8_t b, uint8_t a) {
    /*
     * IMPORTANT:
     * We want the *bytes* in the file/in memory to be RGBA in order:
     *   [0]=R [1]=G [2]=B [3]=A
     * so OpenGL's glColorPointer(..., GL_UNSIGNED_BYTE, ...) reads the expected channels
     * on little-endian targets (Linux x86 + Dreamcast SH-4).
     *
     * On little-endian, that byte layout corresponds to the u32 value 0xAABBGGRR.
     * On big-endian, it corresponds to 0xRRGGBBAA. We handle both so the output file
     * is always RGBA-bytes regardless of the converter host.
     */
    if (host_is_little_endian()) {
        return ((uint32_t)r) | ((uint32_t)g << 8) | ((uint32_t)b << 16) | ((uint32_t)a << 24);
    } else {
        return ((uint32_t)r << 24) | ((uint32_t)g << 16) | ((uint32_t)b << 8) | (uint32_t)a;
    }
}


/* -------------------- Submesh index builders -------------------- */

typedef struct sm_build {
    rcm_submesh_t sm;
    uint16_t *idx;
    uint32_t  idx_len;
    uint32_t  idx_cap;
} sm_build_t;

static sm_build_t *sm_find_or_add(sm_build_t **arr, uint32_t *count, uint32_t *cap,
                                 uint16_t texture_id) {
    if (!arr || !count || !cap) return NULL;

    for (uint32_t i = 0; i < *count; i++) {
        if ((*arr)[i].sm.texture_id == texture_id) return &(*arr)[i];
    }

    if (*count == *cap) {
        const uint32_t new_cap = (*cap == 0) ? 8u : (*cap * 2u);
        sm_build_t *n = (sm_build_t*)realloc(*arr, (size_t)new_cap * sizeof(sm_build_t));
        if (!n) return NULL;
        *arr = n;
        *cap = new_cap;
    }

    sm_build_t *sm = &(*arr)[(*count)++];
    memset(sm, 0, sizeof(*sm));
    sm->sm.texture_id = texture_id;
    sm->sm.color_argb1555 = 0xFFFFu;
    sm->sm.flags = RCM_SM_GOURAUD; /* we intend to use vertex colours */
    sm->sm.first_index = 0;
    sm->sm.index_count = 0;
    return sm;
}

static int sm_push_index(sm_build_t *sm, uint16_t idx) {
    if (!sm) return 0;
    if (sm->idx_len == sm->idx_cap) {
        const uint32_t new_cap = (sm->idx_cap == 0) ? 1024u : (sm->idx_cap * 2u);
        uint16_t *n = (uint16_t*)realloc(sm->idx, (size_t)new_cap * sizeof(uint16_t));
        if (!n) return 0;
        sm->idx = n;
        sm->idx_cap = new_cap;
    }
    sm->idx[sm->idx_len++] = idx;
    return 1;
}

static int tile_is_flat_untextured_uniform(int tx, int ty,
                                           const uint8_t h[TILE_COUNT],
                                           const uint8_t c[TILE_COUNT],
                                           const uint8_t dec[TILE_COUNT],
                                           int overlays_enabled,
                                           uint8_t *out_hv,
                                           uint8_t *out_cv) {
    if (tx < 0 || ty < 0) return 0;

    /* Merging requires the 2x2 vertex quad to exist *inside* the local 48x48 lattice.
       With 48x48 tiles enabled, the last tile row/col pulls vertices from neighbors,
       but the merge helper is intentionally local-only. */
    const int tile_w = REGION_SIZE - 1; /* 47 */
    const int tile_h = REGION_SIZE - 1; /* 47 */
    if (tx >= tile_w || ty >= tile_h) return 0;

    const int vx0 = tx;
    const int vy0 = ty;
    const int vx1 = vx0 + 1; /* guaranteed < 48 */
    const int vy1 = vy0 + 1; /* guaranteed < 48 */

    const int i00 = vx0 * REGION_SIZE + vy0;
    const int i10 = vx1 * REGION_SIZE + vy0;
    const int i01 = vx0 * REGION_SIZE + vy1;
    const int i11 = vx1 * REGION_SIZE + vy1;

    if (overlays_enabled && dec) {
        /* Do not merge decorated tiles (textured OR flat-colour overlays). */
        if (dec[i00] != 0) return 0;
    }

    const uint8_t hv = h[i00];
    if (!(h[i10] == hv && h[i01] == hv && h[i11] == hv)) return 0;

    const uint8_t cv = c[i00];
    if (!(c[i10] == cv && c[i01] == cv && c[i11] == cv)) return 0;

    if (out_hv) *out_hv = hv;
    if (out_cv) *out_cv = cv;
    return 1;
}

static inline uint8_t dec_get(const uint8_t dec[TILE_COUNT], int x, int y) {
    if (x < 0 || x >= REGION_SIZE || y < 0 || y >= REGION_SIZE) return 0;
    return dec[x * REGION_SIZE + y];
}

static int trim_overlay_is_surface_seed(uint8_t ov) {
    if (ov == 0) return 0;

    /* 250 is the client's cross-section edge sentinel.  The runtime resolves it
     * after assembling a 2x2 region; for per-section trimming, treat it as
     * occupied floor so we do not cut away the cell before runtime-equivalent
     * edge rules can matter visually. */
    if (ov == 250) return 1;

    /* Transparent ladder/stair holes are not floor surfaces. They also must not
     * be preserved later as enclosed base-floor cells; the mesh builder has an
     * explicit HOLE skip so these remain actual holes, not black quads. */
    if (overlay_is_hole(ov)) return 0;

    if ((g_overlay_defs[ov].flags & OVF_VALID) == 0) return 0;

    if (g_tilecfg.loaded) {
        const int16_t fill = tilecfg_fill_from_overlay(ov);
        return fill != COLOUR_TRANSPARENT;
    }

    return 1;
}

#define RCL_TRIM_KEEP_TRI_A       0x01u
#define RCL_TRIM_KEEP_TRI_B       0x02u
#define RCL_TRIM_KEEP_BOTH        (RCL_TRIM_KEEP_TRI_A | RCL_TRIM_KEEP_TRI_B)
#define RCL_TRIM_SPLIT_BACKSLASH  0x04u

static int trim_wall_blocks_step_ex(const uint8_t *walls_ns,
                                    const uint8_t *walls_ew,
                                    int n,
                                    int x, int y, int nx, int ny) {
    if (!walls_ns || !walls_ew) return 0;

    if (nx == x + 1 && ny == y) {
        /* Vertical/north-south walls are stored on the tile's east edge
         * (SectorPainter draws them at x+2 inside the tile).  Crossing east
         * from (x,y) is blocked by walls_ns[x,y]. */
        return (x >= 0 && x < n) ? (walls_ns[x * n + y] != 0) : 0;
    }
    if (nx == x - 1 && ny == y) {
        /* Crossing west is blocked by the west neighbour's east edge. */
        return (x - 1 >= 0) ? (walls_ns[(x - 1) * n + y] != 0) : 0;
    }
    if (nx == x && ny == y + 1) {
        return (y + 1 < n) ? (walls_ew[x * n + (y + 1)] != 0) : 0;
    }
    if (nx == x && ny == y - 1) {
        return (y >= 0) ? (walls_ew[x * n + y] != 0) : 0;
    }

    return 0;
}

static inline int trim_is_diag_slash(uint16_t v) {
    return (v >= 1u && v <= 255u);
}

static inline int trim_is_diag_backslash(uint16_t v) {
    return (v > (uint16_t)DIAG_NW_SE_OFFSET &&
            v <= (uint16_t)(DIAG_NW_SE_OFFSET + 255u));
}

static inline int trim_exterior_at(const uint8_t *exterior, int n, int x, int y) {
    if (x < 0 || x >= n || y < 0 || y >= n) return 1;
    return exterior[x * n + y] != 0;
}

/* Build a DAT-only floor footprint from a 3x3 section neighbourhood.
 *
 * The earlier trim used the current 48x48 section edge as the exterior seed.
 * That incorrectly deleted valid overlay==0 base-floor rows when an upstairs
 * room straddled a section boundary.  This version flood-fills from outside a
 * 3x3 neighbourhood and then extracts the centre section, so section-edge
 * interiors are not mistaken for outside void.
 *
 * The output is a per-tile triangle bitmask, not just a boolean.  Diagonal
 * walls can cut only one half of a tile, which is required for hexagonal and
 * 45-degree building outlines. */
static void __attribute__((unused)) build_dat_only_trim_keep_mask(const jag_archive_t *maps,
                                          const jag_archive_t *maps_mem,
                                          int plane, int cx, int cy,
                                          uint8_t keep[TILE_COUNT]) {
    enum { N = REGION_SIZE * 3 };

    uint8_t *dec = (uint8_t *)calloc((size_t)N * (size_t)N, 1);
    uint8_t *walls_ns = (uint8_t *)calloc((size_t)N * (size_t)N, 1);
    uint8_t *walls_ew = (uint8_t *)calloc((size_t)N * (size_t)N, 1);
    uint16_t *wdiag = (uint16_t *)calloc((size_t)N * (size_t)N, sizeof(uint16_t));
    uint8_t *surface = (uint8_t *)calloc((size_t)N * (size_t)N, 1);
    uint8_t *exterior = (uint8_t *)calloc((size_t)N * (size_t)N, 1);
    int *queue = (int *)malloc((size_t)N * (size_t)N * sizeof(int));

    memset(keep, 0, TILE_COUNT);

    if (!dec || !walls_ns || !walls_ew || !wdiag || !surface || !exterior || !queue) {
        free(dec); free(walls_ns); free(walls_ew); free(wdiag);
        free(surface); free(exterior); free(queue);
        return;
    }

    for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
            const int sx = cx + dx;
            const int sy = cy + dy;
            if (sx < 0 || sx > 99 || sy < 0 || sy > 99) continue;

            uint8_t cdec[TILE_COUNT];
            uint8_t cns[TILE_COUNT];
            uint8_t cew[TILE_COUNT];
            uint16_t cdiag[TILE_COUNT];
            memset(cdec, 0, sizeof(cdec));
            memset(cns, 0, sizeof(cns));
            memset(cew, 0, sizeof(cew));
            memset(cdiag, 0, sizeof(cdiag));

            if (!load_chunk_dat_fields(maps, maps_mem, plane, sx, sy,
                                       cdec, cdiag, NULL, cns, cew)) {
                continue;
            }

            const int bx = (dx + 1) * REGION_SIZE;
            const int by = (dy + 1) * REGION_SIZE;
            for (int x = 0; x < REGION_SIZE; ++x) {
                for (int y = 0; y < REGION_SIZE; ++y) {
                    const int li = x * REGION_SIZE + y;
                    const int gi = (bx + x) * N + (by + y);
                    dec[gi] = cdec[li];
                    walls_ns[gi] = cns[li];
                    walls_ew[gi] = cew[li];
                    wdiag[gi] = cdiag[li];
                }
            }
        }
    }

    for (int x = 0; x < N; ++x) {
        for (int y = 0; y < N; ++y) {
            const int idx = x * N + y;
            surface[idx] = (uint8_t)(trim_overlay_is_surface_seed(dec[idx]) ||
                                      trim_is_diag_slash(wdiag[idx]) ||
                                      trim_is_diag_backslash(wdiag[idx]));
        }
    }

    int qh = 0, qt = 0;
#define PUSH_EXT_EX(_x, _y) do { \
        const int _idx = (_x) * N + (_y); \
        if (!surface[_idx] && !exterior[_idx]) { \
            exterior[_idx] = 1; \
            queue[qt++] = _idx; \
        } \
    } while (0)

    for (int x = 0; x < N; ++x) {
        PUSH_EXT_EX(x, 0);
        PUSH_EXT_EX(x, N - 1);
    }
    for (int y = 0; y < N; ++y) {
        PUSH_EXT_EX(0, y);
        PUSH_EXT_EX(N - 1, y);
    }

    while (qh < qt) {
        const int idx = queue[qh++];
        const int x = idx / N;
        const int y = idx % N;
        const int dx4[4] = { 1, -1, 0, 0 };
        const int dy4[4] = { 0, 0, 1, -1 };

        for (int d = 0; d < 4; ++d) {
            const int nx = x + dx4[d];
            const int ny = y + dy4[d];
            if (nx < 0 || nx >= N || ny < 0 || ny >= N) continue;
            if (trim_wall_blocks_step_ex(walls_ns, walls_ew, N, x, y, nx, ny)) continue;
            PUSH_EXT_EX(nx, ny);
        }
    }

#undef PUSH_EXT_EX

    for (int x = 0; x < REGION_SIZE; ++x) {
        for (int y = 0; y < REGION_SIZE; ++y) {
            const int ci = x * REGION_SIZE + y;
            const int gx = REGION_SIZE + x;
            const int gy = REGION_SIZE + y;
            const int gi = gx * N + gy;

            if (overlay_is_hole(dec[gi])) {
                keep[ci] = 0;
                continue;
            }

            uint8_t bits = (uint8_t)((surface[gi] || !exterior[gi]) ? RCL_TRIM_KEEP_BOTH : 0u);

            const uint16_t dg = wdiag[gi];
            if (bits && (trim_is_diag_slash(dg) || trim_is_diag_backslash(dg))) {
                if (trim_is_diag_slash(dg)) {
                    /* '/' split: tri A is NW half, tri B is SE half. */
                    const int nw_ext = trim_exterior_at(exterior, N, gx - 1, gy) ||
                                       trim_exterior_at(exterior, N, gx, gy - 1);
                    const int se_ext = trim_exterior_at(exterior, N, gx + 1, gy) ||
                                       trim_exterior_at(exterior, N, gx, gy + 1);
                    if (nw_ext && se_ext) bits = 0;
                    else if (nw_ext && !se_ext) bits &= (uint8_t)~RCL_TRIM_KEEP_TRI_A;
                    else if (se_ext && !nw_ext) bits &= (uint8_t)~RCL_TRIM_KEEP_TRI_B;
                } else {
                    /* '\\' split: tri A is NE half, tri B is SW half. */
                    const int ne_ext = trim_exterior_at(exterior, N, gx + 1, gy) ||
                                       trim_exterior_at(exterior, N, gx, gy - 1);
                    const int sw_ext = trim_exterior_at(exterior, N, gx - 1, gy) ||
                                       trim_exterior_at(exterior, N, gx, gy + 1);
                    if (ne_ext && sw_ext) bits = 0;
                    else if (ne_ext && !sw_ext) bits &= (uint8_t)~RCL_TRIM_KEEP_TRI_A;
                    else if (sw_ext && !ne_ext) bits &= (uint8_t)~RCL_TRIM_KEEP_TRI_B;
                }
            }

            keep[ci] = bits;
        }
    }

    /* Conservative repair for valid DAT-only interior cells that are stored as
     * tileDecoration==0.  The original DAT-only baseline emitted these as base
     * floor.  The trim pass should remove exterior slabs, but not enclosed rows
     * of plain base floor along the inside of walls.  If a zero-decoration tile
     * was marked exterior, but it is directly wall-separated from exterior and
     * touches existing kept floor, restore it.  This is deliberately local and
     * avoids the broad flood-fill overreach that created exterior grey slabs. */
    for (int x = 0; x < REGION_SIZE; ++x) {
        for (int y = 0; y < REGION_SIZE; ++y) {
            const int ci = x * REGION_SIZE + y;
            if (keep[ci] || dec[(REGION_SIZE + x) * N + (REGION_SIZE + y)] != 0) continue;

            const int gx = REGION_SIZE + x;
            const int gy = REGION_SIZE + y;
            int kept_neighbour = 0;
            int wall_to_exterior = 0;

            const int dx4[4] = { 1, -1, 0, 0 };
            const int dy4[4] = { 0, 0, 1, -1 };
            for (int d = 0; d < 4; ++d) {
                const int nx = x + dx4[d];
                const int ny = y + dy4[d];
                const int ngx = gx + dx4[d];
                const int ngy = gy + dy4[d];
                if (nx >= 0 && nx < REGION_SIZE && ny >= 0 && ny < REGION_SIZE) {
                    if (keep[nx * REGION_SIZE + ny]) kept_neighbour = 1;
                }
                if (trim_exterior_at(exterior, N, ngx, ngy) &&
                    trim_wall_blocks_step_ex(walls_ns, walls_ew, N, gx, gy, ngx, ngy)) {
                    wall_to_exterior = 1;
                }
            }

            if (kept_neighbour && wall_to_exterior) {
                keep[ci] = RCL_TRIM_KEEP_BOTH;
            }
        }
    }

    free(dec); free(walls_ns); free(walls_ew); free(wdiag);
    free(surface); free(exterior); free(queue);
}


/* Conservative DAT-only floor repair/trim.
 *
 * The broad flood-fill attempts were too ambitious: without a faithful 96x96
 * assembled-world context they could preserve exterior slabs and delete valid
 * interior halves.  DAT-only upstairs/basement floors should instead be treated
 * as explicit decoration footprints, with a small repair for zero-decoration
 * cells that are clearly part of a wall/diagonal edge strip.
 */
static int rcl_floor_overlay_can_emit(uint8_t ov) {
    if (ov == 0 || overlay_is_hole(ov)) return 0;
    return trim_overlay_is_surface_seed(ov);
}

static uint8_t rcl_neighbour_floor_overlay(const uint8_t dec[TILE_COUNT], int x, int y) {
    const int dx4[4] = { -1, 1, 0, 0 };
    const int dy4[4] = { 0, 0, -1, 1 };
    for (int i = 0; i < 4; ++i) {
        const int nx = x + dx4[i];
        const int ny = y + dy4[i];
        if (nx < 0 || nx >= REGION_SIZE || ny < 0 || ny >= REGION_SIZE) continue;
        const uint8_t ov = dec[nx * REGION_SIZE + ny];
        if (rcl_floor_overlay_can_emit(ov)) return ov;
    }
    return 0;
}

static int rcl_wall_evidence_near_tile(const uint8_t walls_ns[TILE_COUNT],
                                       const uint8_t walls_ew[TILE_COUNT],
                                       const uint16_t wdiag[TILE_COUNT],
                                       int x, int y) {
    const int i = x * REGION_SIZE + y;

    if (wdiag && (trim_is_diag_slash(wdiag[i]) || trim_is_diag_backslash(wdiag[i]))) return 1;

    if (walls_ns) {
        if (walls_ns[i]) return 1;                       /* east edge of this tile */
        if (x > 0 && walls_ns[(x - 1) * REGION_SIZE + y]) return 1; /* west edge */
    }
    if (walls_ew) {
        if (walls_ew[i]) return 1;                       /* north edge of this tile */
        if (y + 1 < REGION_SIZE && walls_ew[x * REGION_SIZE + (y + 1)]) return 1; /* south edge */
    }

    /* Diagonal edge ownership often sits one tile away from the visually empty
     * cell it bounds.  Use this only as evidence to borrow a neighbouring floor
     * overlay; it does not by itself make arbitrary exterior zero tiles survive. */
    if (wdiag) {
        for (int oy = -1; oy <= 1; ++oy) {
            for (int ox = -1; ox <= 1; ++ox) {
                const int nx = x + ox;
                const int ny = y + oy;
                if (nx < 0 || nx >= REGION_SIZE || ny < 0 || ny >= REGION_SIZE) continue;
                const uint16_t v = wdiag[nx * REGION_SIZE + ny];
                if (trim_is_diag_slash(v) || trim_is_diag_backslash(v)) return 1;
            }
        }
    }

    return 0;
}

static void __attribute__((unused)) rcl_repair_dat_only_floor_edges(uint8_t dec[TILE_COUNT],
                                            const uint8_t walls_ns[TILE_COUNT],
                                            const uint8_t walls_ew[TILE_COUNT],
                                            const uint16_t wdiag[TILE_COUNT]) {
    if (!dec) return;

    uint8_t patched[TILE_COUNT];
    memcpy(patched, dec, TILE_COUNT);

    for (int x = 0; x < REGION_SIZE; ++x) {
        for (int y = 0; y < REGION_SIZE; ++y) {
            const int i = x * REGION_SIZE + y;
            if (dec[i] != 0) continue;

            if (!rcl_wall_evidence_near_tile(walls_ns, walls_ew, wdiag, x, y)) {
                continue;
            }

            const uint8_t borrowed = rcl_neighbour_floor_overlay(dec, x, y);
            if (borrowed) patched[i] = borrowed;
        }
    }

    memcpy(dec, patched, TILE_COUNT);
}

static uint8_t rcl_runtime_decoration_at(const uint8_t dec[TILE_COUNT],
                                         const rcl_nb_t *nb, int x, int y) {
    if (nb) return nb_dec_get(nb, x, y);
    if (!dec || x < 0 || x >= REGION_SIZE || y < 0 || y >= REGION_SIZE) return 0;
    return dec[x * REGION_SIZE + y];
}

static int span_bridge_touching_vertex(const uint8_t dec[TILE_COUNT],
                                       const rcl_nb_t *nb,
                                       int vx, int vy,
                                       int *out_x, int *out_y) {
    const int tx[4] = { vx, vx - 1, vx,     vx - 1 };
    const int ty[4] = { vy, vy,     vy - 1, vy - 1 };

    for (int i = 0; i < 4; ++i) {
        const uint8_t ov = rcl_runtime_decoration_at(dec, nb, tx[i], ty[i]);
        if (overlay_is_bridge_like(ov)) {
            if (out_x) *out_x = tx[i];
            if (out_y) *out_y = ty[i];
            return 1;
        }
    }

    return 0;
}

static uint8_t span_liquid_underlay_for_tile(const uint8_t dec[TILE_COUNT],
                                               const rcl_nb_t *nb,
                                               int x, int y,
                                               float *out_y) {
    /* Bridges/docks can be several tiles wide, so the middle tiles may only
     * touch other bridge tiles.  Search outward for the nearest liquid tile and
     * borrow both its material and height for the underlay. */
    for (int radius = 1; radius <= 4; ++radius) {
        for (int dy = -radius; dy <= radius; ++dy) {
            for (int dx = -radius; dx <= radius; ++dx) {
                if (abs(dx) != radius && abs(dy) != radius) continue;

                const int tx = x + dx;
                const int ty = y + dy;
                const uint8_t ov = rcl_runtime_decoration_at(dec, nb, tx, ty);

                if (ov != 0 && overlay_is_liquid_like(ov)) {
                    if (out_y) {
                        const int vx[4] = { tx, tx + 1, tx,     tx + 1 };
                        const int vy[4] = { ty, ty,     ty + 1, ty + 1 };
                        float hsum = 0.0f;
                        int hcount = 0;

                        for (int i = 0; i < 4; ++i) {
                            if (span_bridge_touching_vertex(dec, nb, vx[i], vy[i], NULL, NULL)) {
                                continue;
                            }
                            hsum += (float)nb_h_get(nb, vx[i], vy[i]);
                            hcount++;
                        }

                        if (hcount == 0) {
                            hsum = (float)nb_h_get(nb, tx, ty) +
                                   (float)nb_h_get(nb, tx + 1, ty) +
                                   (float)nb_h_get(nb, tx, ty + 1) +
                                   (float)nb_h_get(nb, tx + 1, ty + 1);
                            hcount = 4;
                        }

                        *out_y = -(((hsum / (float)hcount) * 3.0f) / VERTEX_SCALE_F);
                    }
                    return ov;
                }
            }
        }
    }

    if (out_y) *out_y = 0.0f;
    return 2;
}

static int span_deck_for_tile(const uint8_t dec[TILE_COUNT], const rcl_nb_t *nb,
                              int x, int y, uint8_t *out_bridge_overlay) {
    const uint8_t center = rcl_runtime_decoration_at(dec, nb, x, y);
    if (overlay_is_bridge_like(center)) {
        if (out_bridge_overlay) *out_bridge_overlay = center;
        return 1;
    }

    if (center != 0 && tilecfg_type_from_overlay(center) == RCL_LIQUID_TILE_TYPE) {
        return 0;
    }

    const uint8_t south = rcl_runtime_decoration_at(dec, nb, x, y + 1);
    if (overlay_is_bridge_like(south)) {
        if (out_bridge_overlay) *out_bridge_overlay = south;
        return 1;
    }

    const uint8_t north = rcl_runtime_decoration_at(dec, nb, x, y - 1);
    if (overlay_is_bridge_like(north)) {
        if (out_bridge_overlay) *out_bridge_overlay = north;
        return 1;
    }

    const uint8_t east = rcl_runtime_decoration_at(dec, nb, x + 1, y);
    if (overlay_is_bridge_like(east)) {
        if (out_bridge_overlay) *out_bridge_overlay = east;
        return 1;
    }

    const uint8_t west = rcl_runtime_decoration_at(dec, nb, x - 1, y);
    if (overlay_is_bridge_like(west)) {
        if (out_bridge_overlay) *out_bridge_overlay = west;
        return 1;
    }

    return 0;
}

static int rcl_runtime_tile_type_class(const uint8_t dec[TILE_COUNT],
                                       const rcl_nb_t *nb, int x, int y) {
    const uint8_t ov = rcl_runtime_decoration_at(dec, nb, x, y);
    if (ov == 0) return -1;
    return tilecfg_type_from_overlay(ov) == RCL_FLOOR_TILE_TYPE ? 1 : 0;
}

static int16_t rcl_runtime_decoration_or_fill(const uint8_t dec[TILE_COUNT],
                                              const rcl_nb_t *nb,
                                              int x, int y, int16_t base_fill) {
    const uint8_t ov = rcl_runtime_decoration_at(dec, nb, x, y);
    if (ov == 0) return base_fill;
    return tilecfg_fill_from_overlay(ov);
}

static void rcl_runtime_tile_face_fills(const uint8_t dec[TILE_COUNT],
                                        const rcl_nb_t *nb,
                                        const uint16_t wdiag[TILE_COUNT],
                                        int plane, int x, int y,
                                        int *out_direction,
                                        int16_t *out_colour,
                                        int16_t *out_colour_1) {
    const int16_t base_fill = (plane == 1 || plane == 2) ? COLOUR_TRANSPARENT : 0;
    int16_t colour = base_fill;
    int16_t colour_1 = base_fill;
    int16_t colour_2 = base_fill;
    int direction = 0;

    if (dec && x >= 0 && x < REGION_SIZE && y >= 0 && y < REGION_SIZE) {
        const int i = x * REGION_SIZE + y;
        const uint8_t decoration = dec[i];

        if (decoration > 0) {
            const uint8_t tile_type = tilecfg_type_from_overlay(decoration);
            const int is_floor = rcl_runtime_tile_type_class(dec, nb, x, y);

            colour = tilecfg_fill_from_overlay(decoration);
            colour_1 = colour;

            if (tile_type == RCL_BRIDGE_TILE_TYPE) {
                colour = 1;
                colour_1 = 1;
            } else if (tile_type == RCL_HOLE_TILE_TYPE) {
                const uint16_t diagonal = wdiag ? wdiag[i] : 0;

                if (diagonal > 0 && diagonal < 24000u) {
                    if (rcl_runtime_decoration_or_fill(dec, nb, x - 1, y, colour_2) != COLOUR_TRANSPARENT &&
                        rcl_runtime_decoration_or_fill(dec, nb, x, y - 1, colour_2) != COLOUR_TRANSPARENT) {
                        colour = rcl_runtime_decoration_or_fill(dec, nb, x - 1, y, colour_2);
                        direction = 0;
                    } else if (rcl_runtime_decoration_or_fill(dec, nb, x + 1, y, colour_2) != COLOUR_TRANSPARENT &&
                               rcl_runtime_decoration_or_fill(dec, nb, x, y + 1, colour_2) != COLOUR_TRANSPARENT) {
                        colour_1 = rcl_runtime_decoration_or_fill(dec, nb, x + 1, y, colour_2);
                        direction = 0;
                    } else if (rcl_runtime_decoration_or_fill(dec, nb, x + 1, y, colour_2) != COLOUR_TRANSPARENT &&
                               rcl_runtime_decoration_or_fill(dec, nb, x, y - 1, colour_2) != COLOUR_TRANSPARENT) {
                        colour_1 = rcl_runtime_decoration_or_fill(dec, nb, x + 1, y, colour_2);
                        direction = 1;
                    } else if (rcl_runtime_decoration_or_fill(dec, nb, x - 1, y, colour_2) != COLOUR_TRANSPARENT &&
                               rcl_runtime_decoration_or_fill(dec, nb, x, y + 1, colour_2) != COLOUR_TRANSPARENT) {
                        colour = rcl_runtime_decoration_or_fill(dec, nb, x - 1, y, colour_2);
                        direction = 1;
                    }
                }
            } else if (tile_type != RCL_FLOOR_TILE_TYPE ||
                       ((wdiag ? wdiag[i] : 0) > 0 && (wdiag ? wdiag[i] : 0) < 24000u)) {
                if (rcl_runtime_tile_type_class(dec, nb, x - 1, y) != is_floor &&
                    rcl_runtime_tile_type_class(dec, nb, x, y - 1) != is_floor) {
                    colour = colour_2;
                    direction = 0;
                } else if (rcl_runtime_tile_type_class(dec, nb, x + 1, y) != is_floor &&
                           rcl_runtime_tile_type_class(dec, nb, x, y + 1) != is_floor) {
                    colour_1 = colour_2;
                    direction = 0;
                } else if (rcl_runtime_tile_type_class(dec, nb, x + 1, y) != is_floor &&
                           rcl_runtime_tile_type_class(dec, nb, x, y - 1) != is_floor) {
                    colour_1 = colour_2;
                    direction = 1;
                } else if (rcl_runtime_tile_type_class(dec, nb, x - 1, y) != is_floor &&
                           rcl_runtime_tile_type_class(dec, nb, x, y + 1) != is_floor) {
                    colour = colour_2;
                    direction = 1;
                }
            }
        }
    }

    if (out_direction) *out_direction = direction;
    if (out_colour) *out_colour = colour;
    if (out_colour_1) *out_colour_1 = colour_1;
}

static uint8_t rcl_runtime_trim_bits_from_faces(int direction, int16_t colour,
                                                int16_t colour_1) {
    uint8_t bits = 0;

    if (direction == 0) {
        if (colour != COLOUR_TRANSPARENT) bits |= RCL_TRIM_KEEP_TRI_A;
        if (colour_1 != COLOUR_TRANSPARENT) bits |= RCL_TRIM_KEEP_TRI_B;
    } else {
        bits |= RCL_TRIM_SPLIT_BACKSLASH;
        if (colour != COLOUR_TRANSPARENT) bits |= RCL_TRIM_KEEP_TRI_B;
        if (colour_1 != COLOUR_TRANSPARENT) bits |= RCL_TRIM_KEEP_TRI_A;
    }

    return bits;
}

static void build_dat_only_runtime_trim_keep(const uint8_t dec[TILE_COUNT],
                                             const uint16_t wdiag[TILE_COUNT],
                                             int plane,
                                             uint8_t keep[TILE_COUNT]) {
    memset(keep, 0, TILE_COUNT);
    if (!dec) return;

    const int16_t base_fill = (plane == 1 || plane == 2) ? COLOUR_TRANSPARENT : 0;

    for (int x = 0; x < REGION_SIZE; ++x) {
        for (int y = 0; y < REGION_SIZE; ++y) {
            const int i = x * REGION_SIZE + y;
            const uint8_t decoration = dec[i];

            int16_t colour = base_fill;
            int16_t colour_1 = base_fill;
            int16_t colour_2 = base_fill;
            int direction = 0;

            if (decoration > 0) {
                const uint8_t tile_type = tilecfg_type_from_overlay(decoration);
                const int is_floor = rcl_runtime_tile_type_class(dec, NULL, x, y);

                colour = tilecfg_fill_from_overlay(decoration);
                colour_1 = colour;

                if (tile_type == RCL_BRIDGE_TILE_TYPE) {
                    colour = 1;
                    colour_1 = 1;

                    /* Runtime special-cases BRIDGE_TILE_DECORATION to fill 31.
                     * The symbolic id is not carried here; preserving the generic
                     * bridge rule is enough for DAT-only floor footprint trimming. */
                } else if (tile_type == RCL_HOLE_TILE_TYPE) {
                    const uint16_t diagonal = wdiag ? wdiag[i] : 0;

                    if (diagonal > 0 && diagonal < 24000u) {
                        if (rcl_runtime_decoration_or_fill(dec, NULL, x - 1, y, colour_2) != COLOUR_TRANSPARENT &&
                            rcl_runtime_decoration_or_fill(dec, NULL, x, y - 1, colour_2) != COLOUR_TRANSPARENT) {
                            colour = rcl_runtime_decoration_or_fill(dec, NULL, x - 1, y, colour_2);
                            direction = 0;
                        } else if (rcl_runtime_decoration_or_fill(dec, NULL, x + 1, y, colour_2) != COLOUR_TRANSPARENT &&
                                   rcl_runtime_decoration_or_fill(dec, NULL, x, y + 1, colour_2) != COLOUR_TRANSPARENT) {
                            colour_1 = rcl_runtime_decoration_or_fill(dec, NULL, x + 1, y, colour_2);
                            direction = 0;
                        } else if (rcl_runtime_decoration_or_fill(dec, NULL, x + 1, y, colour_2) != COLOUR_TRANSPARENT &&
                                   rcl_runtime_decoration_or_fill(dec, NULL, x, y - 1, colour_2) != COLOUR_TRANSPARENT) {
                            colour_1 = rcl_runtime_decoration_or_fill(dec, NULL, x + 1, y, colour_2);
                            direction = 1;
                        } else if (rcl_runtime_decoration_or_fill(dec, NULL, x - 1, y, colour_2) != COLOUR_TRANSPARENT &&
                                   rcl_runtime_decoration_or_fill(dec, NULL, x, y + 1, colour_2) != COLOUR_TRANSPARENT) {
                            colour = rcl_runtime_decoration_or_fill(dec, NULL, x - 1, y, colour_2);
                            direction = 1;
                        }
                    }
                } else if (tile_type != RCL_FLOOR_TILE_TYPE ||
                           ((wdiag ? wdiag[i] : 0) > 0 && (wdiag ? wdiag[i] : 0) < 24000u)) {
                    if (rcl_runtime_tile_type_class(dec, NULL, x - 1, y) != is_floor &&
                        rcl_runtime_tile_type_class(dec, NULL, x, y - 1) != is_floor) {
                        colour = colour_2;
                        direction = 0;
                    } else if (rcl_runtime_tile_type_class(dec, NULL, x + 1, y) != is_floor &&
                               rcl_runtime_tile_type_class(dec, NULL, x, y + 1) != is_floor) {
                        colour_1 = colour_2;
                        direction = 0;
                    } else if (rcl_runtime_tile_type_class(dec, NULL, x + 1, y) != is_floor &&
                               rcl_runtime_tile_type_class(dec, NULL, x, y - 1) != is_floor) {
                        colour_1 = colour_2;
                        direction = 1;
                    } else if (rcl_runtime_tile_type_class(dec, NULL, x - 1, y) != is_floor &&
                               rcl_runtime_tile_type_class(dec, NULL, x, y + 1) != is_floor) {
                        colour = colour_2;
                        direction = 1;
                    }
                }
            }

            keep[i] = rcl_runtime_trim_bits_from_faces(direction, colour, colour_1);
        }
    }
}

static int build_rcl1_mesh(const uint8_t h[TILE_COUNT], const uint8_t c[TILE_COUNT], const uint8_t dec[TILE_COUNT], const uint16_t wdiag[TILE_COUNT], const uint8_t dir[TILE_COUNT],
                           const uint8_t trim_keep[TILE_COUNT],
                           int base_transparent, int with_uv, int dedup, int merge_faces,
                           int gen_spans, const rcl_nb_t *nb, mesh_out_t *out) {
    if (!out) return 0;
    memset(out, 0, sizeof(*out));
    (void)dir; /* currently unused; reserved for future tile-direction based heuristics */
    aabb_init(out);

    int8_t nrm[TILE_COUNT][3];
    if (nb) compute_normals_lattice_nb(nb, nrm);
    else    compute_normals_lattice(h, nrm); /* fallback */

    out->vtx_stride = with_uv ? sizeof(rcl_vtxf_cuv_t) : sizeof(rcl_vtxf_c_t);

    const int tile_w = REGION_SIZE;
    const int tile_h = REGION_SIZE;
    const int overlays_enabled = (dec != NULL);
    const int allow_textures = (with_uv && dec != NULL);
    const int spans = (gen_spans && allow_textures);

    /* The DAT-only trim mask may contain arbitrary holes/footprints.  Rectangular
     * face merging is intentionally disabled in that mode so we cannot merge
     * across a skipped exterior cell and recreate a slab. */
    if (trim_keep) merge_faces = 0;

    const uint32_t naive_vtx = (uint32_t)tile_w * (uint32_t)tile_h * 6u;
    
    const uint32_t base_expected_vtx = dedup
        ? (with_uv ? naive_vtx : ((uint32_t)REGION_SIZE * (uint32_t)REGION_SIZE))
        : naive_vtx;

    /* Span decks can add up to one extra quad (6 verts) per tile in the worst case. */
    const uint32_t vtx_cap = base_expected_vtx + (spans ? naive_vtx : 0u);
    const uint32_t expected_vtx = vtx_cap;

    sm_build_t *sms = NULL;
    uint32_t sm_count = 0, sm_cap = 0;

    vtx_table_t tab;
    memset(&tab, 0, sizeof(tab));

    uint8_t *vpool = NULL;
    uint32_t vcount = 0;

    if (dedup) {
        if (!vtx_table_init(&tab, out->vtx_stride, expected_vtx)) return 0;
    } else {
        vpool = (uint8_t *)malloc((size_t)out->vtx_stride * (size_t)vtx_cap);
        if (!vpool) return 0;
    }

    uint8_t *used = NULL;
    if (merge_faces) {
        used = (uint8_t *)calloc((size_t)tile_w * (size_t)tile_h, 1);
        if (!used) { vtx_table_free(&tab); free(vpool); return 0; }
    }

#define ADD_VERTEX(VPTR, OUTIDX) do { \
        if (dedup) { \
            if (!vtx_table_find_or_add(&tab, (VPTR), &(OUTIDX))) { \
                free(used); vtx_table_free(&tab); free(vpool); \
                for (uint32_t _i=0; _i<sm_count; ++_i) free(sms[_i].idx); \
                free(sms); \
                return 0; \
            } \
        } else { \
            if (vcount >= vtx_cap) { \
                free(used); free(vpool); \
                for (uint32_t _i=0; _i<sm_count; ++_i) free(sms[_i].idx); \
                free(sms); \
                return 0; \
            } \
            memcpy(vpool + (size_t)vcount * out->vtx_stride, (VPTR), out->vtx_stride); \
            (OUTIDX) = (uint16_t)vcount; \
            vcount++; \
        } \
    } while (0)


    for (int vx0 = 0; vx0 < tile_w; vx0++) {
        for (int vy0 = 0; vy0 < tile_h; vy0++) {
            if (used && used[vx0 * tile_h + vy0]) continue;

            const int tile_idx = vx0 * REGION_SIZE + vy0;

            if (trim_keep && (trim_keep[tile_idx] & RCL_TRIM_KEEP_BOTH) == 0) {
                continue;
            }

            /* Overlay 8 is a transparent floor hole/stairwell cutout.  It should
             * remove the tile entirely, both for normal HEI landscape chunks and
             * for trimmed DAT-only upper/interior floors.  Black visible floors
             * are separate overlay IDs and are not affected by overlay_is_hole(). */
            if (dec && overlays_enabled && overlay_is_hole(dec[tile_idx]) && !trim_keep) {
                continue;
            }

            const int vx1 = vx0 + 1;
            const int vy1 = vy0 + 1;

/* Heights/colours sampled through 3x3 neighborhood so x==48/y==48 pulls from neighbors. */
            const uint8_t h00 = nb ? nb_h_get(nb, vx0, vy0) : h[vx0 * REGION_SIZE + vy0];
            const uint8_t h10 = nb ? nb_h_get(nb, vx1, vy0) : h[(vx1 < REGION_SIZE ? vx1 : (REGION_SIZE-1)) * REGION_SIZE + vy0];
            const uint8_t h01 = nb ? nb_h_get(nb, vx0, vy1) : h[vx0 * REGION_SIZE + (vy1 < REGION_SIZE ? vy1 : (REGION_SIZE-1))];
            const uint8_t h11 = nb ? nb_h_get(nb, vx1, vy1) : h[(vx1 < REGION_SIZE ? vx1 : (REGION_SIZE-1)) * REGION_SIZE + (vy1 < REGION_SIZE ? vy1 : (REGION_SIZE-1))];

            const uint8_t c00 = nb ? nb_c_get(nb, vx0, vy0) : c[vx0 * REGION_SIZE + vy0];
            const uint8_t c10 = nb ? nb_c_get(nb, vx1, vy0) : c[(vx1 < REGION_SIZE ? vx1 : (REGION_SIZE-1)) * REGION_SIZE + vy0];
            const uint8_t c01 = nb ? nb_c_get(nb, vx0, vy1) : c[vx0 * REGION_SIZE + (vy1 < REGION_SIZE ? vy1 : (REGION_SIZE-1))];
            const uint8_t c11 = nb ? nb_c_get(nb, vx1, vy1) : c[(vx1 < REGION_SIZE ? vx1 : (REGION_SIZE-1)) * REGION_SIZE + (vy1 < REGION_SIZE ? vy1 : (REGION_SIZE-1))];

            /* Normals: use precomputed nrm for in-chunk vertices, compute for x==48/y==48. */
            const int i00 = vx0 * REGION_SIZE + vy0;
            const int i10 = (vx1 < REGION_SIZE) ? (vx1 * REGION_SIZE + vy0) : -1;
            const int i01 = (vy1 < REGION_SIZE) ? (vx0 * REGION_SIZE + vy1) : -1;
            const int i11 = (vx1 < REGION_SIZE && vy1 < REGION_SIZE) ? (vx1 * REGION_SIZE + vy1) : -1;

            int8_t n10tmp[3], n01tmp[3], n11tmp[3];
            const int8_t *n00p = nrm[i00];
            const int8_t *n10p = (i10 >= 0) ? nrm[i10] : (nb_normal_get(nb, vx1, vy0, n10tmp), n10tmp);
            const int8_t *n01p = (i01 >= 0) ? nrm[i01] : (nb_normal_get(nb, vx0, vy1, n01tmp), n01tmp);
            const int8_t *n11p = (i11 >= 0) ? nrm[i11] : (nb_normal_get(nb, vx1, vy1, n11tmp), n11tmp);

            uint16_t tex_id = RCM_SUBMESH_TEX_NONE;
            uint8_t  ov_raw = 0; /* raw overlay id from .dat */
            if (overlays_enabled) {
                ov_raw = dec[i00];
            }
            if (allow_textures) {
                tex_id = overlay_to_texture_id(ov_raw);
            }

            uint8_t ov_r = 0, ov_g = 0, ov_b = 0;
            const int ov_has_rgb = overlay_get_rgb(ov_raw, &ov_r, &ov_g, &ov_b);
            const uint32_t ov_rgba = ov_has_rgb ? pack_rgba_u32(ov_r, ov_g, ov_b, 255) : 0u;

            const int is_span_tile = (spans && overlay_is_bridge_like(ov_raw));
            float span_underlay_y = 0.0f;
            const uint8_t span_underlay_ov = is_span_tile
                ? span_liquid_underlay_for_tile(dec, nb, vx0, vy0, &span_underlay_y)
                : 0;
            uint16_t base_tex_id = RCM_SUBMESH_TEX_NONE;
            /* Merge only untextured flat uniform tiles */
            if (!base_transparent && !is_span_tile && used && merge_faces && base_tex_id == RCM_SUBMESH_TEX_NONE) {
                uint8_t base_h = 0, base_c = 0;
                if (tile_is_flat_untextured_uniform(vx0, vy0, h, c, dec, overlays_enabled, &base_h, &base_c)) {
                    int maxw = 1;
                    while (vx0 + maxw < tile_w) {
                        if (used[(vx0 + maxw) * tile_h + vy0]) break;
                        uint8_t th = 0, tc = 0;
                        if (!tile_is_flat_untextured_uniform(vx0 + maxw, vy0, h, c, dec, overlays_enabled, &th, &tc)) break;
                        if (th != base_h || tc != base_c) break;
                        maxw++;
                    }

                    int maxh = 1;
                    for (;;) {
                        if (vy0 + maxh >= tile_h) break;
                        int ok = 1;
                        for (int x = 0; x < maxw; ++x) {
                            if (used[(vx0 + x) * tile_h + (vy0 + maxh)]) { ok = 0; break; }
                            uint8_t th = 0, tc = 0;
                            if (!tile_is_flat_untextured_uniform(vx0 + x, vy0 + maxh, h, c, dec, overlays_enabled, &th, &tc)) { ok = 0; break; }
                            if (th != base_h || tc != base_c) { ok = 0; break; }
                        }
                        if (!ok) break;
                        maxh++;
                    }

                    for (int x = 0; x < maxw; ++x)
                        for (int y = 0; y < maxh; ++y)
                            used[(vx0 + x) * tile_h + (vy0 + y)] = 1;

                    const int rx0 = vx0;
                    const int rz0 = vy0;
                    const int rx1 = vx0 + maxw;
                    const int rz1 = vy0 + maxh;

                    const int r00 = rx0 * REGION_SIZE + rz0;
                    const int r10 = rx1 * REGION_SIZE + rz0;
                    const int r01 = rx0 * REGION_SIZE + rz1;
                    const int r11 = rx1 * REGION_SIZE + rz1;

                    const float x0 = rcl_world_x(rx0);
                    const float x1 = rcl_world_x(rx1);
                    const float z0 = rcl_world_z(rz0);
                    const float z1 = rcl_world_z(rz1);

                    const float y0 = -(float)(h[r00] * 3u) / VERTEX_SCALE_F;
                    const rgb8_t col0 = g_terrain_rgb[c[r00]];

                    const int tri_lattice[6] = { r00, r10, r01,  r01, r10, r11 };
                    const float px[6] = { x0, x1, x0,  x0, x1, x1 };
                    const float py[6] = { y0, y0, y0,  y0, y0, y0 };
                    const float pz[6] = { z0, z0, z1,  z1, z0, z1 };

                    sm_build_t *sm = sm_find_or_add(&sms, &sm_count, &sm_cap, RCM_SUBMESH_TEX_NONE);
                    if (!sm) { free(used); vtx_table_free(&tab); free(vpool); return 0; }

                    for (int k = 0; k < 6; k++) {
                        const int li = tri_lattice[k];
                        const int8_t *nn = nrm[li];

                        uint16_t vi = 0;
                        if (with_uv) {
                            rcl_vtxf_cuv_t v;
                            v.x = px[k]; v.y = py[k]; v.z = pz[k];
                            v.rgba = pack_rgba_u32(col0.r, col0.g, col0.b, 255);
                            v.nx = nn[0]; v.ny = nn[1]; v.nz = nn[2];
                            v.pad0 = 0;
                            v.u = 0; v.v = 0; /* untextured */
                            ADD_VERTEX(&v, vi);
                        } else {
                            rcl_vtxf_c_t v;
                            v.x = px[k]; v.y = py[k]; v.z = pz[k];
                            v.rgba = pack_rgba_u32(col0.r, col0.g, col0.b, 255);
                            v.nx = nn[0]; v.ny = nn[1]; v.nz = nn[2];
                            v.pad0 = 0;
                            ADD_VERTEX(&v, vi);
                        }

                        aabb_add(out, px[k], y0, pz[k]);
                        if (!sm_push_index(sm, vi)) { free(used); vtx_table_free(&tab); free(vpool); return 0; }
                    }

                    continue;
                }
            }

            /* -------- Normal single-tile emission  -------- */

            /* Base terrain per-corner colours (use nb-sampled indices so edges match neighbours). */
            const rgb8_t col00 = g_terrain_rgb[c00];
            const rgb8_t col10 = g_terrain_rgb[c10];
            const rgb8_t col01 = g_terrain_rgb[c01];
            const rgb8_t col11 = g_terrain_rgb[c11];

            const float x0 = rcl_world_x(vx0);
            const float x1 = rcl_world_x(vx1);
            const float z0 = rcl_world_z(vy0);
            const float z1 = rcl_world_z(vy1);

            const float y00_orig = -(float)(h00 * 3u) / VERTEX_SCALE_F;
            const float y10_orig = -(float)(h10 * 3u) / VERTEX_SCALE_F;
            const float y01_orig = -(float)(h01 * 3u) / VERTEX_SCALE_F;
            const float y11_orig = -(float)(h11 * 3u) / VERTEX_SCALE_F;

            float y00 = y00_orig;
            float y10 = y10_orig;
            float y01 = y01_orig;
            float y11 = y11_orig;

            if (spans) {
                int bx = 0, by = 0;
                if (span_bridge_touching_vertex(dec, nb, vx0, vy0, &bx, &by))
                    span_liquid_underlay_for_tile(dec, nb, bx, by, &y00);
                if (span_bridge_touching_vertex(dec, nb, vx1, vy0, &bx, &by))
                    span_liquid_underlay_for_tile(dec, nb, bx, by, &y10);
                if (span_bridge_touching_vertex(dec, nb, vx0, vy1, &bx, &by))
                    span_liquid_underlay_for_tile(dec, nb, bx, by, &y01);
                if (span_bridge_touching_vertex(dec, nb, vx1, vy1, &bx, &by))
                    span_liquid_underlay_for_tile(dec, nb, bx, by, &y11);
            }

            enum { C00 = 0, C10 = 1, C01 = 2, C11 = 3 };

            const float cx[4]   = { x0,  x1,  x0,  x1  };
            const float cy[4]   = { y00, y10, y01, y11 };
            const float orig_cy[4] = { y00_orig, y10_orig, y01_orig, y11_orig };
            const float cz[4]   = { z0,  z0,  z1,  z1  };
            const rgb8_t cc4[4] = { col00, col10, col01, col11 };

            /* Corner normals (handles boundary vertices at x==48 / y==48). */
            const int8_t *nn4[4]   = { n00p, n10p, n01p, n11p };

            const float cu01[4] = { 0.0f, 1.0f, 0.0f, 1.0f };
            const float cv01[4] = { 0.0f, 0.0f, 1.0f, 1.0f };

            const int overlay_has_surface =
                (overlays_enabled && overlay_has_runtime_surface(ov_raw));

            /* Default split is '/' : (00,10,01) + (01,10,11) */
            int triA[3] = { C00, C10, C01 };
            int triB[3] = { C01, C10, C11 };

            /* Base terrain is untextured; overlays may be textured. */
            uint16_t texA = base_tex_id;
            uint16_t texB = base_tex_id;
            int triA_is_overlay = 0;
            int triB_is_overlay = 0;

            /* In v63 maps, diagonal walls are encoded as:
               - 1..255        => '/' (NE->SW)
               - 12001..12255  => '\' (NW->SE) (we store diag2 as +DIAG_NW_SE_OFFSET)
               Anything else in the stream is not a usable diagonal wall for floor cutting. */
            /* DAT-only trim carries the runtime-selected split orientation.
             * Do not infer orientation again from neighbouring/raw diagonal-wall
             * ownership here; that guesses wrong for some NE/SW hex corners. */
            if (trim_keep && (trim_keep[tile_idx] & RCL_TRIM_SPLIT_BACKSLASH)) {
                triA[0] = C00; triA[1] = C10; triA[2] = C11;
                triB[0] = C00; triB[1] = C11; triB[2] = C01;
            }

            #define GET_DEC(_x, _y) (nb ? nb_dec_get(nb, (_x), (_y)) : dec_get(dec, (_x), (_y)))
            #define GET_WDIAG(_x, _y) (((_x) >= 0 && (_x) < REGION_SIZE && (_y) >= 0 && (_y) < REGION_SIZE && wdiag) ? wdiag[(_x) * REGION_SIZE + (_y)] : 0)

            if (overlay_has_surface) {
                /* By default, overlay fully covers the tile. */
                texA = tex_id; texB = tex_id;
                triA_is_overlay = 1;
                triB_is_overlay = 1;

                if (is_span_tile) {
                    if (span_underlay_ov != 0) {
                        const uint16_t underlay_tex = overlay_to_texture_id(span_underlay_ov);
                        texA = underlay_tex;
                        texB = underlay_tex;
                        triA_is_overlay = 1;
                        triB_is_overlay = 1;
                    } else {
                        texA = overlay_to_texture_id(2);
                        texB = overlay_to_texture_id(2);
                        triA_is_overlay = 1;
                        triB_is_overlay = 1;
                    }
                } else if (!trim_keep) {
                    int direction = 0;
                    int16_t colour = COLOUR_TRANSPARENT;
                    int16_t colour_1 = COLOUR_TRANSPARENT;
                    const int16_t base_fill = base_transparent ? COLOUR_TRANSPARENT : 0;

                    rcl_runtime_tile_face_fills(dec, nb, wdiag, base_transparent ? 1 : 0, vx0, vy0,
                                                &direction, &colour, &colour_1);

                    const int16_t triA_fill = (direction == 0) ? colour : colour_1;
                    const int16_t triB_fill = (direction == 0) ? colour_1 : colour;

                    if (direction == 1) {
                        triA[0] = C00; triA[1] = C10; triA[2] = C11;
                        triB[0] = C00; triB[1] = C11; triB[2] = C01;
                    }

                    if (triA_fill == base_fill) {
                        texA = base_tex_id;
                        triA_is_overlay = 0;
                    } else if (triA_fill == COLOUR_TRANSPARENT) {
                        texA = base_tex_id;
                        triA_is_overlay = 0;
                    } else {
                        texA = tex_id;
                        triA_is_overlay = 1;
                    }

                    if (triB_fill == base_fill) {
                        texB = base_tex_id;
                        triB_is_overlay = 0;
                    } else if (triB_fill == COLOUR_TRANSPARENT) {
                        texB = base_tex_id;
                        triB_is_overlay = 0;
                    } else {
                        texB = tex_id;
                        triB_is_overlay = 1;
                    }
                }
            }

            #undef GET_DEC
            #undef GET_WDIAG

            for (int t = 0; t < 2; ++t) {
                if (trim_keep) {
                    const uint8_t need = (t == 0) ? RCL_TRIM_KEEP_TRI_A : RCL_TRIM_KEEP_TRI_B;
                    if ((trim_keep[tile_idx] & need) == 0) continue;
                }

                const int *tri = (t == 0) ? triA : triB;
                const uint16_t ttex = (t == 0) ? texA : texB;
                const int tri_is_overlay = (t == 0) ? triA_is_overlay : triB_is_overlay;

                /* Trimmed DAT-only chunks still preserve overlay==0 interior
                 * base floor, but when a decorated/overlay tile was split for
                 * diagonal/AA perimeter handling, the non-overlay half is the
                 * exterior/base half.  Drop only that half. */
                if (trim_keep && ov_raw != 0 && !tri_is_overlay) {
                    continue;
                }
                if (base_transparent && !tri_is_overlay) {
                    continue;
                }

                sm_build_t *sm = sm_find_or_add(&sms, &sm_count, &sm_cap, ttex);
                if (!sm) { free(used); vtx_table_free(&tab); free(vpool); return 0; }

                for (int k = 0; k < 3; ++k) {
                    const int ci = tri[k];
                    const int8_t *nn = nn4[ci];

                    uint32_t rgba = pack_rgba_u32(cc4[ci].r, cc4[ci].g, cc4[ci].b, 255);
                    if (tri_is_overlay) {
                        /* RuneCast behaviour: textured faces are NOT tinted.
                           Only untextured overlay faces use the overlay RGB. */
                        if (ttex != RCM_SUBMESH_TEX_NONE) {
                            rgba = pack_rgba_u32(255, 255, 255, 255);
                        } else {
                            rgba = ov_has_rgb ? ov_rgba : rgba;
                        }
                    }

                    uint16_t vi = 0;
                    if (with_uv) {
                        rcl_vtxf_cuv_t v;
                        v.x = cx[ci]; v.y = cy[ci]; v.z = cz[ci];
                        v.rgba = rgba;
                        v.nx = nn[0]; v.ny = nn[1]; v.nz = nn[2];
                        v.pad0 = 0;

                        if (ttex != RCM_SUBMESH_TEX_NONE) {
                            atlas_uv_to_q15(ttex, cu01[ci], cv01[ci], &v.u, &v.v);
                        } else {
                            v.u = 0;
                            v.v = 0;
                        }

                        ADD_VERTEX(&v, vi);
                    } else {
                        rcl_vtxf_c_t v;
                        v.x = cx[ci]; v.y = cy[ci]; v.z = cz[ci];
                        v.rgba = rgba;
                        v.nx = nn[0]; v.ny = nn[1]; v.nz = nn[2];
                        v.pad0 = 0;
                        ADD_VERTEX(&v, vi);
                    }

                    aabb_add(out, cx[ci], cy[ci], cz[ci]);
                    if (!sm_push_index(sm, vi)) { free(used); vtx_table_free(&tab); free(vpool); return 0; }
                }
            }

            uint8_t span_bridge_overlay = 0;
            if (spans && span_deck_for_tile(dec, nb, vx0, vy0, &span_bridge_overlay)) {
                /* Emit the bridge/dock/platform as an additive deck pass.  Use
                 * the original terrain heights for the bridge deck; span base
                 * terrain may have been replaced with a liquid underlay. Mesh Y
                 * is -height, so a smaller Y value is visually above. */
                const float SPAN_LIFT = 0.0025f;
                const uint32_t white = pack_rgba_u32(255, 255, 255, 255);
                const int8_t nx = 0, ny = 127, nz = 0;
                const uint16_t deck_tex = overlay_to_texture_id(span_bridge_overlay);

                sm_build_t *sm = sm_find_or_add(&sms, &sm_count, &sm_cap, deck_tex);
                if (!sm) { free(used); vtx_table_free(&tab); free(vpool); return 0; }

                /* Full quad, no diagonal splitting */
                const int quad[6] = { C00, C10, C11,  C00, C11, C01 };

                for (int k = 0; k < 6; ++k) {
                    const int ci = quad[k];
                    const float deck_y = orig_cy[ci] - SPAN_LIFT;

                    uint16_t vi = 0;
                    rcl_vtxf_cuv_t v;
                    v.x = cx[ci]; v.y = deck_y; v.z = cz[ci];
                    v.rgba = white;
                    v.nx = nx; v.ny = ny; v.nz = nz;
                    v.pad0 = 0;
                    atlas_uv_to_q15(deck_tex, cu01[ci], cv01[ci], &v.u, &v.v);
                    ADD_VERTEX(&v, vi);

                    aabb_add(out, cx[ci], deck_y, cz[ci]);
                    if (!sm_push_index(sm, vi)) { free(used); vtx_table_free(&tab); free(vpool); return 0; }
                }
            }
        }
    }

#undef ADD_VERTEX
    free(used);

    if (dedup) {
        out->vtx_bytes = tab.pool;
        tab.pool = NULL;
        out->vtx_count = (uint32_t)tab.count;
        vtx_table_free(&tab);
    } else {
        out->vtx_bytes = vpool;
        out->vtx_count = vcount;
    }

    uint32_t total_idx = 0;
    for (uint32_t i = 0; i < sm_count; i++) total_idx += sms[i].idx_len;
    if (total_idx == 0) { free(sms); return 0; }

    out->idx = (uint16_t*)malloc((size_t)total_idx * sizeof(uint16_t));
    out->sub = (rcm_submesh_t*)calloc(sm_count, sizeof(rcm_submesh_t));
    if (!out->idx || !out->sub) {
        free(out->idx); free(out->sub);
        for (uint32_t i = 0; i < sm_count; i++) free(sms[i].idx);
        free(sms);
        return 0;
    }

    uint32_t cur = 0;
    uint32_t live_sm = 0;
    for (uint32_t i = 0; i < sm_count; i++) {
        if (sms[i].idx_len == 0) { free(sms[i].idx); continue; }

        sms[i].sm.first_index = cur;
        sms[i].sm.index_count = sms[i].idx_len;

        memcpy(&out->idx[cur], sms[i].idx, (size_t)sms[i].idx_len * sizeof(uint16_t));
        cur += sms[i].idx_len;

        out->sub[live_sm++] = sms[i].sm;
        free(sms[i].idx);
    }
    free(sms);

    out->idx_count = cur;
    out->sub_count = live_sm;
    return 1;
}


static void rlm_write_u16le(uint8_t *b, size_t off, uint16_t v) {
    b[off] = (uint8_t)(v & 0xffu);
    b[off + 1] = (uint8_t)(v >> 8);
}

static void rlm_write_u32le(uint8_t *b, size_t off, uint32_t v) {
    b[off] = (uint8_t)(v & 0xffu);
    b[off + 1] = (uint8_t)((v >> 8) & 0xffu);
    b[off + 2] = (uint8_t)((v >> 16) & 0xffu);
    b[off + 3] = (uint8_t)((v >> 24) & 0xffu);
}

static void rlm_bitset_set(uint8_t *bits, int idx) {
    bits[idx >> 3] |= (uint8_t)(1u << (idx & 7));
}

static int rlm_tile_is_pickable(int plane, int have_hei, int have_dat,
                                const uint8_t decoration[TILE_COUNT],
                                const uint8_t *trim_keep, int x, int y) {
    const int idx = x * REGION_SIZE + y;
    const uint8_t overlay = (have_dat && decoration) ? decoration[idx] : 0;
    const uint8_t tile_type = tilecfg_type_from_overlay(overlay);

    if (trim_keep && !trim_keep[idx]) return 0;
    if (g_tilecfg.loaded && tile_type == RCL_HOLE_TILE_TYPE) return 0;
    if (plane == 1 || plane == 2) return overlay != 0;
    if (!have_hei && !have_dat) return 0;
    return 1;
}

static void build_rlm1_block(uint8_t out[RLM1_BLOCK_SIZE], int plane, int sx, int sy,
                             int have_hei, int have_dat,
                             const uint8_t heights[TILE_COUNT],
                             const uint8_t decoration[TILE_COUNT],
                             const uint8_t *trim_keep) {
    memset(out, 0, RLM1_BLOCK_SIZE);

    uint8_t *pick = out + RLM1_HEADER_SIZE;
    uint8_t *height_out = pick + RLM_PICK_BYTES;

    for (int x = 0; x < RLM_PICK_SIZE; x++) {
        for (int y = 0; y < RLM_PICK_SIZE; y++) {
            const int pidx = x * RLM_PICK_SIZE + y;
            if (rlm_tile_is_pickable(plane, have_hei, have_dat, decoration,
                                     trim_keep, x, y)) {
                rlm_bitset_set(pick, pidx);
            }
        }
    }

    if (heights) memcpy(height_out, heights, TILE_COUNT);

    memcpy(out, "RLM1", 4);
    rlm_write_u16le(out, 4, (uint16_t)RLM1_HEADER_SIZE);
    rlm_write_u16le(out, 6, 1);
    out[8] = (uint8_t)plane;
    out[9] = (uint8_t)sx;
    out[10] = (uint8_t)sy;
    rlm_write_u16le(out, 12, RLM_PICK_SIZE);
    rlm_write_u16le(out, 14, RLM_PICK_SIZE);
    rlm_write_u16le(out, 16, REGION_SIZE);
    rlm_write_u16le(out, 18, REGION_SIZE);

    uint32_t flags = RLM1_FLAGS_HAS_PICK_MASK | RLM1_FLAGS_HAS_HEIGHTS;
    if (g_tilecfg.loaded) flags |= RLM1_FLAGS_HAS_CONFIG;
    if (!have_hei && have_dat) flags |= RLM1_FLAGS_DAT_ONLY;
    if (have_dat) flags |= RLM1_FLAGS_HAS_DAT;
    if (have_hei) flags |= RLM1_FLAGS_HAS_HEI;
    rlm_write_u32le(out, 20, flags);
    rlm_write_u32le(out, 24, RLM1_HEADER_SIZE);
    rlm_write_u32le(out, 28, RLM1_HEADER_SIZE + RLM_PICK_BYTES);
}

/* -------------------- Write RCL1 file -------------------- */

static int write_rcl1_file(const char *out_path, const mesh_out_t *mesh, int with_uv,
                           int plane, int sx, int sy, int have_hei, int have_dat,
                           const uint8_t heights[TILE_COUNT],
                           const uint8_t decoration[TILE_COUNT],
                           const uint8_t *trim_keep) {
    if (!out_path || !mesh || !mesh->vtx_bytes || !mesh->idx || !mesh->sub || mesh->sub_count == 0) return 0;

    rcm_header_t hdr;
    memset(&hdr, 0, sizeof(hdr));
    memcpy(hdr.magic, "RCL1", 4);
    hdr.version = 1;
    hdr.flags = RCL1_FLAG_HAS_RLM;

    hdr.vertex_count  = mesh->vtx_count;
    hdr.index_count   = mesh->idx_count;
    hdr.submesh_count = mesh->sub_count;

    hdr.vertex_stride = (uint16_t)mesh->vtx_stride;
    hdr.index_stride  = 2;
    hdr.uv_divisor    = with_uv ? (uint16_t)UV_Q15_DIVISOR : 0;

    hdr.submesh_off = (uint32_t)align_up_sz(sizeof(hdr), RCM_FILE_ALIGN);
    hdr.vertex_off  = (uint32_t)align_up_sz((size_t)hdr.submesh_off +
                                            (size_t)hdr.submesh_count * sizeof(rcm_submesh_t),
                                            RCM_FILE_ALIGN);
    hdr.index_off   = (uint32_t)align_up_sz((size_t)hdr.vertex_off +
                                            (size_t)hdr.vertex_count * hdr.vertex_stride,
                                            RCM_FILE_ALIGN);

    const size_t idx_bytes = (size_t)hdr.index_count * sizeof(uint16_t);
    const size_t rlm_off = align_up_sz((size_t)hdr.index_off + idx_bytes, RCM_FILE_ALIGN);
    const size_t end_off = rlm_off + RLM1_BLOCK_SIZE;
    hdr.file_size = (uint32_t)align_up_sz(end_off, RCM_FILE_ALIGN);

    hdr.reserved1[0] = RCL1_RLM_RESERVED_MAGIC;
    hdr.reserved1[1] = (uint32_t)rlm_off;
    hdr.reserved1[2] = (uint32_t)RLM1_BLOCK_SIZE;

    /* AABB stored in fixed units scaled by 100, same convention as RCM headers */
    hdr.aabb_min[0] = (int16_t)lrintf(mesh->aabb_min[0] * VERTEX_SCALE_F);
    hdr.aabb_min[1] = (int16_t)lrintf(mesh->aabb_min[1] * VERTEX_SCALE_F);
    hdr.aabb_min[2] = (int16_t)lrintf(mesh->aabb_min[2] * VERTEX_SCALE_F);
    hdr.aabb_max[0] = (int16_t)lrintf(mesh->aabb_max[0] * VERTEX_SCALE_F);
    hdr.aabb_max[1] = (int16_t)lrintf(mesh->aabb_max[1] * VERTEX_SCALE_F);
    hdr.aabb_max[2] = (int16_t)lrintf(mesh->aabb_max[2] * VERTEX_SCALE_F);

    uint8_t *blob = (uint8_t*)calloc(1, hdr.file_size);
    if (!blob) return 0;

    uint8_t rlm[RLM1_BLOCK_SIZE];
    build_rlm1_block(rlm, plane, sx, sy, have_hei, have_dat, heights,
                     decoration, trim_keep);

    memcpy(blob, &hdr, sizeof(hdr));
    memcpy(blob + hdr.submesh_off, mesh->sub, (size_t)hdr.submesh_count * sizeof(rcm_submesh_t));
    memcpy(blob + hdr.vertex_off,  mesh->vtx_bytes, (size_t)hdr.vertex_count * hdr.vertex_stride);
    memcpy(blob + hdr.index_off,   mesh->idx, idx_bytes);
    memcpy(blob + rlm_off, rlm, RLM1_BLOCK_SIZE);

    const int ok = write_entire_file(out_path, blob, hdr.file_size);
    free(blob);
    return ok;
}

/* -------------------- Main conversion loop -------------------- */

static void usage(const char *exe) {
    fprintf(stderr,
        "Usage: %s land63.jag out_dir [--maps maps63.jag] [--config config63.jag]\n"
        "              [--deduplicate-vertices] [--no-uvs] [--merge-faces] [--no-spans] [--no-trim-dat-only] [--debug-dat]\n"
        "\n"
        "Notes:\n"
        "  --config is strongly recommended for correct overlay texture_id mapping.\n"
        "\n"
        "Outputs: one RCL1 .rcm per landscape chunk found in the archive.\n"
        "Naming: mPXXYY.rcm (where source was mPXXYY.hei)\n",
        exe);
}

int main(int argc, char **argv) {
    if (argc < 3) { usage(argv[0]); return 1; }

    const char *in_path = argv[1];
    const char *out_dir = argv[2];

    const char *land_mem_path = NULL;
    const char *maps_path = NULL;
    const char *maps_mem_path = NULL;
    const char *config_path = NULL;

    int dedup = 0;
    int with_uv = 1;
    int merge_faces = 0;
    int darken = 0;
    int gen_spans = 1;
    int trim_dat_only = 1;

    for (int i = 3; i < argc; i++) {
        if (!strcmp(argv[i], "--deduplicate-vertices")) {
            dedup = 1;
        } else if (!strcmp(argv[i], "--no-uvs")) {
            with_uv = 0;
        } else if (!strcmp(argv[i], "--merge-faces")) {
            merge_faces = 1;
        } else if (!strcmp(argv[i], "--darken")) {
            darken = 1;
        } else if (!strcmp(argv[i], "--no-spans")) {
            gen_spans = 0;

        } else if (!strcmp(argv[i], "--no-trim-dat-only")) {
            trim_dat_only = 0;

        } else if (!strcmp(argv[i], "--debug-dat")) {
            g_debug_dat = 1;

        } else if (!strcmp(argv[i], "--land-mem") && i + 1 < argc) {
            land_mem_path = argv[++i];
        } else if (!strcmp(argv[i], "--maps") && i + 1 < argc) {
            maps_path = argv[++i];
        } else if (!strcmp(argv[i], "--maps-mem") && i + 1 < argc) {
            maps_mem_path = argv[++i];
        } else if (!strcmp(argv[i], "--config") && i + 1 < argc) {
            config_path = argv[++i];
} else {
            usage(argv[0]);
            return 1;
        }
    }

    if (ensure_dir(out_dir) != 0) {
        fprintf(stderr, "Failed to create output dir %s\n", out_dir);
        return 1;
    }

    init_terrain_rgb(darken);

    if (config_path) {
        (void)tilecfg_load_from_config_jag(config_path);
    } else {
        fprintf(stderr, "warn: no --config provided; overlay textures may be mismapped\n");
    }

    jag_archive_t land = jag_load(in_path);

    jag_archive_t land_mem = {0};
    jag_archive_t maps = {0};
    jag_archive_t maps_mem = {0};

    if (land_mem_path) {
        land_mem = jag_load(land_mem_path);
        if (!land_mem.blob) {
            fprintf(stderr, "warn: failed to load land mem pack: %s\n", land_mem_path);
        }
    }
    if (maps_path) {
        maps = jag_load(maps_path);
        if (!maps.blob) {
            fprintf(stderr, "warn: failed to load maps pack: %s\n", maps_path);
        }
    }
    if (maps_mem_path) {
        maps_mem = jag_load(maps_mem_path);
        if (!maps_mem.blob) {
            fprintf(stderr, "warn: failed to load maps mem pack: %s\n", maps_mem_path);
        }
    }

    size_t converted = 0;

    /* Enumerate possible chunk names and convert those present.
       Name pattern matches RuneCast: m<plane><x/10><x%10><y/10><y%10>.

       Important: upper/interior planes can exist as .dat-only sections. The
       original runtime loader does not require .hei before decoding .dat; if
       .hei is absent, it zeroes terrain height/colour and continues. RCL must
       mirror that or upstairs/interior floor chunks such as m15250 are never
       emitted. */
    size_t hei_sections = 0;
    size_t dat_sections = 0;
    size_t dat_only_sections = 0;

    for (int plane = 0; plane < 4; plane++) {
        for (int x = 0; x < 100; x++) {
            for (int y = 0; y < 100; y++) {
                char base[16];
                char hei_name[24];
                char dat_name[24];

                snprintf(base, sizeof(base), "m%d%d%d%d%d",
                         plane, x / 10, x % 10, y / 10, y % 10);
                snprintf(hei_name, sizeof(hei_name), "%s.hei", base);
                snprintf(dat_name, sizeof(dat_name), "%s.dat", base);

                const uint32_t hh = rsc_hash_name(hei_name);
                const jag_entry_t *he = jag_find_by_hash(&land, hh);
                if (!he && land_mem.blob) he = jag_find_by_hash(&land_mem, hh);

                const uint32_t dh = rsc_hash_name(dat_name);
                const jag_entry_t *de = NULL;
                if (maps.blob) {
                    de = jag_find_by_hash(&maps, dh);
                    if (!de && maps_mem.blob) de = jag_find_by_hash(&maps_mem, dh);
                }

                if (!he && !de) continue;

                uint8_t heights[TILE_COUNT];
                uint8_t colours[TILE_COUNT];

                if (he) {
                    size_t hei_len = 0;
                    uint8_t *hei = jag_extract_entry(he, &hei_len);
                    if (!hei) continue;

                    if (!decode_hei(hei, hei_len, heights, colours)) {
                        fprintf(stderr, "warn: decode failed for %s (len=%zu)\n", hei_name, hei_len);
                        free(hei);
                        continue;
                    }
                    free(hei);
                    hei_sections++;
                } else {
                    /* Missing .hei is not an error. Runtime world_load_section_files()
                     * clears height/colour and still consumes the .dat file. */
                    memset(heights, 0, sizeof(heights));
                    memset(colours, 0, sizeof(colours));
                    dat_only_sections++;
                }

                uint8_t decoration[TILE_COUNT];
                memset(decoration, 0, sizeof(decoration));

                uint8_t walls_ns[TILE_COUNT];
                memset(walls_ns, 0, sizeof(walls_ns));

                uint8_t walls_ew[TILE_COUNT];
                memset(walls_ew, 0, sizeof(walls_ew));

                uint16_t walls_diag[TILE_COUNT];
                memset(walls_diag, 0, sizeof(walls_diag));

                uint8_t tile_dir[TILE_COUNT];
                memset(tile_dir, 0, sizeof(tile_dir));

                if (de) {
                    dat_sections++;
                    if (with_uv) {
                        size_t dat_len = 0;
                        uint8_t *dat = jag_extract_entry(de, &dat_len);
                        if (dat) {
                            if (!decode_dat_v63(dat, dat_len, decoration, walls_diag, tile_dir)) {
                                fprintf(stderr, "warn: .dat decode failed for %s (len=%zu)\n", dat_name, dat_len);
                                memset(decoration, 0, sizeof(decoration));
                                memset(walls_ns, 0, sizeof(walls_ns));
                                memset(walls_ew, 0, sizeof(walls_ew));
                                memset(walls_diag, 0, sizeof(walls_diag));
                                memset(tile_dir, 0, sizeof(tile_dir));
                            } else if (!decode_dat_v63_walls_selected(dat, dat_len, walls_ns, walls_ew)) {
                                fprintf(stderr, "warn: .dat wall decode failed for %s (len=%zu)\n", dat_name, dat_len);
                                memset(walls_ns, 0, sizeof(walls_ns));
                                memset(walls_ew, 0, sizeof(walls_ew));
                            }

                            /* Optional: dump decode stats to help validate wall-diagonal handling. */
                            dat_debug_print(dat_name, decoration, walls_diag, tile_dir);

                            free(dat);
                        }
                    }
                }

                rcl_nb_storage_t nbs;
                build_nb_3x3(&nbs, &land, &land_mem, &maps, &maps_mem, with_uv,
                             plane, x, y, heights, colours, (with_uv ? decoration : NULL));

                uint8_t trim_keep[TILE_COUNT];
                const uint8_t *trim_keep_ptr = NULL;
                if (trim_dat_only && !he && de && with_uv) {
                    /* DAT-only upper/lower floor trim:
                     *   - holes/zero-decoration cells are cut out;
                     *   - explicit floor decorations survive;
                     *   - zero-decoration cells stay zero. Runtime world_decoration_or_colour()
                     *     does not borrow neighbouring overlays, and planes 1/2 use transparent
                     *     base terrain. Mutating decoration here invents floors the game never
                     *     emits.
                     */
                    build_dat_only_runtime_trim_keep(decoration, walls_diag, plane, trim_keep);

                    int trim_any = 0;
                    for (int ti = 0; ti < TILE_COUNT; ++ti) {
                        if (trim_keep[ti]) { trim_any = 1; break; }
                    }

                    if (!trim_any) {
                        continue;
                    }

                    trim_keep_ptr = trim_keep;
                }

                mesh_out_t mesh;
                const int base_transparent = (plane == 1 || plane == 2);

                if (!build_rcl1_mesh(heights, colours, (with_uv ? decoration : NULL),
                                     (with_uv ? walls_diag : NULL),
                                     (with_uv ? tile_dir : NULL),
                                     trim_keep_ptr, base_transparent,
                                     with_uv, dedup, merge_faces, gen_spans,
                                     &nbs.nb, &mesh)) {
                    fprintf(stderr, "warn: mesh build failed for %s\n", base);
                    free(mesh.vtx_bytes);
                    free(mesh.idx);
                    free(mesh.sub);
                    continue;
                }

                char out_path[512];
                snprintf(out_path, sizeof(out_path), "%s/%s.rcm", out_dir, base);

                if (!write_rcl1_file(out_path, &mesh, with_uv, plane, x, y, he != NULL, de != NULL, heights, decoration, trim_keep_ptr)) {
                    fprintf(stderr, "warn: write failed %s\n", out_path);
                } else {
                    converted++;
                    printf("Wrote %s  verts=%u idx=%u%s%s%s\n",
                           out_path, mesh.vtx_count, mesh.idx_count,
                           with_uv ? " uv" : " no-uv",
                           dedup ? " dedup" : "",
                           he ? "" : (trim_keep_ptr ? " dat-only-trim" : " dat-only"));
                }

                free(mesh.vtx_bytes);
                free(mesh.idx);
                free(mesh.sub);
            }
        }
    }

    printf("Considered %zu HEI sections, %zu DAT sections (%zu DAT-only)\n",
           hei_sections, dat_sections, dat_only_sections);
    printf("Converted %zu landscape chunks to RCL1\n", converted);

    jag_free(&maps_mem);
    jag_free(&maps);
    jag_free(&land_mem);
    jag_free(&land);
    return 0;
}
