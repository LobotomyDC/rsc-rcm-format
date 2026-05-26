// rsc2rcm.c - Convert RSC Classic OB3 models inside JAG/MEM into .rcm (RCM1/RCM2/RCM3)
// C99, host tool. Requires libbz2 (-lbz2) and libm (-lm).
//
// Build from the repository root:
//   gcc -std=c99 -O2 -Wall -Wextra tools/rsc-rcm-format/rsc2rcm.c -lbz2 -lm -o tools/rsc-rcm-format/rsc2rcm
//
// Usage:
//   tools/rsc-rcm-format/rsc2rcm <archive.jag|archive.mem> <out_dir> [options]
//
// Common RuneCast/PVR usage:
//   tools/rsc-rcm-format/rsc2rcm cache/models36.jag cache/rcm/models --rcm1
//
// Options:
//   --rcm1                 Output RCM1: float positions + UVs. Preferred for PVR streaming.
//   --rcm2                 Output RCM2: int16 positions + UVs. Default, compact canonical form.
//   --rcm3                 Output RCM3: float positions, no UVs; only valid for untextured models.
//   --alpha-ids file.txt   Mark listed texture IDs as alpha submeshes, one integer ID per line.
//   --flip-uvs             Flip textured RCM1/RCM2 V coordinates inside each atlas region.
//   --coplanar-backface    Preserve old behavior: emit both OB3 face sides when both are valid.
//   --noreduction          Disable exact vertex deduplication for debugging.
//   --double-sided         Emit a back side for faces that only have one visible side.
//
// Output: one .rcm per OB3 entry. Filenames are model_<hash>.rcm by default.
//
// v1 goals implemented:
// - Triangulated geometry (fan triangulation, matches rsc-c GL path)
// - Normals computed (snorm8) but no baked lighting
// - Textured faces keep atlas UVs packed Q0.15 (0..32767)
// - Flat-colored faces store material color as ARGB1555 (per-submesh)
// - By default, valid front/back fills share one plane to avoid coplanar fighting on PVR
// - --coplanar-backface preserves separate front/back fills by emitting reversed back-side triangles
//
// Notes:
// - Transparency in OB3 is COLOUR_TRANSPARENT (INT16_MAX). If a side is transparent, we skip that side.
// - For textured transparency (alpha textures), we cannot infer reliably from OB3 alone.
//   Provide --alpha-ids list if you want to tag some texture IDs as alpha submeshes.

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <errno.h>
#include <sys/stat.h>
#include <inttypes.h>
#include <bzlib.h>

#ifndef _WIN32
#include <unistd.h>
#endif

// Convert a 32-bit hash value to a signed int32_t to generate a filename that corresponds with what's in the .jag
static int32_t hash_u32_to_s32(uint32_t h) {
    if (h & 0x80000000u) {
        // Interpret as negative two's-complement value.
        return (int32_t)((int64_t)h - 0x100000000LL);
    }
    return (int32_t)h;
}


// ---------------------- RSC-compatible constants ----------------------

#define COLOUR_TRANSPARENT INT16_MAX
#define VERTEX_SCALE 100
#define UV_Q15_DIVISOR 32767
#define RCM_FILE_ALIGN 32

static size_t align_up_sz(size_t v, size_t a) {
    return (v + (a - 1)) & ~(a - 1);
}

typedef enum rcm_out_format {
    RCM_OUT_RCM1 = 1, /* float positions */
    RCM_OUT_RCM2 = 2, /* int16 positions (current format) */
    RCM_OUT_RCM3 = 3  /* float positions, NO UV; only untextured models */
} rcm_out_format_t;


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

static const float gl_tri_face_us[]  = {0.0f, 1.0f, 0.0f};
static const float gl_tri_face_vs[]  = {1.0f, 1.0f, 0.0f};
static const float gl_quad_face_us[] = {0.0f, 1.0f, 1.0f, 0.0f};
static const float gl_quad_face_vs[] = {1.0f, 1.0f, 0.0f, 0.0f};

// ---------------------- RCM format (v1) ----------------------

/* ---------------------- RCM (RuneCast Mesh) format ----------------------
 *
 * RCM2 is alignment-friendly for Dreamcast:
 * - header is 96 bytes, naturally aligned (no packing)
 * - section offsets are padded to RCM_FILE_ALIGN bytes (default 32)
 * - intended usage: read whole file into an aligned blob and point into it
 */

typedef struct rcm_header {
    char     magic[4];          // "RCM2"
    uint16_t version;           // 2
    uint16_t flags;             // reserved

    uint32_t file_size;         // total file bytes including padding to RCM_FILE_ALIGN

    uint32_t vertex_count;
    uint32_t index_count;       // number of uint16 indices
    uint32_t submesh_count;

    uint32_t submesh_off;       // aligned to RCM_FILE_ALIGN
    uint32_t vertex_off;        // aligned to RCM_FILE_ALIGN
    uint32_t index_off;         // aligned to RCM_FILE_ALIGN

    uint16_t vertex_stride;     // 16
    uint16_t index_stride;      // 2

    uint16_t uv_divisor;        // usually 32767
    uint16_t reserved0;

    int16_t  aabb_min[3];
    int16_t  aabb_max[3];
    int16_t  bounds_pad;

    int16_t  pad_align0;

    uint32_t reserved1[9];
} rcm_header_t;

typedef struct rcm_submesh {
    uint16_t texture_id;       // 0xFFFF = untextured
    uint16_t color_argb1555;   // base material tint; textured typically 0xFFFF
    uint16_t flags;            // RCM_SM_ALPHA, RCM_SM_GOURAUD
    uint16_t reserved0;
    uint32_t first_index;
    uint32_t index_count;
} rcm_submesh_t;

typedef struct rcm_vtx16 {
    int16_t x, y, z;
    int16_t pad0;
    int8_t  nx, ny, nz;
    uint8_t pad1;
    int16_t u, v;              // Q0.15
} rcm_vtx16_t;

/*
 * RCM1 vertex: float positions for GLdc compatibility.
 * - Positions are world-space floats (RSC units / VERTEX_SCALE).
 * - Normals remain snorm8.
 * - UVs remain Q0.15 int16 (0..uv_divisor).
 *
 * Layout is intentionally ordered to avoid implicit padding surprises.
 */
typedef struct rcm_vtxf {
    float  x, y, z;
    int8_t nx, ny, nz;
    uint8_t pad1;
    int16_t u, v;
} rcm_vtxf_t;
#if defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 201112L)
_Static_assert(sizeof(rcm_vtxf_t) == 20, "rcm_vtxf_t must be 20 bytes");
#endif

/*
 * RCM3 vertex: float positions, NO UVs (for fully untextured models).
 * - Positions are world-space floats (RSC units / VERTEX_SCALE).
 * - Normals remain snorm8.
 *
 * Size is 16 bytes for excellent alignment and bandwidth.
 */
typedef struct rcm_vtxf_n {
    float   x, y, z;
    int8_t  nx, ny, nz;
    uint8_t pad1;
} rcm_vtxf_n_t;
#if defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 201112L)
_Static_assert(sizeof(rcm_vtxf_n_t) == 16, "rcm_vtxf_n_t must be 16 bytes");
#endif


#if defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 201112L)
_Static_assert(sizeof(rcm_header_t)  == 96, "rcm_header_t must be 96 bytes");
_Static_assert(sizeof(rcm_submesh_t) == 16, "rcm_submesh_t must be 16 bytes");
_Static_assert(sizeof(rcm_vtx16_t)   == 16, "rcm_vtx16_t must be 16 bytes");
#endif




enum {
    RCM_SM_ALPHA   = 1u << 0,
    RCM_SM_GOURAUD = 1u << 1,
};

// ---------------------- Helpers (file IO / endian / bzip) ----------------------

static void die(const char *msg) {
    fprintf(stderr, "FATAL: %s\n", msg);
    exit(1);
}

static void *xmalloc(size_t n) {
    void *p = malloc(n ? n : 1);
    if (!p) die("out of memory");
    return p;
}

static void *xrealloc(void *p, size_t n) {
    void *q = realloc(p, n ? n : 1);
    if (!q) die("out of memory");
    return q;
}

static uint16_t read_u16be(const uint8_t *b, size_t off, size_t len) {
    if (off + 2 > len) die("buffer underrun u16be");
    return (uint16_t)((b[off] << 8) | b[off+1]);
}

static int16_t read_s16be(const uint8_t *b, size_t off, size_t len) {
    return (int16_t)read_u16be(b, off, len);
}

static uint8_t read_u8(const uint8_t *b, size_t off, size_t len) {
    if (off + 1 > len) die("buffer underrun u8");
    return b[off];
}

static uint32_t read_u24be(const uint8_t *b, size_t off, size_t len) {
    if (off + 3 > len) die("buffer underrun u24be");
    return ((uint32_t)b[off] << 16) | ((uint32_t)b[off+1] << 8) | (uint32_t)b[off+2];
}

static uint32_t read_u32be(const uint8_t *b, size_t off, size_t len) {
    if (off + 4 > len) die("buffer underrun u32be");
    return ((uint32_t)b[off] << 24) | ((uint32_t)b[off+1] << 16) | ((uint32_t)b[off+2] << 8) | (uint32_t)b[off+3];
}

static uint8_t *read_entire_file(const char *path, size_t *out_len) {
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "Failed to open %s: %s\n", path, strerror(errno));
        exit(1);
    }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (sz < 0) die("ftell failed");
    uint8_t *buf = (uint8_t*)xmalloc((size_t)sz);
    if (fread(buf, 1, (size_t)sz, f) != (size_t)sz) die("fread failed");
    fclose(f);
    *out_len = (size_t)sz;
    return buf;
}

// JAG/MEM bzip streams are missing the "BZh1" header; rsc-c prepends it.
static uint8_t *bz_decompress_with_bzh1(const uint8_t *src, size_t src_len, size_t out_len_expected) {
    const char hdr[4] = {'B','Z','h','1'};
    size_t tmp_len = src_len + 4;
    uint8_t *tmp = (uint8_t*)xmalloc(tmp_len);
    memcpy(tmp, hdr, 4);
    memcpy(tmp + 4, src, src_len);

    unsigned int dest_len = (unsigned int)out_len_expected;
    uint8_t *dest = (uint8_t*)xmalloc(out_len_expected);

    int rc = BZ2_bzBuffToBuffDecompress((char*)dest, &dest_len, (char*)tmp, (unsigned int)tmp_len, 0, 0);
    free(tmp);

    if (rc != BZ_OK) {
        fprintf(stderr, "BZ2 decompress failed rc=%d\n", rc);
        free(dest);
        return NULL;
    }
    if ((size_t)dest_len != out_len_expected) {
        // Some archives may be slightly inconsistent; tolerate if smaller.
        // But keep buffer sized to expected.
    }
    return dest;
}

static int ensure_dir(const char *path) {
#ifndef _WIN32
    struct stat st;
    if (stat(path, &st) == 0) {
        if (S_ISDIR(st.st_mode)) return 0;
        return -1;
    }
    if (mkdir(path, 0755) != 0) return -1;
    return 0;
#else
    (void)path;
    return 0;
#endif
}

static int write_entire_file(const char *path, const void *data, size_t len) {
    FILE *f = fopen(path, "wb");
    if (!f) return -1;
    if (fwrite(data, 1, len, f) != len) { fclose(f); return -1; }
    fclose(f);
    return 0;
}

// ---------------------- JAG/MEM archive parsing ----------------------

typedef struct jag_entry {
    uint32_t hash;
    uint32_t unpacked_len;
    uint32_t packed_len;
    const uint8_t *payload; // points into decompressed archive blob (packed bytes)
} jag_entry_t;

typedef struct jag_archive {
    uint8_t     *blob;      // decompressed outer blob
    size_t       blob_len;
    jag_entry_t *entries;
    uint16_t     count;
} jag_archive_t;

// Load outer archive: [u24 unpacked][u24 packed][packed bytes...], where packed bytes are bzip without BZh1.
static jag_archive_t jag_load(const char *path) {
    jag_archive_t a;
    memset(&a, 0, sizeof(a));

    size_t file_len = 0;
    uint8_t *file = read_entire_file(path, &file_len);
    if (file_len < 6) die("archive too small");

    uint32_t unpacked = read_u24be(file, 0, file_len);
    uint32_t packed   = read_u24be(file, 3, file_len);

    if (6 + packed > file_len) die("archive header sizes invalid");

    if (packed != unpacked) {
        a.blob = bz_decompress_with_bzh1(file + 6, packed, unpacked);
        if (!a.blob) die("outer bzip decompress failed");
        a.blob_len = unpacked;
    } else {
        a.blob = (uint8_t*)xmalloc(unpacked);
        memcpy(a.blob, file + 6, unpacked);
        a.blob_len = unpacked;
    }
    free(file);

    // Inner directory
    size_t off = 0;
    uint16_t count = read_u16be(a.blob, off, a.blob_len);
    off += 2;

    size_t dir_len = 2 + (size_t)count * 10;
    if (dir_len > a.blob_len) die("inner directory overruns blob");

    a.entries = (jag_entry_t*)xmalloc(sizeof(jag_entry_t) * count);
    a.count = count;

    // Parse directory entries: [u32 hash][u24 unpacked][u24 packed] big-endian, then packed bytes follow sequentially.
    size_t data_off = dir_len;
    for (uint16_t i = 0; i < count; i++) {
        size_t eoff = 2 + (size_t)i * 10;
        uint32_t hash = read_u32be(a.blob, eoff + 0, a.blob_len);
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

static uint8_t *jag_extract_entry(const jag_entry_t *e, size_t *out_len) {
    if (e->packed_len == e->unpacked_len) {
        uint8_t *buf = (uint8_t*)xmalloc(e->unpacked_len);
        memcpy(buf, e->payload, e->unpacked_len);
        *out_len = e->unpacked_len;
        return buf;
    }
    uint8_t *buf = bz_decompress_with_bzh1(e->payload, e->packed_len, e->unpacked_len);
    if (!buf) return NULL;
    *out_len = e->unpacked_len;
    return buf;
}

// ---------------------- OB3 parsing (matches game_model_new_ob3) ----------------------

typedef struct ob3_model {
    uint16_t vertex_count;
    uint16_t face_count;

    int16_t  *vx, *vy, *vz;         // vertex arrays
    uint8_t  *face_vcount;          // per face
    int16_t  *fill_front;           // signed
    int16_t  *fill_back;            // signed
    uint8_t  *face_gouraud;         // 0/1

    uint16_t **face_vertices;       // array of pointers, each length face_vcount[i]
} ob3_model_t;

static void ob3_free(ob3_model_t *m) {
    if (!m) return;
    if (m->face_vertices) {
        for (uint16_t i = 0; i < m->face_count; i++) free(m->face_vertices[i]);
    }
    free(m->face_vertices);
    free(m->face_gouraud);
    free(m->fill_back);
    free(m->fill_front);
    free(m->face_vcount);
    free(m->vz);
    free(m->vy);
    free(m->vx);
    memset(m, 0, sizeof(*m));
}

static int ob3_parse(const uint8_t *data, size_t len, ob3_model_t *out) {
    memset(out, 0, sizeof(*out));
    if (len < 4) return 0;

    size_t off = 0;
    uint16_t vc = read_u16be(data, off, len); off += 2;
    uint16_t fc = read_u16be(data, off, len); off += 2;

    // Sanity
    if (vc == 0 || fc == 0) return 0;

    out->vertex_count = vc;
    out->face_count   = fc;

    out->vx = (int16_t*)xmalloc(sizeof(int16_t) * vc);
    out->vy = (int16_t*)xmalloc(sizeof(int16_t) * vc);
    out->vz = (int16_t*)xmalloc(sizeof(int16_t) * vc);

    for (uint16_t i = 0; i < vc; i++) { out->vx[i] = read_s16be(data, off, len); off += 2; }
    for (uint16_t i = 0; i < vc; i++) { out->vy[i] = read_s16be(data, off, len); off += 2; }
    for (uint16_t i = 0; i < vc; i++) { out->vz[i] = read_s16be(data, off, len); off += 2; }

    out->face_vcount = (uint8_t*)xmalloc(fc);
    for (uint16_t i = 0; i < fc; i++) { out->face_vcount[i] = read_u8(data, off++, len); }

    out->fill_front = (int16_t*)xmalloc(sizeof(int16_t) * fc);
    for (uint16_t i = 0; i < fc; i++) {
        int16_t v = read_s16be(data, off, len); off += 2;
        if (v == 32767) v = COLOUR_TRANSPARENT;
        out->fill_front[i] = v;
    }

    out->fill_back = (int16_t*)xmalloc(sizeof(int16_t) * fc);
    for (uint16_t i = 0; i < fc; i++) {
        int16_t v = read_s16be(data, off, len); off += 2;
        if (v == 32767) v = COLOUR_TRANSPARENT;
        out->fill_back[i] = v;
    }

    out->face_gouraud = (uint8_t*)xmalloc(fc);
    for (uint16_t i = 0; i < fc; i++) {
        uint8_t is_g = read_u8(data, off++, len);
        out->face_gouraud[i] = is_g ? 1 : 0;
    }

    out->face_vertices = (uint16_t**)xmalloc(sizeof(uint16_t*) * fc);
    for (uint16_t i = 0; i < fc; i++) {
        uint8_t n = out->face_vcount[i];
        if (n < 3) return 0;
        out->face_vertices[i] = (uint16_t*)xmalloc(sizeof(uint16_t) * n);
        for (uint8_t j = 0; j < n; j++) {
            if (vc < 256) {
                out->face_vertices[i][j] = (uint16_t)read_u8(data, off++, len);
            } else {
                out->face_vertices[i][j] = read_u16be(data, off, len);
                off += 2;
            }
            if (out->face_vertices[i][j] >= vc) return 0;
        }
    }

    return 1;
}

// ---------------------- Math helpers ----------------------

static float vtx_to_float(int16_t v) { return (float)v / (float)VERTEX_SCALE; }

static void vec3_sub(const float a[3], const float b[3], float out[3]) {
    out[0] = a[0] - b[0]; out[1] = a[1] - b[1]; out[2] = a[2] - b[2];
}
static float vec3_dot(const float a[3], const float b[3]) {
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
}
static void vec3_cross(const float a[3], const float b[3], float out[3]) {
    out[0] = a[1]*b[2] - a[2]*b[1];
    out[1] = a[2]*b[0] - a[0]*b[2];
    out[2] = a[0]*b[1] - a[1]*b[0];
}
static float vec3_len(const float v[3]) {
    return sqrtf(vec3_dot(v,v));
}
static int vec3_normalize(float v[3]) {
    float L = vec3_len(v);
    if (L < 1e-12f) return 0;
    v[0] /= L; v[1] /= L; v[2] /= L;
    return 1;
}

static int8_t snorm8(float x) {
    if (x < -1.0f) x = -1.0f;
    if (x >  1.0f) x =  1.0f;
    return (int8_t)lrintf(x * 127.0f);
}

// ---------------------- UV unwrap + atlas offset (RSC-compatible) ----------------------

static void game_model_gl_unwrap_uvs(const ob3_model_t *m, const uint16_t *face_vertices,
                                    int face_vertex_count, float *us, float *vs) {
    if (face_vertex_count <= 4) {
        const float *face_us = NULL;
        const float *face_vs = NULL;

        if (face_vertex_count == 3) { face_us = gl_tri_face_us;  face_vs = gl_tri_face_vs; }
        if (face_vertex_count == 4) { face_us = gl_quad_face_us; face_vs = gl_quad_face_vs; }

        for (int i = 0; i < face_vertex_count; i++) {
            us[i] = face_us[i];
            vs[i] = face_vs[i]; // RENDER_GL path (no 3DS flip)
        }
        return;
    }

    // Build face vertex positions as floats
    float verts[64][3]; // RSC faces are small; if you ever see >64, bump.
    if (face_vertex_count > 64) {
        // fallback: just do something stable
        for (int i = 0; i < face_vertex_count; i++) { us[i] = 0.0f; vs[i] = 0.0f; }
        return;
    }

    for (int i = 0; i < face_vertex_count; i++) {
        uint16_t vi = face_vertices[i];
        verts[i][0] = vtx_to_float(m->vx[vi]);
        verts[i][1] = vtx_to_float(m->vy[vi]);
        verts[i][2] = vtx_to_float(m->vz[vi]);
    }

    float location_x[3] = {0};
    float delta[3] = {0};
    vec3_sub(verts[1], verts[0], location_x);
    vec3_sub(verts[2], verts[0], delta);

    float normal[3] = {0};
    vec3_cross(location_x, delta, normal);

    float location_y[3] = {0};
    vec3_cross(normal, location_x, location_y);

    vec3_normalize(location_x);
    vec3_normalize(location_y);

    float max_x = 0, min_x = 0, max_y = 0, min_y = 0;

    for (int i = 0; i < face_vertex_count; i++) {
        float v[3] = {0};
        vec3_sub(verts[i], verts[0], v);

        float x = vec3_dot(v, location_x);
        float y = vec3_dot(v, location_y);

        if (i == 0 || x > max_x) max_x = x;
        if (i == 0 || x < min_x) min_x = x;
        if (i == 0 || y > max_y) max_y = y;
        if (i == 0 || y < min_y) min_y = y;

        us[i] = x;
        vs[i] = y;
    }

    float dx = (max_x - min_x);
    float dy = (max_y - min_y);
    if (fabsf(dx) < 1e-12f) dx = 1.0f;
    if (fabsf(dy) < 1e-12f) dy = 1.0f;

    for (int i = 0; i < face_vertex_count; i++) {
        float x = us[i];
        float y = vs[i];
        us[i] = (x - min_x) / dx;
        vs[i] = 1.0f - (y - min_y) / dy; // RENDER_GL path
    }
}

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

// ---------------------- Material mapping (RSC-compatible) ----------------------

typedef struct {
    int      valid;          // 0 => transparent / skip
    uint16_t texture_id;     // 0xFFFF => untextured
    uint16_t color_argb1555; // base tint
    uint16_t flags;          // alpha/gouraud intent etc
} material_t;

typedef struct alpha_id_list {
    uint16_t *ids;
    size_t    count;
} alpha_id_list_t;

static int alpha_id_contains(const alpha_id_list_t *lst, uint16_t id) {
    for (size_t i = 0; i < lst->count; i++) if (lst->ids[i] == id) return 1;
    return 0;
}

// Convert OB3 face_fill value into material.
// RSC rule: if fill == transparent => skip.
// if fill < 0 => color = -1-fill (RGB555), untextured.
// if fill >= 0 => texture index.
static material_t fill_to_material(int16_t fill, uint8_t gouraud, const alpha_id_list_t *alpha_ids) {
    material_t m;
    memset(&m, 0, sizeof(m));

    if (fill == COLOUR_TRANSPARENT) {
        m.valid = 0;
        return m;
    }

    m.valid = 1;
    if (gouraud) m.flags |= RCM_SM_GOURAUD;

    if (fill < 0) {
        int16_t rgb555 = (int16_t)(-1 - fill);
        // ARGB1555: set alpha=1, carry rgb bits as-is
        m.texture_id = 0xFFFF;
        m.color_argb1555 = (uint16_t)(0x8000u | ((uint16_t)rgb555 & 0x7FFFu));
    } else {
        uint16_t tex = (uint16_t)fill;
        m.texture_id = tex;
        m.color_argb1555 = 0xFFFFu; // opaque white tint for textured
        if (alpha_ids && alpha_id_contains(alpha_ids, tex)) {
            m.flags |= RCM_SM_ALPHA;
        }
    }
    return m;
}

// ---------------------- RCM build buffers ----------------------

typedef struct {
    uint16_t texture_id;
    uint16_t color_argb1555;
    uint16_t flags;

    rcm_vtx16_t *v;
    uint16_t    *i;
    uint32_t     vcount, icount;
    uint32_t     vcap, icap;

    /* Optional vertex dedup table (open addressing). slot = vertex_index+1, 0 = empty */
    uint32_t *vtab;
    uint32_t vtab_cap;
    uint32_t vtab_used;
} submesh_builder_t;

static int submesh_key_eq(const submesh_builder_t *a, uint16_t texture_id,
                          uint16_t color_argb1555, uint16_t flags) {
    return a->texture_id == texture_id &&
           a->color_argb1555 == color_argb1555 &&
           a->flags == flags;
}

static uint32_t hash_vtx16_fnv1a(const rcm_vtx16_t *v) {
    const uint8_t *p = (const uint8_t*)v;
    uint32_t h = 2166136261u;
    for (size_t k = 0; k < sizeof(*v); k++) {
        h ^= (uint32_t)p[k];
        h *= 16777619u;
    }
    return h;
}

static uint32_t next_pow2_u32(uint32_t x) {
    if (x < 2u) return 2u;
    x--;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    return x + 1u;
}

static void submesh_vtab_rebuild(submesh_builder_t *s, uint32_t new_cap) {
    new_cap = next_pow2_u32(new_cap);
    uint32_t *nt = (uint32_t*)calloc((size_t)new_cap, sizeof(uint32_t));
    if (!nt) die("out of memory (vtab)");

    uint32_t mask = new_cap - 1u;
    for (uint32_t vi = 0; vi < s->vcount; vi++) {
        uint32_t h = hash_vtx16_fnv1a(&s->v[vi]);
        uint32_t slot = h & mask;
        while (nt[slot] != 0u) slot = (slot + 1u) & mask;
        nt[slot] = vi + 1u;
    }

    free(s->vtab);
    s->vtab = nt;
    s->vtab_cap = new_cap;
    s->vtab_used = s->vcount;
}

static uint16_t submesh_add_vertex(submesh_builder_t *s, const rcm_vtx16_t *v,
                                   int reduce_vertices) {
    if (!reduce_vertices) {
        /* legacy: no reduction, always append */
        if (s->vcount + 1u > s->vcap) {
            s->vcap = (s->vcap ? (s->vcap * 2u) : 256u);
            s->v = (rcm_vtx16_t*)realloc(s->v, (size_t)s->vcap * sizeof(*s->v));
            if (!s->v) die("out of memory (vtx)");
        }
        if (s->vcount > 0xFFFFu) die("too many vertices in submesh (need 32-bit indices)");
        s->v[s->vcount] = *v;
        return (uint16_t)s->vcount++;
    }

    /* Reduced path: initialize table lazily. Keep load factor ~70%. */
    if (!s->vtab) submesh_vtab_rebuild(s, 512u);

    if ((s->vtab_used + 1u) * 10u >= s->vtab_cap * 7u) {
        submesh_vtab_rebuild(s, s->vtab_cap ? (s->vtab_cap * 2u) : 512u);
    }

    uint32_t h = hash_vtx16_fnv1a(v);
    uint32_t mask = s->vtab_cap - 1u;
    uint32_t slot = h & mask;

    while (s->vtab[slot] != 0u) {
        uint32_t vi = s->vtab[slot] - 1u;
        if (memcmp(&s->v[vi], v, sizeof(*v)) == 0) {
            if (vi > 0xFFFFu) die("too many vertices in submesh (need 32-bit indices)");
            return (uint16_t)vi;
        }
        slot = (slot + 1u) & mask;
    }

    /* append new vertex */
    if (s->vcount + 1u > s->vcap) {
        s->vcap = (s->vcap ? (s->vcap * 2u) : 256u);
        s->v = (rcm_vtx16_t*)realloc(s->v, (size_t)s->vcap * sizeof(*s->v));
        if (!s->v) die("out of memory (vtx)");
    }
    if (s->vcount > 0xFFFFu) die("too many vertices in submesh (need 32-bit indices)");

    s->v[s->vcount] = *v;
    s->vtab[slot] = s->vcount + 1u;
    s->vcount++;
    s->vtab_used++;
    return (uint16_t)(s->vcount - 1u);
}

static void submesh_free(submesh_builder_t *S) {
    free(S->v);
    free(S->i);
    free(S->vtab);
    memset(S, 0, sizeof(*S));
}

static void submesh_push_triangle(submesh_builder_t *S, const rcm_vtx16_t tri[3],
                                  int reduce_vertices) {
    uint16_t vi0 = submesh_add_vertex(S, &tri[0], reduce_vertices);
    uint16_t vi1 = submesh_add_vertex(S, &tri[1], reduce_vertices);
    uint16_t vi2 = submesh_add_vertex(S, &tri[2], reduce_vertices);

    if (S->icount + 3u > S->icap) {
        S->icap = (S->icap ? (S->icap * 2u) : 256u);
        S->i = (uint16_t*)realloc(S->i, (size_t)S->icap * sizeof(uint16_t));
        if (!S->i) die("out of memory (idx)");
    }

    S->i[S->icount++] = vi0;
    S->i[S->icount++] = vi1;
    S->i[S->icount++] = vi2;
}

// ---------------------- Conversion: OB3 -> RCM ----------------------

typedef struct {
    int8_t *nx, *ny, *nz; // per original vertex index, snorm8
} vertex_normals_t;

static vertex_normals_t compute_vertex_normals(const ob3_model_t *m) {
    // Accumulate face normals for gouraud faces
    float *acc = (float*)xmalloc(sizeof(float) * (size_t)m->vertex_count * 3);
    memset(acc, 0, sizeof(float) * (size_t)m->vertex_count * 3);

    for (uint16_t fi = 0; fi < m->face_count; fi++) {
        if (!m->face_gouraud[fi]) continue;

        uint8_t n = m->face_vcount[fi];
        const uint16_t *fv = m->face_vertices[fi];
        if (n < 3) continue;

        // Compute polygon normal from first 3 vertices (matches general approach)
        float p0[3] = { vtx_to_float(m->vx[fv[0]]), vtx_to_float(m->vy[fv[0]]), vtx_to_float(m->vz[fv[0]]) };
        float p1[3] = { vtx_to_float(m->vx[fv[1]]), vtx_to_float(m->vy[fv[1]]), vtx_to_float(m->vz[fv[1]]) };
        float p2[3] = { vtx_to_float(m->vx[fv[2]]), vtx_to_float(m->vy[fv[2]]), vtx_to_float(m->vz[fv[2]]) };

        float a[3], b[3], nrm[3];
        vec3_sub(p1, p0, a);
        vec3_sub(p2, p0, b);
        vec3_cross(a, b, nrm);

        // Area weight by length (same as using unnormalized cross)
        // But normalize afterward when encoding.
        for (uint8_t j = 0; j < n; j++) {
            uint16_t vi = fv[j];
            acc[(size_t)vi*3 + 0] += nrm[0];
            acc[(size_t)vi*3 + 1] += nrm[1];
            acc[(size_t)vi*3 + 2] += nrm[2];
        }
    }

    vertex_normals_t vn;
    vn.nx = (int8_t*)xmalloc(m->vertex_count);
    vn.ny = (int8_t*)xmalloc(m->vertex_count);
    vn.nz = (int8_t*)xmalloc(m->vertex_count);

    for (uint16_t vi = 0; vi < m->vertex_count; vi++) {
        float nrm[3] = { acc[(size_t)vi*3+0], acc[(size_t)vi*3+1], acc[(size_t)vi*3+2] };
        if (!vec3_normalize(nrm)) {
            // fallback normal
            nrm[0] = 0.0f; nrm[1] = 1.0f; nrm[2] = 0.0f;
        }
        vn.nx[vi] = snorm8(nrm[0]);
        vn.ny[vi] = snorm8(nrm[1]);
        vn.nz[vi] = snorm8(nrm[2]);
    }

    free(acc);
    return vn;
}

static void free_vertex_normals(vertex_normals_t *vn) {
    free(vn->nx); free(vn->ny); free(vn->nz);
    memset(vn, 0, sizeof(*vn));
}

static void compute_face_normal(const ob3_model_t *m, const uint16_t *fv, float out_n[3]) {
    float p0[3] = { vtx_to_float(m->vx[fv[0]]), vtx_to_float(m->vy[fv[0]]), vtx_to_float(m->vz[fv[0]]) };
    float p1[3] = { vtx_to_float(m->vx[fv[1]]), vtx_to_float(m->vy[fv[1]]), vtx_to_float(m->vz[fv[1]]) };
    float p2[3] = { vtx_to_float(m->vx[fv[2]]), vtx_to_float(m->vy[fv[2]]), vtx_to_float(m->vz[fv[2]]) };
    float a[3], b[3];
    vec3_sub(p1, p0, a);
    vec3_sub(p2, p0, b);
    vec3_cross(a, b, out_n);
    if (!vec3_normalize(out_n)) {
        out_n[0]=0; out_n[1]=1; out_n[2]=0;
    }
}

static int texture_id_in_range(uint16_t tex) {
    return tex < (uint16_t)(sizeof(gl_texture_atlas_positions)/sizeof(gl_texture_atlas_positions[0]));
}

static void convert_ob3_to_rcm(const ob3_model_t *m, const alpha_id_list_t *alpha_ids,
                              int reduce_vertices, int force_double_sided,
                              int flip_uvs, int coplanar_backface,
                              rcm_out_format_t out_fmt,
                              uint8_t **out_blob, size_t *out_len) {
    /* RCM3 only supports fully-untextured models. Proactively reject any model
     * that references a texture on either face side (front/back). This avoids
     * doing expensive work (UV unwrap, atlas lookup, normal build) for models
     * we will discard anyway.
     */
    if (out_fmt == RCM_OUT_RCM3) {
        for (uint16_t fi = 0; fi < m->face_count; fi++) {
            const int16_t ff = m->fill_front[fi];
            const int16_t fb = m->fill_back[fi];

            if (ff != COLOUR_TRANSPARENT && ff >= 0) { *out_blob = NULL; *out_len = 0; return; }
            if (fb != COLOUR_TRANSPARENT && fb >= 0) { *out_blob = NULL; *out_len = 0; return; }
        }
    }

    // Compute per-vertex normals for gouraud faces.
    vertex_normals_t vnorm = compute_vertex_normals(m);

    submesh_builder_t *subs = NULL;
    size_t sub_count = 0, sub_cap = 0;

    // Iterate faces, emit front and back sides if present.
    for (uint16_t fi = 0; fi < m->face_count; fi++) {
        const uint16_t *fv = m->face_vertices[fi];
        uint8_t n = m->face_vcount[fi];
        if (n < 3) continue;

        // Precompute unwrap UVs once per face (local), then apply atlas per side (front/back).
        // Only needed if we are outputting UVs AND at least one side is textured.
        float us[256], vs[256];
        int need_uv = (out_fmt != RCM_OUT_RCM3) &&
                      ((m->fill_front[fi] != COLOUR_TRANSPARENT && m->fill_front[fi] >= 0) ||
                       (m->fill_back[fi]  != COLOUR_TRANSPARENT && m->fill_back[fi]  >= 0));
        if (need_uv) {
            game_model_gl_unwrap_uvs(m, fv, (int)n, us, vs);
        }

        // Face normal (for flat faces)
        float fn[3] = {0};
        compute_face_normal(m, fv, fn);

        material_t mats[2];
        mats[0] = fill_to_material(m->fill_front[fi], m->face_gouraud[fi], alpha_ids);
        mats[1] = fill_to_material(m->fill_back[fi],  m->face_gouraud[fi], alpha_ids);

        for (int side = 0; side < 2; side++) {
            if (side == 1 && mats[0].valid && mats[1].valid && !coplanar_backface) {
                continue;
            }

            material_t mat = mats[side];
            if (!mat.valid) continue;

            // Validate texture id if textured. Keep geometry even if the atlas
            // table does not contain this texture; fall back to an opaque white
            // untextured material instead of silently dropping the face side.
            gl_atlas_position tp = {0};
            if (mat.texture_id != 0xFFFF) {
                if (!texture_id_in_range(mat.texture_id)) {
                    mat.texture_id = 0xFFFF;
                    mat.color_argb1555 = 0xFFFFu;
                    mat.flags &= (uint16_t)~RCM_SM_ALPHA;
                } else {
                    tp = gl_texture_atlas_positions[mat.texture_id];
                }
            }

            // Find/create submesh bucket
            submesh_builder_t *S = NULL;
            for (size_t si = 0; si < sub_count; si++) {
                if (submesh_key_eq(&subs[si], mat.texture_id, mat.color_argb1555, mat.flags)) {
                    S = &subs[si];
                    break;
                }
            }
            if (!S) {
                if (sub_count + 1 > sub_cap) {
                    sub_cap = (sub_cap == 0) ? 16 : (sub_cap * 2);
                    subs = (submesh_builder_t*)xrealloc(subs, sub_cap * sizeof(submesh_builder_t));
                }
                S = &subs[sub_count++];
                memset(S, 0, sizeof(*S));
                S->texture_id = mat.texture_id;
                S->color_argb1555 = mat.color_argb1555;
                S->flags = mat.flags;
            }

            // Emit triangles (fan). For back side, reverse winding and flip normals.
            for (uint8_t j = 0; j < (uint8_t)(n - 2); j++) {
                uint16_t i0 = fv[0];
                uint16_t i1 = fv[j + 1];
                uint16_t i2 = fv[j + 2];

                // Back side: swap i1/i2 for reversed winding
                if (side == 1) {
                    uint16_t tmp = i1; i1 = i2; i2 = tmp;
                }

                // Prepare 3 packed vertices
                uint16_t ids[3] = { i0, i1, i2 };
                rcm_vtx16_t tri[3];

                for (int k = 0; k < 3; k++) {
                    uint16_t vi = ids[k];
                    rcm_vtx16_t vtx;
                    memset(&vtx, 0, sizeof(vtx));

                    vtx.x = m->vx[vi];
                    vtx.y = m->vy[vi];
                    vtx.z = m->vz[vi];
                    vtx.pad0 = 0;

                    // Normal selection:
                    // gouraud => per-vertex normal, else face normal
                    if (m->face_gouraud[fi]) {
                        vtx.nx = vnorm.nx[vi];
                        vtx.ny = vnorm.ny[vi];
                        vtx.nz = vnorm.nz[vi];
                    } else {
                        vtx.nx = snorm8(fn[0]);
                        vtx.ny = snorm8(fn[1]);
                        vtx.nz = snorm8(fn[2]);
                    }

                    // Back side flips normal (important for future lighting)
                    if (side == 1) {
                        vtx.nx = (int8_t)(-vtx.nx);
                        vtx.ny = (int8_t)(-vtx.ny);
                        vtx.nz = (int8_t)(-vtx.nz);
                    }

                    // UVs only meaningful for textured
                    if (mat.texture_id != 0xFFFF) {
                        // Need UV of the corresponding vertex in the face polygon.
                        // Find polygon-local index for vi.
                        int poly_idx = -1;
                        for (uint8_t t = 0; t < n; t++) if (fv[t] == vi) { poly_idx = (int)t; break; }
                        if (poly_idx < 0) poly_idx = 0;

                        float u = us[poly_idx];
                        float v = vs[poly_idx];
                        gl_offset_texture_uvs_atlas(tp, &u, &v);
                        if (flip_uvs && out_fmt != RCM_OUT_RCM3) {
                            v = tp.top_v + tp.bottom_v - v;
                        }

                        vtx.u = pack_uv_q15(u);
                        vtx.v = pack_uv_q15(v);
                    } else {
                        vtx.u = 0;
                        vtx.v = 0;
                    }

                    vtx.pad1 = 0;
                    tri[k] = vtx;
                }

                submesh_push_triangle(S, tri, reduce_vertices);
                if (force_double_sided && !mats[1 - side].valid) {
                    rcm_vtx16_t tri2[3] = { tri[0], tri[2], tri[1] };
                    for (int k = 0; k < 3; k++) {
                        tri2[k].nx = (int8_t)-tri2[k].nx;
                        tri2[k].ny = (int8_t)-tri2[k].ny;
                        tri2[k].nz = (int8_t)-tri2[k].nz;
                    }
                    submesh_push_triangle(S, tri2, reduce_vertices);
                }
            }
        }
    }

    // Flatten submeshes into final contiguous buffers, build header + tables.
    size_t total_v = 0, total_i = 0;
    for (size_t si = 0; si < sub_count; si++) {
        total_v += subs[si].vcount;
        total_i += subs[si].icount;
    }

    // Compute bounds from final vertex positions
    int16_t minx =  32767, miny =  32767, minz =  32767;
    int16_t maxx = -32768, maxy = -32768, maxz = -32768;

    for (size_t si = 0; si < sub_count; si++) {
        for (size_t vi = 0; vi < subs[si].vcount; vi++) {
            rcm_vtx16_t *v = &subs[si].v[vi];
            if (v->x < minx) minx = v->x;
            if (v->x > maxx) maxx = v->x;
            if (v->y < miny) miny = v->y;
            if (v->y > maxy) maxy = v->y;
            if (v->z < minz) minz = v->z;
            if (v->z > maxz) maxz = v->z;
        }
    }

    size_t hdr_sz = sizeof(rcm_header_t);
    size_t sub_sz = sizeof(rcm_submesh_t) * sub_count;
    const size_t vtx_stride = (out_fmt == RCM_OUT_RCM1) ? sizeof(rcm_vtxf_t) :
                             (out_fmt == RCM_OUT_RCM3) ? sizeof(rcm_vtxf_n_t) :
                             sizeof(rcm_vtx16_t);
    size_t vtx_sz = vtx_stride * total_v;
    size_t idx_sz = sizeof(uint16_t) * total_i;

    /* Pad each section start to RCM_FILE_ALIGN (Dreamcast-friendly). */
    size_t sub_off = align_up_sz(hdr_sz, RCM_FILE_ALIGN);
    size_t vtx_off = align_up_sz(sub_off + sub_sz, RCM_FILE_ALIGN);
    size_t idx_off = align_up_sz(vtx_off + vtx_sz, RCM_FILE_ALIGN);
    size_t file_sz = align_up_sz(idx_off + idx_sz, RCM_FILE_ALIGN);

    uint8_t *blob = (uint8_t*)xmalloc(file_sz);
    memset(blob, 0, file_sz);

    rcm_header_t H;
    memset(&H, 0, sizeof(H));
    if (out_fmt == RCM_OUT_RCM1) {
        memcpy(H.magic, "RCM1", 4);
        H.version = 1;
    } else if (out_fmt == RCM_OUT_RCM3) {
        memcpy(H.magic, "RCM3", 4);
        H.version = 3;
    } else {
        memcpy(H.magic, "RCM2", 4);
        H.version = 2;
    }
    H.flags = 0;

    H.file_size = (uint32_t)file_sz;

    H.vertex_count  = (uint32_t)total_v;
    H.index_count   = (uint32_t)total_i;
    H.submesh_count = (uint32_t)sub_count;

    H.submesh_off = (uint32_t)sub_off;
    H.vertex_off  = (uint32_t)vtx_off;
    H.index_off   = (uint32_t)idx_off;

    H.vertex_stride = (uint16_t)vtx_stride;
    H.index_stride  = (uint16_t)sizeof(uint16_t);

    H.uv_divisor = (out_fmt == RCM_OUT_RCM3) ? 0u : (uint16_t)UV_Q15_DIVISOR;

    H.aabb_min[0] = minx; H.aabb_min[1] = miny; H.aabb_min[2] = minz;
    H.aabb_max[0] = maxx; H.aabb_max[1] = maxy; H.aabb_max[2] = maxz;
    H.bounds_pad = 0;
    H.pad_align0 = 0;

    memcpy(blob, &H, sizeof(H));


    rcm_submesh_t *SM = (rcm_submesh_t*)(blob + sub_off);
    void          *Vraw = (void*)(blob + vtx_off);
    uint16_t      *I = (uint16_t*)(blob + idx_off);

    size_t v_cursor = 0;
    size_t i_cursor = 0;

    for (size_t si = 0; si < sub_count; si++) {
        submesh_builder_t *S = &subs[si];

        // Write submesh record
        SM[si].texture_id = S->texture_id;
        SM[si].color_argb1555 = S->color_argb1555;
        SM[si].flags = S->flags;
        SM[si].reserved0 = 0;
        SM[si].first_index = (uint32_t)i_cursor;
        SM[si].index_count = (uint32_t)S->icount;

        // Copy / convert vertices into final packed buffer
        if (out_fmt == RCM_OUT_RCM1) {
            rcm_vtxf_t *Vf = (rcm_vtxf_t *)Vraw;
            for (size_t vi = 0; vi < S->vcount; ++vi) {
                const rcm_vtx16_t *src = &S->v[vi];
                rcm_vtxf_t *dst = &Vf[v_cursor + vi];

                dst->x = (float)src->x / (float)VERTEX_SCALE;
                dst->y = (float)src->y / (float)VERTEX_SCALE;
                dst->z = (float)src->z / (float)VERTEX_SCALE;

                dst->nx = src->nx;
                dst->ny = src->ny;
                dst->nz = src->nz;
                dst->pad1 = 0;

                dst->u = src->u;
                dst->v = src->v;
            }
        } else if (out_fmt == RCM_OUT_RCM3) {
            rcm_vtxf_n_t *Vn = (rcm_vtxf_n_t *)Vraw;
            for (size_t vi = 0; vi < S->vcount; ++vi) {
                const rcm_vtx16_t *src = &S->v[vi];
                rcm_vtxf_n_t *dst = &Vn[v_cursor + vi];

                dst->x = (float)src->x / (float)VERTEX_SCALE;
                dst->y = (float)src->y / (float)VERTEX_SCALE;
                dst->z = (float)src->z / (float)VERTEX_SCALE;

                dst->nx = src->nx;
                dst->ny = src->ny;
                dst->nz = src->nz;
                dst->pad1 = 0;
            }
        } else {
            rcm_vtx16_t *Vi = (rcm_vtx16_t *)Vraw;
            memcpy(&Vi[v_cursor], S->v, S->vcount * sizeof(rcm_vtx16_t));
        }

        // Copy indices with base offset
        for (size_t ii = 0; ii < S->icount; ii++) {
            uint32_t adj = (uint32_t)S->i[ii] + (uint32_t)v_cursor;
            if (adj > 0xFFFFu) die("index overflow (too many vertices in one model)");
            I[i_cursor + ii] = (uint16_t)adj;
        }

        v_cursor += S->vcount;
        i_cursor += S->icount;
    }

    // Cleanup builders
    for (size_t si = 0; si < sub_count; si++) submesh_free(&subs[si]);
    free(subs);
    free_vertex_normals(&vnorm);

    *out_blob = blob;
    *out_len = file_sz;
}

// ---------------------- Alpha IDs list ----------------------

static alpha_id_list_t load_alpha_ids(const char *path) {
    alpha_id_list_t lst;
    memset(&lst, 0, sizeof(lst));
    if (!path) return lst;

    FILE *f = fopen(path, "r");
    if (!f) {
        fprintf(stderr, "Warning: couldn't open alpha ids file %s\n", path);
        return lst;
    }

    size_t cap = 64;
    lst.ids = (uint16_t*)xmalloc(sizeof(uint16_t) * cap);

    while (!feof(f)) {
        unsigned int v = 0;
        if (fscanf(f, "%u", &v) == 1) {
            if (lst.count + 1 > cap) {
                cap *= 2;
                lst.ids = (uint16_t*)xrealloc(lst.ids, sizeof(uint16_t) * cap);
            }
            lst.ids[lst.count++] = (uint16_t)v;
        } else {
            // eat line
            int c;
            do { c = fgetc(f); } while (c != '\n' && c != EOF);
        }
    }
    fclose(f);
    return lst;
}

static void free_alpha_ids(alpha_id_list_t *lst) {
    free(lst->ids);
    memset(lst, 0, sizeof(*lst));
}

// ---------------------- Main ----------------------

static void usage(const char *argv0) {
    fprintf(stderr,
        "Usage:\n"
        "  %s <archive.jag|archive.mem> <out_dir> [options]\n\n"
        "Options:\n"
        "  --rcm1                 Output RCM1 (float positions + UV; GLdc-friendly)\n"
        "  --rcm2                 Output RCM2 (int16 positions + UV; compact canonical)\n"
        "  --rcm3                 Output RCM3 (float positions, NO UV; only untextured models)\n"
        "                         (default is --rcm2)\n"
        "  --alpha-ids file.txt   Texture IDs that should be treated as alpha (one per line)\n"
        "  --flip-uvs             Flip textured RCM1/RCM2 V coordinates inside each atlas region\n"
        "  --coplanar-backface    Preserve old behavior: emit both valid OB3 face sides\n"
        "  --noreduction          Disable vertex deduplication (larger .rcm, easier debugging)\n"
        "  --double-sided         Force single-sided faces to be emitted two-sided\n\n"
        "Example:\n"
        "  %s models36.jag out_rcm --alpha-ids alpha_ids.txt\n",
        argv0, argv0);
}

int main(int argc, char **argv) {
    if (argc < 3) { usage(argv[0]); return 1; }

    const char *in_path = argv[1];
    const char *out_dir = argv[2];
    const char *alpha_path = NULL;
    int reduce_vertices = 1;
    int force_double_sided = 0;
    int flip_uvs = 0;
    int coplanar_backface = 0;
    rcm_out_format_t out_fmt = RCM_OUT_RCM2;
    int out_fmt_set = 0;

    for (int i = 3; i < argc; i++) {
        if (!strcmp(argv[i], "--rcm1")) {
            if (out_fmt_set && out_fmt != RCM_OUT_RCM1) { fprintf(stderr, "Error: choose only one of --rcm1, --rcm2, or --rcm3\n"); return 1; }
            out_fmt = RCM_OUT_RCM1; out_fmt_set = 1;
        } else if (!strcmp(argv[i], "--rcm2")) {
            if (out_fmt_set && out_fmt != RCM_OUT_RCM2) { fprintf(stderr, "Error: choose only one of --rcm1, --rcm2, or --rcm3\n"); return 1; }
            out_fmt = RCM_OUT_RCM2; out_fmt_set = 1;
        } else if (!strcmp(argv[i], "--rcm3")) {
            if (out_fmt_set && out_fmt != RCM_OUT_RCM3) { fprintf(stderr, "Error: choose only one of --rcm1, --rcm2, or --rcm3\n"); return 1; }
            out_fmt = RCM_OUT_RCM3; out_fmt_set = 1;
        } else if (!strcmp(argv[i], "--alpha-ids") && i + 1 < argc) {
            alpha_path = argv[++i];
        } else if (!strcmp(argv[i], "--flip-uvs")) {
            flip_uvs = 1;
        } else if (!strcmp(argv[i], "--coplanar-backface")) {
            coplanar_backface = 1;
        } else if (!strcmp(argv[i], "--noreduction") || !strcmp(argv[i], "--no-reduction")) {
            reduce_vertices = 0;
        } else if (!strcmp(argv[i], "--double-sided") || !strcmp(argv[i], "--two-sided")) {
            force_double_sided = 1;
        } else {
            usage(argv[0]);
            return 1;
        }
    }

    if (ensure_dir(out_dir) != 0) {
        fprintf(stderr, "Failed to create output dir %s\n", out_dir);
        return 1;
    }

    alpha_id_list_t alpha_ids = load_alpha_ids(alpha_path);

    jag_archive_t arch = jag_load(in_path);

    size_t converted = 0;
    for (uint16_t ei = 0; ei < arch.count; ei++) {
        const jag_entry_t *e = &arch.entries[ei];

        size_t entry_len = 0;
        uint8_t *entry = jag_extract_entry(e, &entry_len);
        if (!entry) continue;

        ob3_model_t model;
        if (!ob3_parse(entry, entry_len, &model)) {
            free(entry);
            continue;
        }

        uint8_t *rcm_blob = NULL;
        size_t   rcm_len  = 0;
        convert_ob3_to_rcm(&model, (alpha_ids.count ? &alpha_ids : NULL),
                           reduce_vertices, force_double_sided, flip_uvs,
                           coplanar_backface, out_fmt, &rcm_blob, &rcm_len);

        if (!rcm_blob || rcm_len == 0) {
            if (out_fmt == RCM_OUT_RCM3) {
                const int32_t shash = hash_u32_to_s32(e->hash);
                fprintf(stderr, "[RCM3] Skipping model_%" PRId32 " (textured model rejected)\n", shash);
            }
            ob3_free(&model);
            free(entry);
            continue;
        }



        char out_path[512];
        const int32_t shash = hash_u32_to_s32(e->hash);
        snprintf(out_path, sizeof(out_path), "%s/model_%" PRId32 ".rcm", out_dir, shash);

        if (write_entire_file(out_path, rcm_blob, rcm_len) != 0) {
            fprintf(stderr, "Failed to write %s\n", out_path);
        } else {
            converted++;
        }

        free(rcm_blob);
        ob3_free(&model);
        free(entry);
    }

    printf("Converted %zu models to .rcm\n", converted);

    free_alpha_ids(&alpha_ids);
    jag_free(&arch);
    return 0;
}
