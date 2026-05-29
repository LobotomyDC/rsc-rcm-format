#ifndef RCM_VIEWER_H
#define RCM_VIEWER_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// RSC uses /100 scaling for model coords in the original pipeline.
#define VERTEX_SCALE   100.0f

// UVs are stored as Q0.15 in [0..uv_divisor]. (Usually 32767.)
#define UV_Q15_DIVISOR 32767.0f

// File alignment used by RCM2 (sections padded so offsets are multiples of this).
#define RCM_FILE_ALIGN 32u

typedef struct atlas_pos {
    float u0, u1, v0, v1;
} atlas_pos_t;

typedef enum rcm_format {
    RCM_FMT_UNKNOWN = 0,
    RCM_FMT_RCM1,
    RCM_FMT_RCM2,
    RCM_FMT_RCM3,
    RCM_FMT_RCL1,
    RCM_FMT_RCW1
} rcm_format_t;

static inline const char *rcm_format_name(rcm_format_t f) {
    switch (f) {
        case RCM_FMT_RCM1: return "RCM1";
        case RCM_FMT_RCM2: return "RCM2";
        case RCM_FMT_RCM3: return "RCM3";
        case RCM_FMT_RCL1: return "RCL1";
        case RCM_FMT_RCW1: return "RCW1";
        default: return "????";
    }
}

/* Atlas rect table (defined in rcm_atlas.c) */
extern const atlas_pos_t g_model_atlas_pos[];
extern const size_t      g_model_atlas_pos_count;

extern const atlas_pos_t g_font_bold12_atlas_pos[95];
extern const atlas_pos_t g_font_bold12_shadow_atlas_pos[95];

/* Special atlas rects used by RSC for “white” / “transparent” model materials */
extern const atlas_pos_t g_white_model_atlas_pos;
extern const atlas_pos_t g_transparent_model_atlas_pos;

enum {
    RCM_SM_ALPHA   = 1u << 0,
    RCM_SM_GOURAUD = 1u << 1,
};

/*
 * RCM2: alignment-friendly on-disk layout.
 *
 * NOTE: Do NOT pack these structs. They are intentionally ordered to avoid
 * implicit compiler padding and keep 32-bit fields naturally aligned.
 */
typedef struct rcm_header {
    char     magic[4];          // "RCM2"
    uint16_t version;           // 2
    uint16_t flags;             // reserved for future

    uint32_t file_size;         // total file bytes including padding to RCM_FILE_ALIGN

    uint32_t vertex_count;
    uint32_t index_count;       // number of uint16 indices
    uint32_t submesh_count;

    uint32_t submesh_off;       // aligned to RCM_FILE_ALIGN
    uint32_t vertex_off;        // aligned to RCM_FILE_ALIGN
    uint32_t index_off;         // aligned to RCM_FILE_ALIGN

    uint16_t vertex_stride;     // 16
    uint16_t index_stride;      // 2 (uint16 indices)

    uint16_t uv_divisor;        // usually 32767
    uint16_t reserved0;

    int16_t  aabb_min[3];
    int16_t  aabb_max[3];
    int16_t  bounds_pad;

    int16_t  pad_align0;        // keeps next uint32_t array 4-byte aligned

    uint32_t reserved1[9];      // pad header to 96 bytes; future expansion
} rcm_header_t;

typedef struct rcm_submesh {
    uint16_t texture_id;       // 0xFFFF = untextured
    uint16_t color_argb1555;   // base tint (textured usually 0xFFFF)
    uint16_t flags;            // RCM_SM_ALPHA, RCM_SM_GOURAUD
    uint16_t reserved0;
    uint32_t first_index;      // index into idx[] (not bytes)
    uint32_t index_count;      // number of indices in this submesh
} rcm_submesh_t;

/*
 * 16-byte vertex.
 * Positions: int16
 * Normals:   snorm8
 * UV:        Q0.15 int16 (0..uv_divisor)
 */
typedef struct rcm_vtx16 {
    int16_t x, y, z;
    int16_t pad0;
    int8_t  nx, ny, nz;
    uint8_t pad1;
    int16_t u, v;
} rcm_vtx16_t;

/*
 * RCM1 vertex (float positions + Q15 UV).
 * Positions are stored as floats in world units (RSC units / 100).
 */
typedef struct rcm_vtxf {
    float  x, y, z;
    int8_t nx, ny, nz;
    uint8_t pad1;
    int16_t u, v;
} rcm_vtxf_t;

/*
 * RCM3 vertex (float positions, no UV).
 * Positions are stored as floats in world units (RSC units / 100).
 */
typedef struct rcm_vtxf_n {
    float  x, y, z;
    int8_t nx, ny, nz;
    uint8_t pad1;
} rcm_vtxf_n_t;

/* RCL1 vertex with per-vertex RGBA and Q15 UV (24 bytes) */
typedef struct rcl_vtxf_cuv {
    float    x, y, z;
    uint32_t rgba;      /* stored as 0xRRGGBBAA by rsc2rcl */
    int8_t   nx, ny, nz;
    uint8_t  pad0;
    int16_t  u, v;
} rcl_vtxf_cuv_t;

/* RCL1 vertex with per-vertex RGBA, no UV (20 bytes) */
typedef struct rcl_vtxf_c {
    float    x, y, z;
    uint32_t rgba;      /* stored as 0xRRGGBBAA by rsc2rcl */
    int8_t   nx, ny, nz;
    uint8_t  pad0;
} rcl_vtxf_c_t;




/* Compile-time sanity checks */
#if defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 201112L)
#define RCM_STATIC_ASSERT(cond, msg) _Static_assert((cond), msg)
#else
#define RCM_STATIC_ASSERT(cond, msg) typedef char static_assertion_##msg[(cond) ? 1 : -1]
#endif

RCM_STATIC_ASSERT(sizeof(rcm_header_t)  == 96, rcm_header_must_be_96_bytes);
RCM_STATIC_ASSERT(sizeof(rcm_submesh_t) == 16, rcm_submesh_must_be_16_bytes);
RCM_STATIC_ASSERT(sizeof(rcm_vtx16_t)   == 16, rcm_vertex_must_be_16_bytes);
RCM_STATIC_ASSERT(sizeof(rcm_vtxf_t)   == 20, rcm_vertexf_must_be_20_bytes);
RCM_STATIC_ASSERT(sizeof(rcm_vtxf_n_t) == 16, rcm_vertexf_n_must_be_16_bytes);
RCM_STATIC_ASSERT(sizeof(rcl_vtxf_cuv_t) == 24, rcl_vertex_cuv_must_be_24_bytes);
RCM_STATIC_ASSERT(sizeof(rcl_vtxf_c_t)   == 20, rcl_vertex_c_must_be_20_bytes);

typedef struct rcm_model {
    rcm_header_t   hdr;        // cached copy of header
    rcm_format_t   fmt;        // derived from hdr.magic

    uint8_t       *blob;       // owning pointer to entire file (aligned if possible)
    size_t         blob_size;

    rcm_submesh_t *sub;        // points into blob
    uint8_t      *vtx;        // points into blob (format-dependent)
    uint16_t     *idx;        // points into blob
} rcm_model_t;

#ifdef __cplusplus
} // extern "C"
#endif

#endif // RCM_VIEWER_H
