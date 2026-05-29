#include "dtex.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _arch_dreamcast
#include <malloc.h>
#include <GL/gl.h>
#include <GL/glkos.h>
#include <GL/glext.h>
#endif

#ifndef DTEX_FILE_ALIGN
#define DTEX_FILE_ALIGN 32u
#endif

/* PVR type bits (from KOS / pvrtex conventions) */
#define DTEX_VQ_SHIFT           30
#define DTEX_PIXELFMT_SHIFT     27
#define DTEX_NOT_TWIDDLED_SHIFT 26

#define DTEX_PIXELFMT_ARGB1555  0
#define DTEX_PIXELFMT_RGB565    1
#define DTEX_PIXELFMT_ARGB4444  2

void glCompressedTexImage2DARB(GLenum target, GLint level, GLenum internalformat,
                               GLsizei width, GLsizei height, GLint border,
                               GLsizei imageSize, const GLvoid *data);

static void *dtex_aligned_alloc(size_t align, size_t size) {
#if defined(_MSC_VER)
    return _aligned_malloc(size, align);
#elif defined(_arch_dreamcast)
    return memalign(align, size);
#else
    void *p = NULL;
    if (posix_memalign(&p, align, size) != 0) return NULL;
    return p;
#endif
}

static void dtex_aligned_free(void *p) {
#if defined(_MSC_VER)
    _aligned_free(p);
#else
    free(p);
#endif
}

static int dtex_read_whole_file_aligned(const char *path, uint8_t **out_buf, size_t *out_sz) {
    *out_buf = NULL;
    *out_sz  = 0;

    FILE *f = fopen(path, "rb");
    if (!f) return 0;

    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return 0; }
    long sz_l = ftell(f);
    if (sz_l <= 0) { fclose(f); return 0; }
    size_t sz = (size_t)sz_l;
    if (fseek(f, 0, SEEK_SET) != 0) { fclose(f); return 0; }

    /* Pad to 32 bytes so PVR payload stays aligned. */
    size_t padded = (sz + (DTEX_FILE_ALIGN - 1)) & ~(DTEX_FILE_ALIGN - 1);

    uint8_t *buf = (uint8_t*)dtex_aligned_alloc(DTEX_FILE_ALIGN, padded);
    if (!buf) { fclose(f); return 0; }
    memset(buf, 0, padded);

    if (fread(buf, 1, sz, f) != sz) {
        fclose(f);
        dtex_aligned_free(buf);
        return 0;
    }

    fclose(f);
    *out_buf = buf;
    *out_sz  = padded;
    return 1;
}

static size_t dtex_header_bytes(const dtex_header_t *h) {
    /* header_size is in 32-byte units, minus 1 */
    return (size_t)(h->header_size + 1u) * 32u;
}

int dtex_load_file(const char *path, dtex_image_t *out) {
    memset(out, 0, sizeof(*out));

    uint8_t *blob = NULL;
    size_t blob_sz = 0;
    if (!dtex_read_whole_file_aligned(path, &blob, &blob_sz)) return 0;
    if (blob_sz < sizeof(dtex_header_t)) {
        dtex_aligned_free(blob);
        return 0;
    }

    dtex_header_t h;
    memcpy(&h, blob, sizeof(h));

    if (memcmp(h.fourcc, "DcTx", 4) != 0) {
        dtex_aligned_free(blob);
        return 0;
    }

    size_t hdr_bytes = dtex_header_bytes(&h);
    if (hdr_bytes < 32u || hdr_bytes > blob_sz) {
        dtex_aligned_free(blob);
        return 0;
    }

    size_t declared = (size_t)h.chunk_size;
    if (declared < hdr_bytes) {
        dtex_aligned_free(blob);
        return 0;
    }

    /* Clamp to actual read size (blob may be padded). */
    size_t total = declared;
    if (total > blob_sz) total = blob_sz;

    const uint8_t *pvr = blob + hdr_bytes;
    size_t pvr_sz = (total > hdr_bytes) ? (total - hdr_bytes) : 0;

    out->hdr = h;
    out->blob = blob;
    out->blob_size = blob_sz;
    out->pvr_data = pvr;
    out->pvr_size = pvr_sz;
    return 1;
}

void dtex_free(dtex_image_t *img) {
    if (!img) return;
    if (img->blob) dtex_aligned_free(img->blob);
    memset(img, 0, sizeof(*img));
}

static int dtex_is_vq(uint32_t pvr_type) {
    return (int)((pvr_type >> DTEX_VQ_SHIFT) & 1u);
}

static int dtex_is_twiddled(uint32_t pvr_type) {
    /* bit 26: NOT twiddled (0 => twiddled) */
    return (((pvr_type >> DTEX_NOT_TWIDDLED_SHIFT) & 1u) == 0u);
}

static int dtex_pixfmt(uint32_t pvr_type) {
    return (int)((pvr_type >> DTEX_PIXELFMT_SHIFT) & 7u);
}

static GLenum dtex_gl_internalformat_from_pvr(uint32_t pvr_type) {
    const int vq = dtex_is_vq(pvr_type);
    const int tw = dtex_is_twiddled(pvr_type);
    const int pf = dtex_pixfmt(pvr_type);

    if (vq) {
        if (tw) {
            switch (pf) {
                case DTEX_PIXELFMT_ARGB1555: return GL_COMPRESSED_ARGB_1555_VQ_TWID_KOS;
                case DTEX_PIXELFMT_RGB565:   return GL_COMPRESSED_RGB_565_VQ_TWID_KOS;
                case DTEX_PIXELFMT_ARGB4444: return GL_COMPRESSED_ARGB_4444_VQ_TWID_KOS;
                default: return 0;
            }
        } else {
            switch (pf) {
                case DTEX_PIXELFMT_ARGB1555: return GL_COMPRESSED_ARGB_1555_VQ_KOS;
                case DTEX_PIXELFMT_RGB565:   return GL_COMPRESSED_RGB_565_VQ_KOS;
                case DTEX_PIXELFMT_ARGB4444: return GL_COMPRESSED_ARGB_4444_VQ_KOS;
                default: return 0;
            }
        }
    }

    return 0;
}

int dtex_load_gl_texture(const char *path, GLuint *out_tex, int *out_w, int *out_h) {
    if (out_tex) *out_tex = 0;
    if (out_w)   *out_w   = 0;
    if (out_h)   *out_h   = 0;

#ifndef _arch_dreamcast
    /* PC build: DTEX not supported in this project anymore. */
    (void)path;
    return 0;
#else
    dtex_image_t img;
    if (!dtex_load_file(path, &img)) return 0;

    const int w = (int)img.hdr.width_pixels;
    const int h = (int)img.hdr.height_pixels;

    if (w <= 0 || h <= 0 || img.pvr_size == 0 || !img.pvr_data) {
        dtex_free(&img);
        return 0;
    }

    /* Your pipeline produces VQ textures; keep it strict. */
    if (!dtex_is_vq(img.hdr.pvr_type)) {
        dtex_free(&img);
        return 0;
    }

    /* KOS/GLdc compressed internalformat (VQ + twiddled/non-twiddled + pixfmt) */
    const GLenum ifmt = dtex_gl_internalformat_from_pvr(img.hdr.pvr_type);
    if (!ifmt) {
        dtex_free(&img);
        return 0;
    }

    GLuint tex = 0;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

#if defined(GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
#else
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
#endif

    /* GLdc exposes the ARB entrypoint name. */
    glCompressedTexImage2DARB(GL_TEXTURE_2D, 0, ifmt, w, h, 0,
                              (GLsizei)img.pvr_size, img.pvr_data);

    dtex_free(&img);

    if (out_tex) *out_tex = tex;
    if (out_w)   *out_w   = w;
    if (out_h)   *out_h   = h;
    return 1;
#endif
}

