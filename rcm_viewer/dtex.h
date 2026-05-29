/* dtex.h */
#ifndef DTEX_H
#define DTEX_H

#include <stdint.h>
#include <stddef.h>
#include <GL/gl.h>
#if defined(_arch_dreamcast)
  #define DTEX_DREAMCAST 1
#else
  #define DTEX_DREAMCAST 0
#endif

/* Forward-declare GLuint without forcing GL headers into everyone */
#ifndef GLuint
typedef unsigned int GLuint;
#endif

#pragma pack(push, 1)
typedef struct dtex_header {
    char     fourcc[4];       /* "DcTx" */
    uint32_t chunk_size;      /* total bytes including header, padded to 32 */
    uint8_t  version;
    uint8_t  header_size;     /* in 32-byte units minus 1 (0 => 32 bytes) */
    uint8_t  codebook_size;   /* VQ entries minus 1 (255 => 256 entries) */
    uint8_t  reserved0;
    uint16_t width_pixels;
    uint16_t height_pixels;
    uint32_t pvr_type;
    uint8_t  reserved1[32 - 4 - 4 - 1 - 1 - 1 - 1 - 2 - 2 - 4];
} dtex_header_t;
#pragma pack(pop)

typedef struct dtex_image {
    dtex_header_t  hdr;
    uint8_t       *blob;
    size_t         blob_size;
    const uint8_t *pvr_data;
    size_t         pvr_size;
} dtex_image_t;

int  dtex_load_file(const char *path, dtex_image_t *out);
void dtex_free(dtex_image_t *img);

#ifdef _arch_dreamcast
int dtex_load_gl_texture(const char *path, GLuint *out_tex, int *out_w, int *out_h);
#endif

#endif

