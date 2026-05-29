#ifndef RCM_FONT_H
#define RCM_FONT_H

#ifndef _arch_dreamcast
#include <SDL2/SDL_opengl.h> // PC build
#else
#include <GL/gl.h> //Dreamcast Build
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* Begin/end a UI text pass (2D ortho + sane GL state). */
void font_begin(GLuint tex, int win_w, int win_h);
void font_end(void);

/* Metrics for FONT_BOLD_12. */
int font_bold12_text_width_px(const char *s, int sprites_w, int sprites_h);
int font_bold12_line_height_px(int sprites_w, int sprites_h);

/* Draw a single line of FONT_BOLD_12 text at (x0, top_y). */
void font_draw_bold12(GLuint sprites_tex,
                          int sprites_w, int sprites_h,
                          int win_w, int win_h,
                          int x0, int top_y,
                          const char *text);

#ifdef __cplusplus
}
#endif

#endif /* RCM_FONT_H */

