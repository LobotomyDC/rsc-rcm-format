// rcm_viewer.c - RuneCast Model (.rcm) viewer (GL1.1 + SDL2, Linux)
// Loads RCM1/RCM2/RCM3 files produced by rsc2rcm, and applies model_textures atlas (where applicable).
//
// Build:
//   sudo apt install libsdl2-dev (if not installed)
//   curl -L -o stb_image.h https://raw.githubusercontent.com/nothings/stb/master/stb_image.h
//   gcc -std=c99 -O2 -Wall -Wextra -D_POSIX_C_SOURCE=200112L -I. rcm_viewer.c rcm_atlas.c rcm_font.c dtex.c $(sdl2-config --cflags --libs) -lGL -lm -o rcm_viewer
//
// Run:
//   ./rcm_viewer <model.rcm|directory> [--atlas textures/model_textures.png]
//
// Controls:
//   LMB drag: rotate   | Wheel: zoom
//   Left/Right arrows: previous/next model (when a directory is provided, or when
//                       a file is provided and its directory contains .rcm files)
//   W: wireframe       | L: toggle lighting | B: toggle blending
//   ESC/Q: quit
//
// Notes:
// - GL1.1 compliant: uses client arrays, no shaders/VBOs.
// - UVs are Q0.15 (0..32767). We feed GL_SHORT texcoords and scale via GL_TEXTURE matrix.
#include "rcm_viewer.h"
#include "rcm_font.h"

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

#include <inttypes.h>

#include <dirent.h>
#include <sys/stat.h>

#include <SDL2/SDL.h>
#include <GL/gl.h>

#ifdef _arch_dreamcast
#include <GL/glkos.h>
#include <malloc.h>
#include <dc/maple.h>
#include <dc/maple/controller.h>
#endif


#include <limits.h>

#ifdef _arch_dreamcast
#include "dtex.h"
#endif

#ifndef _arch_dreamcast
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
//----------------------------Forward Declarations----------------------------
/* --- forward decls for overlay helpers (must appear before first use) --- */
static const atlas_pos_t *viewer_atlas_pos_for_texture_id(uint16_t texture_id);

/*static size_t viewer_collect_model_texture_ids(const rcm_model_t *m,
                                               uint16_t *out_ids,
                                               size_t cap);*/

static const char *viewer_submesh_flag_string(uint16_t flags);


static void viewer_draw_model_overlay(GLuint sprites_tex,
                                      int sprites_w, int sprites_h,
                                      int win_w, int win_h,
                                      const char *model_paths,
                                      size_t model_index,
                                      size_t model_total,
                                      const rcm_model_t *model);

static int load_model_by_index(rcm_model_t *model, const char *path,
                               float *out_dist, SDL_Window *win,
                               size_t idx, size_t total);

#ifdef _arch_dreamcast
static void dc_free_draw_buffers(void);
static int  dc_build_draw_buffers(const rcm_model_t *m);
#endif



#if defined(__GNUC__)
#define RCM_UNUSED __attribute__((unused))
#else
#define RCM_UNUSED
#endif

static rcm_format_t rcm_format_from_magic4(const char magic[4]) {
    if (!magic) return RCM_FMT_UNKNOWN;
    if (memcmp(magic, "RCM1", 4) == 0) return RCM_FMT_RCM1;
    if (memcmp(magic, "RCM2", 4) == 0) return RCM_FMT_RCM2;
    if (memcmp(magic, "RCM3", 4) == 0) return RCM_FMT_RCM3;
    if (memcmp(magic, "RCL1", 4) == 0) return RCM_FMT_RCL1;
    if (memcmp(magic, "RCW1", 4) == 0) return RCM_FMT_RCW1;
    return RCM_FMT_UNKNOWN;
}

static int model_has_uv(const rcm_model_t *m) {
    if (!m) return 0;
    switch (m->fmt) {
        case RCM_FMT_RCM1:
            return 1;

        case RCM_FMT_RCW1:
            return (m->hdr.vertex_stride == (uint16_t)sizeof(rcm_vtxf_t)) &&
                   (m->hdr.uv_divisor != 0);

        case RCM_FMT_RCM2:
            return 1;

        case RCM_FMT_RCL1:
            return (m->hdr.vertex_stride == (uint16_t)sizeof(rcl_vtxf_cuv_t)) &&
                   (m->hdr.uv_divisor != 0);

        default:
            return 0;
    }
}

static float rcm_draw_scale_to_world(const rcm_model_t *m) {
    /* RCM2 stores positions in RSC units (scaled by 100). RCM1/RCM3 store world floats already. */
    return (m && m->fmt == RCM_FMT_RCM2) ? (1.0f / VERTEX_SCALE) : 1.0f;
}

static float rcm_center_units_scale(const rcm_model_t *m) {
    /* Convert header AABB (always in RSC units) to the same units as the vertex positions. */
    return (m && m->fmt == RCM_FMT_RCM2) ? 1.0f : (1.0f / VERTEX_SCALE);
}

//----------------------------Font/UI Draw------------------------------------
static void viewer_draw_model_overlay(GLuint sprites_tex,
                                      int sprites_w, int sprites_h,
                                      int win_w, int win_h,
                                      const char *model_paths,
                                      size_t model_index,
                                      size_t model_total,
                                      const rcm_model_t *model) {
    if (!sprites_tex || !model || !model_paths) return;

    const char *base = model_paths;
    const char *slash = strrchr(model_paths, '/');
    if (slash) base = slash + 1;

    char line[256];

    const int x0 = 8;
    const int margin_y = 8;
    const int line_h   = font_bold12_line_height_px(sprites_w, sprites_h);
    const int step_y   = line_h + 2;

    int y = margin_y;

    snprintf(line, sizeof(line), "Model: %s", base);
    font_draw_bold12(sprites_tex, sprites_w, sprites_h, win_w, win_h, x0, y, line); y += step_y;

    snprintf(line, sizeof(line), "[%zu/%zu]", (size_t)(model_index + 1), (size_t)model_total);
    font_draw_bold12(sprites_tex, sprites_w, sprites_h, win_w, win_h, x0, y, line); y += step_y;

    snprintf(line, sizeof(line), "Format: %s", rcm_format_name(model->fmt));
    font_draw_bold12(sprites_tex, sprites_w, sprites_h, win_w, win_h, x0, y, line); y += step_y;

    snprintf(line, sizeof(line), "Vertices: %u", (unsigned)model->hdr.vertex_count);
    font_draw_bold12(sprites_tex, sprites_w, sprites_h, win_w, win_h, x0, y, line); y += step_y;

    snprintf(line, sizeof(line), "Indices: %u", (unsigned)model->hdr.index_count);
    font_draw_bold12(sprites_tex, sprites_w, sprites_h, win_w, win_h, x0, y, line); y += step_y;

    snprintf(line, sizeof(line), "Submeshes: %u", (unsigned)model->hdr.submesh_count);
    font_draw_bold12(sprites_tex, sprites_w, sprites_h, win_w, win_h, x0, y, line); y += step_y;

    /* If triangles, show polys */
    if ((model->hdr.index_count % 3u) == 0u) {
        snprintf(line, sizeof(line), "Polygons: %u", (unsigned)(model->hdr.index_count / 3u));
        font_draw_bold12(sprites_tex, sprites_w, sprites_h, win_w, win_h, x0, y, line); y += step_y;
    }

    /* ---- Submesh breakdown ---- */
    font_draw_bold12(sprites_tex, sprites_w, sprites_h, win_w, win_h, x0, y, "Submesh list:"); y += step_y;

    /* Don’t spam off-screen; cap by remaining height */
    const int max_lines = (win_h - y - 8) / step_y;
    int lines_used = 0;

    for (uint32_t i = 0; i < model->hdr.submesh_count; ++i) {
        if (lines_used >= max_lines - 1) {
            font_draw_bold12(sprites_tex, sprites_w, sprites_h, win_w, win_h, x0, y, "...");
            break;
        }

        const rcm_submesh_t *sm = &model->sub[i];
        const uint32_t idxc = sm->index_count;
        const uint32_t tris = (idxc % 3u) ? 0u : (idxc / 3u);

        if (sm->texture_id == 0xFFFFu) {
            /* Untextured: show tint + poly count */
            snprintf(line, sizeof(line),
                     "SM%u: untextured  poly:%u  idx:%u  tint:0x%04X  fl:%s",
                     (unsigned)i, (unsigned)tris, (unsigned)idxc,
                     (unsigned)sm->color_argb1555,
                     viewer_submesh_flag_string(sm->flags));
        } else {
            const atlas_pos_t *r = viewer_atlas_pos_for_texture_id(sm->texture_id);
            if (r) {
                snprintf(line, sizeof(line),
                         "SM%u: tex:%u  poly:%u  idx:%u  u[%.3f..%.3f] v[%.3f..%.3f]  fl:%s",
                         (unsigned)i, (unsigned)sm->texture_id,
                         (unsigned)tris, (unsigned)idxc,
                         r->u0, r->u1, r->v0, r->v1,
                         viewer_submesh_flag_string(sm->flags));
            } else {
                snprintf(line, sizeof(line),
                         "SM%u: tex:%u  poly:%u  idx:%u  (no atlas entry)  fl:%s",
                         (unsigned)i, (unsigned)sm->texture_id,
                         (unsigned)tris, (unsigned)idxc,
                         viewer_submesh_flag_string(sm->flags));
            }
        }

        font_draw_bold12(sprites_tex, sprites_w, sprites_h, win_w, win_h, x0, y, line);
        y += step_y;
        lines_used++;
    }
}

static void viewer_draw_controls_hint_bottom_center(GLuint sprites_tex,
                                                    int sprites_w, int sprites_h,
                                                    int win_w, int win_h) {
    if (!sprites_tex) return;

    #ifdef _arch_dreamcast
    const char *line1 =
        "L-Stick: Orbit Cam    R-Stick: Orbit Light    LT/RT: Zoom Out/In";
    const char *line2 =
        "D-Pad L/R: Prev/Next    A: Toggle Lighting    Y: Reset    Start: Quit";
#else
    const char *line1 =
        "RMB/LMB+Drag: Orbit Camera/Light    Mousewheel: Zoom    L/R Arrow: Prev/Next";
    const char *line2 =
        "R/W/L/ESC: Reset Light and Camera/Wireframe/Lights/Quit";
#endif

const int margin = 8;
    const int line_h = font_bold12_line_height_px(sprites_w, sprites_h);
    const int step_y = line_h + 2;

    /* place two lines at bottom; line2 is the bottom-most */
    const int top_y2 = win_h - margin - line_h;
    const int top_y1 = top_y2 - step_y;

    int tw1 = font_bold12_text_width_px(line1, sprites_w, sprites_h);
    int x1  = (win_w - tw1) / 2;
    if (x1 < 8) x1 = 8;

    int tw2 = font_bold12_text_width_px(line2, sprites_w, sprites_h);
    int x2  = (win_w - tw2) / 2;
    if (x2 < 8) x2 = 8;

    font_draw_bold12(sprites_tex, sprites_w, sprites_h, win_w, win_h, x1, top_y1, line1);
    font_draw_bold12(sprites_tex, sprites_w, sprites_h, win_w, win_h, x2, top_y2, line2);
}


//-------------------------------FPS COUNTER----------------------------------------------
typedef struct viewer_fps_counter {
    uint64_t last;
    uint64_t accum;
    uint32_t frames;
    float    fps;
    uint64_t freq;
} viewer_fps_counter;

static void viewer_fps_init(viewer_fps_counter *fc) {
    memset(fc, 0, sizeof(*fc));
    fc->freq = (uint64_t)SDL_GetPerformanceFrequency();
    fc->last = (uint64_t)SDL_GetPerformanceCounter();
    fc->fps  = 0.0f;
}

static void viewer_fps_tick(viewer_fps_counter *fc) {
    const uint64_t now = (uint64_t)SDL_GetPerformanceCounter();
    const uint64_t dt  = now - fc->last;
    fc->last = now;

    fc->accum  += dt;
    fc->frames += 1;

    /* update ~4x/sec for stability */
    const uint64_t quarter_sec = fc->freq / 4u;
    if (fc->accum >= quarter_sec && fc->freq) {
        const double secs = (double)fc->accum / (double)fc->freq;
        fc->fps = (secs > 0.0) ? (float)((double)fc->frames / secs) : 0.0f;
        fc->accum = 0;
        fc->frames = 0;
    }
}

static void viewer_draw_fps_top_right_left(GLuint sprites_tex,
                                           int sprites_w, int sprites_h,
                                           int win_w, int win_h,
                                           const viewer_fps_counter *fc) {
    if (!sprites_tex || !fc) return;

    char buf[32];
    snprintf(buf, sizeof(buf), "FPS: %.1f", (double)fc->fps);

    /* reserve a stable box so the text does NOT jitter with digit width changes */
    static int reserved_w = -1;
    if (reserved_w < 0) {
        reserved_w = font_bold12_text_width_px("FPS: 999.9", sprites_w, sprites_h);
    }

    const int margin = 8;
    const int x0 = win_w - margin - reserved_w;
    const int y0 = margin;

    font_draw_bold12(sprites_tex, sprites_w, sprites_h, win_w, win_h,
                                 x0, y0, buf);
}


//-------------------------------LIGHTING-----------------------------------------------------------
static float g_light_yaw_deg_default   = -135.0f; 
static float g_light_pitch_deg_default = -13.0f;
/* Matches old {-0.6,-0.2,-0.6} roughly. Good enough for the girls we date, anyway. */

static float g_light_yaw_deg   = -135.0f;
static float g_light_pitch_deg = -13.0f;

typedef struct viewer_controls {
    int cam_down;
    int light_down;
    int last_cam_mx, last_cam_my;
    int last_light_mx, last_light_my;
} viewer_controls;

/* used by draw_model() / apply_q15_texcoord_transform() */
static int g_flip_v_local = 1;

static inline void reset_light_to_default(void) {
    g_light_yaw_deg   = g_light_yaw_deg_default;
    g_light_pitch_deg = g_light_pitch_deg_default;
}

static inline void viewer_reset_all(float *yaw, float *pitch, float *dist, float dist_default) {
    *yaw = 35.0f;
    *pitch = 20.0f;
    *dist = dist_default;
    reset_light_to_default();
}


#ifdef _arch_dreamcast
static void viewer_handle_dc_controller(int *running,
                                       float *yaw, float *pitch,
                                       float *dist, float *dist_default,
                                       int *enable_light,
                                       size_t *current_idx, size_t model_count, char **model_paths,
                                       rcm_model_t *model, SDL_Window *win) {
    if (!running || !yaw || !pitch || !dist || !dist_default ||
        !enable_light || !current_idx || !model) {
        return;
    }

    maple_device_t *dev = maple_enum_type(0, MAPLE_FUNC_CONTROLLER);
    if (!dev) return;

    cont_state_t *st = (cont_state_t *)maple_dev_status(dev);
    if (!st) return;

    /* -------- Orbit (left stick) -------- */
    {
        int joyx = (int)(int8_t)st->joyx;
        int joyy = (int)(int8_t)st->joyy;

        const int deadzone = 12;
        if (joyx > -deadzone && joyx < deadzone) joyx = 0;
        if (joyy > -deadzone && joyy < deadzone) joyy = 0;

        /* degrees per frame per unit (-128..127) */
        const float joy_sens = 0.02f;

        *yaw   += (float)joyx * joy_sens;
        *pitch += (float)joyy * joy_sens;

        if (*pitch >  89.0f) *pitch =  89.0f;
        if (*pitch < -89.0f) *pitch = -89.0f;
    }


    /* -------- Orbit light (right stick) -------- */
    {
        int joyx = (int)(int8_t)st->joy2x;
        int joyy = (int)(int8_t)st->joy2y;

        const int deadzone = 12;
        if (joyx > -deadzone && joyx < deadzone) joyx = 0;
        if (joyy > -deadzone && joyy < deadzone) joyy = 0;

        /* degrees per frame per unit (-128..127) */
        const float joy_sens = 0.1f;

        g_light_yaw_deg   += (float)joyx * joy_sens;
        g_light_pitch_deg += (float)joyy * joy_sens;

        if (g_light_pitch_deg >  89.0f) g_light_pitch_deg =  89.0f;
        if (g_light_pitch_deg < -89.0f) g_light_pitch_deg = -89.0f;
    }

    /* -------- Zoom (L/R triggers) --------
       RT zooms in  (decrease dist)
       LT zooms out (increase dist)
    */
    {
        int lt = (int)st->ltrig;
        int rt = (int)st->rtrig;

        const int trig_deadzone = 6;
        if (lt < trig_deadzone) lt = 0;
        if (rt < trig_deadzone) rt = 0;

        /* multiplicative per-frame amount at full trigger */
        const float zoom_sens = 0.020f;

        if (rt) {
            const float t = (float)rt / 255.0f;
            *dist *= 1.0f - (zoom_sens * t);
        }
        if (lt) {
            const float t = (float)lt / 255.0f;
            *dist *= 1.0f + (zoom_sens * t);
        }

        if (*dist < 0.2f) *dist = 0.2f;
    }

    /* -------- Buttons (edge-triggered) -------- */
    {
        /* KOS controller driver normalizes the raw Maple bits so that:
           - (state->buttons & CONT_*) != 0 means "pressed"
           This is the convention used throughout KOS examples/docs. */
        static uint32_t prev_buttons = 0;
        static int prev_valid = 0;

        const uint32_t btn = st->buttons;

        if (prev_valid) {
            const int a_pressed      = (btn & CONT_A)          && !(prev_buttons & CONT_A);
            const int y_pressed      = (btn & CONT_Y)          && !(prev_buttons & CONT_Y);
            const int start_pressed  = (btn & CONT_START)      && !(prev_buttons & CONT_START);
            const int left_pressed   = (btn & CONT_DPAD_LEFT)  && !(prev_buttons & CONT_DPAD_LEFT);
            const int right_pressed  = (btn & CONT_DPAD_RIGHT) && !(prev_buttons & CONT_DPAD_RIGHT);

            if (a_pressed) {
                *enable_light = !*enable_light;
            }

            if (y_pressed) {
                viewer_reset_all(yaw, pitch, dist, *dist_default);
            }

            if (start_pressed) {
                *running = 0;
                prev_buttons = btn;
                return;
            }

            if (model_count > 1 && model_paths) {
                if (left_pressed) {
                    *current_idx = (*current_idx == 0) ? (model_count - 1) : (*current_idx - 1);
                    if (load_model_by_index(model, model_paths[*current_idx], dist,
                                            win, *current_idx, model_count)) {
                        *dist_default = *dist;
                        reset_light_to_default();
                    }
                } else if (right_pressed) {
                    *current_idx = (*current_idx + 1) % model_count;
                    if (load_model_by_index(model, model_paths[*current_idx], dist,
                                            win, *current_idx, model_count)) {
                        *dist_default = *dist;
                        reset_light_to_default();
                    }
                }
            }
        } else {
            /* Prime edge detection with the first observed button state so we don't
               accidentally trigger actions on the very first frame. */
            prev_valid = 1;
        }

        prev_buttons = btn;
    }
}
#endif

//------------------------------CONTROLS-----------------------------------------------------
static void viewer_handle_event(const SDL_Event *e,
                                int *running,
                                viewer_controls *ctl,
                                float *yaw, float *pitch,
                                float *dist, float *dist_default,
                                int *wireframe, int *enable_light, int *enable_blend,
                                size_t *current_idx, size_t model_count, char **model_paths,
                                rcm_model_t *model, SDL_Window *win) {
    switch (e->type) {
        case SDL_QUIT:
            *running = 0;
            return;

        case SDL_MOUSEBUTTONDOWN:
            if (e->button.button == SDL_BUTTON_RIGHT) {
                ctl->cam_down = 1;
                ctl->last_cam_mx = e->button.x;
                ctl->last_cam_my = e->button.y;
            } else if (e->button.button == SDL_BUTTON_LEFT) {
                ctl->light_down = 1;
                ctl->last_light_mx = e->button.x;
                ctl->last_light_my = e->button.y;
            }
            return;

        case SDL_MOUSEBUTTONUP:
            if (e->button.button == SDL_BUTTON_RIGHT) ctl->cam_down = 0;
            if (e->button.button == SDL_BUTTON_LEFT)  ctl->light_down = 0;
            return;

        case SDL_MOUSEMOTION: {
            const int mx = e->motion.x;
            const int my = e->motion.y;

            if (ctl->cam_down) {
                const int dx = mx - ctl->last_cam_mx;
                const int dy = my - ctl->last_cam_my;
                ctl->last_cam_mx = mx;
                ctl->last_cam_my = my;

                *yaw   += dx * 0.35f;
                *pitch += dy * 0.35f;
                if (*pitch >  89.0f) *pitch =  89.0f;
                if (*pitch < -89.0f) *pitch = -89.0f;
            }

            if (ctl->light_down) {
                const int dx = mx - ctl->last_light_mx;
                const int dy = my - ctl->last_light_my;
                ctl->last_light_mx = mx;
                ctl->last_light_my = my;

                g_light_yaw_deg   += dx * 0.35f;
                g_light_pitch_deg += dy * 0.35f;
                if (g_light_pitch_deg >  89.0f) g_light_pitch_deg =  89.0f;
                if (g_light_pitch_deg < -89.0f) g_light_pitch_deg = -89.0f;
            }
            return;
        }

        case SDL_MOUSEWHEEL:
            if (e->wheel.y > 0) *dist *= 0.9f;
            if (e->wheel.y < 0) *dist *= 1.1f;
            if (*dist < 0.2f) *dist = 0.2f;
            return;

        case SDL_KEYDOWN:
            switch (e->key.keysym.sym) {
                case SDLK_ESCAPE:
                case SDLK_q:
                    *running = 0;
                    return;

                case SDLK_w:
                    *wireframe = !*wireframe;
                    return;

                case SDLK_l:
                    *enable_light = !*enable_light;
                    return;

                case SDLK_b:
                    *enable_blend = !*enable_blend;
                    return;

                case SDLK_r:
                    viewer_reset_all(yaw, pitch, dist, *dist_default);
                    return;

                case SDLK_LEFT:
                    if (model_count > 1) {
                        *current_idx = (*current_idx == 0) ? (model_count - 1) : (*current_idx - 1);
                        if (load_model_by_index(model, model_paths[*current_idx], dist,
                                                win, *current_idx, model_count)) {
                            *dist_default = *dist;
                            reset_light_to_default();
                        }
                    }
                    return;

                case SDLK_RIGHT:
                    if (model_count > 1) {
                        *current_idx = (*current_idx + 1) % model_count;
                        if (load_model_by_index(model, model_paths[*current_idx], dist,
                                                win, *current_idx, model_count)) {
                            *dist_default = *dist;
                            reset_light_to_default();
                        }
                    }
                    return;

                default:
                    return;
            }

        default:
            return;
    }
}

//---------------------------TEXTURE ATLAS GRABBER------------------------------
/* Returns a short flag string like "AG" or "-" */
static const char *viewer_submesh_flag_string(uint16_t flags) {
    static char buf[32];

    /* Common case */
    if (flags == 0) {
        strcpy(buf, "-");
        return buf;
    }

    buf[0] = 0;
    int first = 1;

    if (flags & RCM_SM_ALPHA) {
        strcat(buf, first ? "ALPHA" : "|ALPHA");
        first = 0;
    }
    if (flags & RCM_SM_GOURAUD) {
        strcat(buf, first ? "GOURAUD" : "|GOURAUD");
        first = 0;
    }

    /* If anything unknown remains, append it */
    {
        uint16_t known = (uint16_t)(RCM_SM_ALPHA | RCM_SM_GOURAUD);
        uint16_t unk = (uint16_t)(flags & (uint16_t)~known);
        if (unk) {
            char tmp[16];
            snprintf(tmp, sizeof(tmp), "%s0x%X", first ? "" : "|", (unsigned)unk);
            strcat(buf, tmp);
        }
    }

    return buf;
}

static const atlas_pos_t *viewer_atlas_pos_for_texture_id(uint16_t texture_id) {
    /* Regular atlas entries */
    if ((size_t)texture_id < g_model_atlas_pos_count) {
        return &g_model_atlas_pos[texture_id];
    }

    // RSC special materials
    if (texture_id == 0xFFFEu) return &g_white_model_atlas_pos;
    if (texture_id == 0xFFFDu) return &g_transparent_model_atlas_pos;

    return NULL;
}

/*static int viewer_u16_seen(const uint16_t *arr, size_t n, uint16_t v) {
    for (size_t i = 0; i < n; ++i) {
        if (arr[i] == v) return 1;
    }
    return 0;
}

// Collect unique texture IDs referenced by submeshes.
// Returns count written to out_ids (up to cap). 
static size_t viewer_collect_model_texture_ids(const rcm_model_t *m,
                                              uint16_t *out_ids, size_t cap) {
    size_t n = 0;
    if (!m || !m->sub || cap == 0) return 0;

    for (uint32_t i = 0; i < (uint32_t)m->hdr.submesh_count; ++i) {
        uint16_t tid = m->sub[i].texture_id;

        // include untextured too if you want to see it in the list
        // if (tid == 0xFFFFu) continue;

        if (!viewer_u16_seen(out_ids, n, tid)) {
            if (n < cap) out_ids[n++] = tid;
            else break;
        }
    }

    return n;
}
*/
//------------------------ERROR HANDLING---------------------------
static void die(const char *msg) {
    fprintf(stderr, "FATAL: %s\n", msg);
    exit(1);
}

static char *strdup_safe(const char *s) {
    size_t n = strlen(s);
    char *p = (char*)malloc(n + 1);
    if (!p) die("out of memory");
    memcpy(p, s, n + 1);
    return p;
}

//------------------------FILESYSTEM STUFF--------------------------
static const char *path_basename(const char *p) {
    const char *s = strrchr(p, '/');
    return s ? (s + 1) : p;
}

static char *path_dirname_dup(const char *p) {
    const char *slash = strrchr(p, '/');
    if (!slash) return strdup_safe(".");
    size_t n = (size_t)(slash - p);
    if (n == 0) n = 1; // root "/"
    char *d = (char*)malloc(n + 1);
    if (!d) die("out of memory");
    memcpy(d, p, n);
    d[n] = 0;
    return d;
}

static char *path_join_dup(const char *dir, const char *file) {
    size_t dl = strlen(dir);
    size_t fl = strlen(file);
    int need_slash = (dl > 0 && dir[dl - 1] != '/');
    size_t n = dl + (need_slash ? 1 : 0) + fl;
    char *out = (char*)malloc(n + 1);
    if (!out) die("out of memory");
    memcpy(out, dir, dl);
    size_t off = dl;
    if (need_slash) out[off++] = '/';
    memcpy(out + off, file, fl);
    out[n] = 0;
    return out;
}

static int is_dir_path(const char *path) {
#ifdef _arch_dreamcast
    DIR *d = opendir(path);
    if (d) { closedir(d); return 1; }
    return 0;
#else
    struct stat st;
    if (stat(path, &st) != 0) return 0;
    return S_ISDIR(st.st_mode);
#endif
}

static int is_regular_file(const char *path) {
#ifdef _arch_dreamcast
    FILE *f = fopen(path, "rb");
    if (!f) return 0;
    fclose(f);
    return 1;
#else
    struct stat st;
    if (stat(path, &st) != 0) return 0;
    return S_ISREG(st.st_mode);
#endif
}

static int ends_with_rcm(const char *name) {
    size_t n = strlen(name);
    return (n >= 4 && strcmp(name + (n - 4), ".rcm") == 0);
}

static int cmp_cstr_ptr(const void *a, const void *b) {
    const char *sa = *(const char* const*)a;
    const char *sb = *(const char* const*)b;
    return strcmp(sa, sb);
}

static int list_rcm_files(const char *dir, char ***out_paths, size_t *out_count) {
    *out_paths = NULL;
    *out_count = 0;

    DIR *d = opendir(dir);
    if (!d) return 0;

    size_t cap = 64;
    char **arr = (char**)malloc(sizeof(char*) * cap);
    if (!arr) die("out of memory");

    struct dirent *ent;
    while ((ent = readdir(d)) != NULL) {
        const char *name = ent->d_name;
        if (name[0] == '.') continue;
        if (!ends_with_rcm(name)) continue;

        char *full = path_join_dup(dir, name);
        if (!is_regular_file(full)) {
            free(full);
            continue;
        }

        if (*out_count + 1 > cap) {
            cap *= 2;
            char **tmp = (char**)realloc(arr, sizeof(char*) * cap);
            if (!tmp) die("out of memory");
            arr = tmp;
        }
        arr[*out_count] = full;
        (*out_count)++;
    }
    closedir(d);

    if (*out_count == 0) {
        free(arr);
        return 0;
    }

    qsort(arr, *out_count, sizeof(char*), cmp_cstr_ptr);
    *out_paths = arr;
    return 1;
}

static void free_path_list(char **paths, size_t count) {
    if (!paths) return;
    for (size_t i = 0; i < count; i++) free(paths[i]);
    free(paths);
}

static uint8_t *read_file_aligned(const char *path, size_t *out_len, size_t alignment) {
    *out_len = 0;

    FILE *f = fopen(path, "rb");
    if (!f) return NULL;

    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return NULL; }
    long szl = ftell(f);
    if (szl <= 0) { fclose(f); return NULL; }
    if (fseek(f, 0, SEEK_SET) != 0) { fclose(f); return NULL; }

    const size_t file_bytes = (size_t)szl;

    /* aligned_alloc requires size multiple of alignment; do the same everywhere */
    size_t len = file_bytes;
    if (alignment) {
        len = (file_bytes + (alignment - 1)) & ~(alignment - 1);
    }

    void *buf = NULL;

#ifdef _arch_dreamcast
    /* KOS: memalign is the reliable aligned allocator */
    if (alignment) buf = memalign(alignment, len);
    else buf = malloc(len);
#else
#  if defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 201112L)
    if (alignment) buf = aligned_alloc(alignment, len);
    else buf = malloc(len);
#  elif defined(_POSIX_VERSION)
    if (alignment) {
        if (posix_memalign(&buf, alignment, len) != 0) buf = NULL;
    } else {
        buf = malloc(len);
    }
#  else
    (void)alignment;
    buf = malloc(len);
#  endif
#endif

    if (!buf) { fclose(f); return NULL; }

    if (fread(buf, 1, file_bytes, f) != file_bytes) {
        fclose(f);
        free(buf);
        return NULL;
    }
    fclose(f);

    /* Zero pad */
    if (len > file_bytes) {
        memset((uint8_t*)buf + file_bytes, 0, len - file_bytes);
    }

    *out_len = len;
    return (uint8_t*)buf;
}

static int rcm_load(const char *path, rcm_model_t *out) {
    memset(out, 0, sizeof(*out));

    /* Read just the header first (unaligned safe). */
    FILE *f = fopen(path, "rb");
    if (!f) return 0;

    rcm_header_t hdr;
    if (fread(&hdr, 1, sizeof(hdr), f) != sizeof(hdr)) {
        fclose(f);
        return 0;
    }

    out->fmt = rcm_format_from_magic4(hdr.magic);
    if (out->fmt == RCM_FMT_UNKNOWN) {
        fprintf(stderr, "%s: bad magic (expected RCM1/RCM2/RCM3/RCL1/RCW1, got %.4s)\n",
                path, hdr.magic);
        fclose(f);
        return 0;
    }

    /* Basic version/stride validation per format. */
    if (out->fmt == RCM_FMT_RCM1) {
        if (hdr.version != 1) {
            fprintf(stderr, "%s: unsupported RCM1 version %u\n", path, (unsigned)hdr.version);
            fclose(f);
            return 0;
        }
        if (hdr.vertex_stride != (uint16_t)sizeof(rcm_vtxf_t)) {
            fprintf(stderr, "%s: unexpected RCM1 vertex_stride %u\n", path, (unsigned)hdr.vertex_stride);
            fclose(f);
            return 0;
        }

    } else if (out->fmt == RCM_FMT_RCM3) {
        if (hdr.version != 3) {
            fprintf(stderr, "%s: unsupported RCM3 version %u\n", path, (unsigned)hdr.version);
            fclose(f);
            return 0;
        }
        if (hdr.vertex_stride != (uint16_t)sizeof(rcm_vtxf_n_t)) {
            fprintf(stderr, "%s: unexpected RCM3 vertex_stride %u\n", path, (unsigned)hdr.vertex_stride);
            fclose(f);
            return 0;
        }

    } else if (out->fmt == RCM_FMT_RCM2) {
        if (hdr.version != 2) {
            fprintf(stderr, "%s: unsupported RCM2 version %u\n", path, (unsigned)hdr.version);
            fclose(f);
            return 0;
        }
        if (hdr.vertex_stride != (uint16_t)sizeof(rcm_vtx16_t)) {
            fprintf(stderr, "%s: unexpected RCM2 vertex_stride %u\n", path, (unsigned)hdr.vertex_stride);
            fclose(f);
            return 0;
        }

    } else if (out->fmt == RCM_FMT_RCL1) {
        if (hdr.version != 1) {
            fprintf(stderr, "%s: unsupported RCL1 version %u\n", path, (unsigned)hdr.version);
            fclose(f);
            return 0;
        }
        if (hdr.vertex_stride != (uint16_t)sizeof(rcl_vtxf_cuv_t) &&
            hdr.vertex_stride != (uint16_t)sizeof(rcl_vtxf_c_t)) {
            fprintf(stderr, "%s: unexpected RCL1 vertex_stride %u\n", path, (unsigned)hdr.vertex_stride);
            fclose(f);
            return 0;
        }
    } else if (out->fmt == RCM_FMT_RCW1) {
        if (hdr.version != 1) {
            fprintf(stderr, "%s: unsupported RCW1 version %u\n",
                    path, (unsigned)hdr.version);
            fclose(f);
            return 0;
        }

        if (hdr.vertex_stride != (uint16_t)sizeof(rcm_vtxf_t)) {
            fprintf(stderr, "%s: unexpected RCW1 vertex_stride %u\n",
                    path, (unsigned)hdr.vertex_stride);
            fclose(f);
            return 0;
        }
    }

    if (hdr.index_stride != 2u) {
        fprintf(stderr, "%s: unexpected index_stride %u\n", path, (unsigned)hdr.index_stride);
        fclose(f);
        return 0;
    }

    /* Determine file size. Prefer header's file_size, but verify against actual. */
    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return 0; }
    long real_sz_l = ftell(f);
    if (real_sz_l <= 0) { fclose(f); return 0; }
    size_t real_sz = (size_t)real_sz_l;
    fclose(f);

    size_t want_sz = hdr.file_size ? (size_t)hdr.file_size : real_sz;
    if (want_sz > real_sz) {
        fprintf(stderr, "%s: file truncated (declared %zu, real %zu)\n",
                path, want_sz, real_sz);
        return 0;
    }

    /* Read whole file into one aligned blob (enables zero-copy pointers). */
    size_t blob_len = 0;
    uint8_t *blob = read_file_aligned(path, &blob_len, RCM_FILE_ALIGN);
    if (!blob) return 0;

    if (blob_len < sizeof(rcm_header_t)) { free(blob); return 0; }

    memcpy(&out->hdr, blob, sizeof(rcm_header_t));

    out->fmt = rcm_format_from_magic4(out->hdr.magic);


    uint32_t sub_off = out->hdr.submesh_off;
    uint32_t vtx_off = out->hdr.vertex_off;
    uint32_t idx_off = out->hdr.index_off;

    size_t sub_sz = (size_t)out->hdr.submesh_count * sizeof(rcm_submesh_t);
    size_t vtx_sz = (size_t)out->hdr.vertex_count  * (size_t)out->hdr.vertex_stride;
    size_t idx_sz = (size_t)out->hdr.index_count   * sizeof(uint16_t);

    size_t range = out->hdr.file_size ? (size_t)out->hdr.file_size : real_sz;

    if ((size_t)sub_off + sub_sz > range ||
        (size_t)vtx_off + vtx_sz > range ||
        (size_t)idx_off + idx_sz > range) {
        fprintf(stderr, "%s: offsets out of range\n", path);
        free(blob);
        return 0;
    }

    out->blob = blob;
    out->blob_size = blob_len;

    out->sub = (rcm_submesh_t*)(blob + sub_off);
    out->vtx = (uint8_t*)(blob + vtx_off);
    out->idx = (uint16_t*)(blob + idx_off);

    return 1;
}


static void rcm_free(rcm_model_t *m) {
    if (!m) return;

#ifdef _arch_dreamcast
    dc_free_draw_buffers(); /* frees g_dc_* globals */
#endif

    /* sub/vtx/idx all point inside blob; only blob is owned */
    free(m->blob);

    memset(m, 0, sizeof(*m));
}


static int streq_icase(const char *a, const char *b) {
    while (*a && *b) {
        unsigned char ca = (unsigned char)*a++;
        unsigned char cb = (unsigned char)*b++;
        if (tolower(ca) != tolower(cb)) return 0;
    }
    return *a == *b;
}

int has_ext(const char *path, const char *ext) {
    size_t lp = strlen(path), le = strlen(ext);
    return (lp >= le) && (strcmp(path + (lp - le), ext) == 0);
}

static const char *sprites_path_for_atlas(const char *atlas_path) {
#ifdef _arch_dreamcast
    if (atlas_path && has_ext(atlas_path, ".dt")) {
        return "textures/sprites.dt";
    }
#else
    (void)atlas_path;
#endif
    return "textures/sprites.png";
}
#ifndef _arch_dreamcast
static GLuint load_texture_png(const char *path, int flip_y, int *out_w, int *out_h) {
    int w = 0, h = 0, n = 0;

    /* stb flip is global; set/reset per call */
    stbi_set_flip_vertically_on_load(flip_y ? 1 : 0);
    stbi_uc *pixels = stbi_load(path, &w, &h, &n, 4);
    stbi_set_flip_vertically_on_load(0);

    if (!pixels) {
        fprintf(stderr, "Failed to load %s: %s\n", path, stbi_failure_reason());
        return 0;
    }

    GLuint tex = 0;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0,
                 GL_RGBA, GL_UNSIGNED_BYTE, pixels);

    stbi_image_free(pixels);

    if (out_w) *out_w = w;
    if (out_h) *out_h = h;
    return tex;
}
#endif

static GLuint load_texture_auto(const char *path, int flip_y, int *out_w, int *out_h) {
#ifdef _arch_dreamcast
    (void)flip_y;
    GLuint tex = 0;
    if (!path) return 0;
    if (!dtex_load_gl_texture(path, &tex, out_w, out_h)) return 0;
    return tex;
#else
    if (!path) return 0;
    if (has_ext(path, ".png")) return load_texture_png(path, flip_y, out_w, out_h);
    return 0;
#endif

    /* State cache */
    int last_tex_on = -1;
    GLuint last_tex = 0;
    int last_alpha = -1;
    int last_blend = -1;
    GLenum last_shade = (GLenum)-1;

}


static uint8_t argb1555_a(uint16_t c) { return (c & 0x8000u) ? 255 : 0; }
static uint8_t argb1555_r(uint16_t c) { return (uint8_t)(((c >> 10) & 0x1Fu) * 255 / 31); }
static uint8_t argb1555_g(uint16_t c) { return (uint8_t)(((c >>  5) & 0x1Fu) * 255 / 31); }
static uint8_t argb1555_b(uint16_t c) { return (uint8_t)(((c >>  0) & 0x1Fu) * 255 / 31); }

#ifdef _arch_dreamcast
/* GLdc vertex-array path is float-only for positions/normals/uvs.
   Convert packed RCM vertices once per model load. */
static GLfloat  *g_dc_pos = NULL;   /* [vcount * 3] */
static GLfloat  *g_dc_nrm = NULL;   /* [vcount * 3] */
static GLfloat  *g_dc_uv  = NULL;   /* [vcount * 2] Q15-as-float (texture matrix still works) */
static int      g_dc_has_uv = 1;
static uint8_t  *g_dc_col = NULL;   /* [vcount * 4] RGBA */
static uint8_t  *g_dc_col_base = NULL;  /* [vcount * 4] unlit RGBA (tint only) */
static uint16_t *g_dc_idx = NULL;   /* [icount] */

static uint32_t g_dc_vcount = 0;
static uint32_t g_dc_icount = 0;
static uint32_t g_dc_epoch = 1;


static void dc_free_draw_buffers(void) {
    free(g_dc_pos); g_dc_pos = NULL;
    free(g_dc_nrm); g_dc_nrm = NULL;
    free(g_dc_uv);  g_dc_uv  = NULL;
    g_dc_has_uv = 1;

    free(g_dc_col);      g_dc_col = NULL;
    free(g_dc_col_base); g_dc_col_base = NULL;

    free(g_dc_idx); g_dc_idx = NULL;

    g_dc_vcount = 0;
    g_dc_icount = 0;

    /* Invalidate lighting cache for next model */
    g_dc_epoch++;
}

static float dc_snorm8_to_float(int8_t v) {
    /* Map [-128..127] -> [-1..1]. Clamp to avoid -128 producing slightly < -1. */
    float f = (float)v / 127.0f;
    if (f < -1.0f) f = -1.0f;
    if (f >  1.0f) f =  1.0f;
    return f;
}

static int dc_build_draw_buffers(const rcm_model_t *m) {
    dc_free_draw_buffers();

    if (!m || !m->vtx || !m->idx || !m->sub) return 0;
    const uint32_t vc = m->hdr.vertex_count;
    const uint32_t ic = m->hdr.index_count;
    if (vc == 0 || ic == 0) return 0;

    const int has_uv = model_has_uv(m);
    g_dc_has_uv = has_uv ? 1 : 0;

    g_dc_pos = (GLfloat*)memalign(32, (size_t)vc * 3u * sizeof(GLfloat));
    g_dc_nrm = (GLfloat*)memalign(32, (size_t)vc * 3u * sizeof(GLfloat));
    g_dc_uv  = has_uv ? (GLfloat*)memalign(32, (size_t)vc * 2u * sizeof(GLfloat)) : NULL;

    g_dc_col_base = (uint8_t*)memalign(32, (size_t)vc * 4u);
    g_dc_col      = (uint8_t*)memalign(32, (size_t)vc * 4u);

    g_dc_idx = (uint16_t*)memalign(32, (size_t)ic * sizeof(uint16_t));

    if (!g_dc_pos || !g_dc_nrm || !g_dc_col_base || !g_dc_col || !g_dc_idx || (has_uv && !g_dc_uv)) {
        dc_free_draw_buffers();
        return 0;
    }

    const float inv127 = 1.0f / 127.0f;

    for (uint32_t i = 0; i < vc; i++) {
        float px = 0.0f, py = 0.0f, pz = 0.0f;
        int8_t nx8 = 0, ny8 = 0, nz8 = 0;
        int16_t u16 = 0, v16 = 0;
        uint32_t rgba32 = 0xFFFFFFFFu;
        int have_rgba = 0;
        if (m->fmt == RCM_FMT_RCM1 || m->fmt == RCM_FMT_RCW1) {
            const rcm_vtxf_t *v = &((const rcm_vtxf_t*)m->vtx)[i];
            px = v->x; py = v->y; pz = v->z;
            nx8 = v->nx; ny8 = v->ny; nz8 = v->nz;
            u16 = v->u; v16 = v->v;

        } else if (m->fmt == RCM_FMT_RCL1) {
            if (m->hdr.vertex_stride == (uint16_t)sizeof(rcl_vtxf_cuv_t)) {
                const rcl_vtxf_cuv_t *v = &((const rcl_vtxf_cuv_t*)m->vtx)[i];
                px = v->x; py = v->y; pz = v->z;
                nx8 = v->nx; ny8 = v->ny; nz8 = v->nz;
                u16 = v->u; v16 = v->v;
                rgba32 = v->rgba;
                have_rgba = 1;
            } else {
                const rcl_vtxf_c_t *v = &((const rcl_vtxf_c_t*)m->vtx)[i];
                px = v->x; py = v->y; pz = v->z;
                nx8 = v->nx; ny8 = v->ny; nz8 = v->nz;
                rgba32 = v->rgba;
                have_rgba = 1;
            }

        } else if (m->fmt == RCM_FMT_RCM3) {
            const rcm_vtxf_n_t *v = &((const rcm_vtxf_n_t*)m->vtx)[i];
            px = v->x; py = v->y; pz = v->z;
            nx8 = v->nx; ny8 = v->ny; nz8 = v->nz;

        } else { /* RCM2 */
            const rcm_vtx16_t *v = &((const rcm_vtx16_t*)m->vtx)[i];
            px = (float)v->x; py = (float)v->y; pz = (float)v->z;
            nx8 = v->nx; ny8 = v->ny; nz8 = v->nz;
            u16 = v->u; v16 = v->v;
        }

        g_dc_pos[i*3u + 0] = (GLfloat)px;
        g_dc_pos[i*3u + 1] = (GLfloat)py;
        g_dc_pos[i*3u + 2] = (GLfloat)pz;

        /* snorm8 -> float */
        float nx = (float)nx8 * inv127;
        float ny = (float)ny8 * inv127;
        float nz = (float)nz8 * inv127;

        /* clamp -128 edge */
        if (nx < -1.0f) nx = -1.0f;
        if (nx >  1.0f) nx =  1.0f;
        if (ny < -1.0f) ny = -1.0f;
        if (ny >  1.0f) ny =  1.0f;
        if (nz < -1.0f) nz = -1.0f;
        if (nz >  1.0f) nz =  1.0f;

        g_dc_nrm[i*3u + 0] = (GLfloat)nx;
        g_dc_nrm[i*3u + 1] = (GLfloat)ny;
        g_dc_nrm[i*3u + 2] = (GLfloat)nz;

        if (has_uv) {
            /* Keep Q15 as float; texture matrix scales by 1/UV_Q15_DIVISOR */
            g_dc_uv[i*2u + 0] = (GLfloat)u16;
            g_dc_uv[i*2u + 1] = (GLfloat)v16;
        }

        if (have_rgba) {
            /* Preserve byte order exactly as stored in-file (matches Linux glColorPointer path). */
            memcpy(&g_dc_col_base[i*4u], &rgba32, 4);
        } else {
            g_dc_col_base[i*4u + 0] = 255;
            g_dc_col_base[i*4u + 1] = 255;
            g_dc_col_base[i*4u + 2] = 255;
            g_dc_col_base[i*4u + 3] = 255;
        }
    }

    memcpy(g_dc_idx, m->idx, (size_t)ic * sizeof(uint16_t));

    if (m->fmt != RCM_FMT_RCL1) {
        for (uint32_t si = 0; si < m->hdr.submesh_count; si++) {
            const rcm_submesh_t *sm = &m->sub[si];
            const int textured = (sm->texture_id != 0xFFFFu);

            const uint8_t a = argb1555_a(sm->color_argb1555);
            const uint8_t r = argb1555_r(sm->color_argb1555);
            const uint8_t g = argb1555_g(sm->color_argb1555);
            const uint8_t b = argb1555_b(sm->color_argb1555);
            const uint8_t out_a = textured ? 255 : a;

            const uint32_t first = sm->first_index;
            const uint32_t count = sm->index_count;
            if (first >= ic) continue;
            const uint32_t end = (first + count > ic) ? ic : (first + count);

            for (uint32_t ii = first; ii < end; ii++) {
                const uint16_t vi = g_dc_idx[ii];
                if (vi >= vc) continue;
                g_dc_col_base[vi*4u + 0] = r;
                g_dc_col_base[vi*4u + 1] = g;
                g_dc_col_base[vi*4u + 2] = b;
                g_dc_col_base[vi*4u + 3] = out_a;
            }
        }

    }

    memcpy(g_dc_col, g_dc_col_base, (size_t)vc * 4u);

    g_dc_vcount = vc;
    g_dc_icount = ic;

    /* Invalidate lighting cache for this new model */
    g_dc_epoch++;

    return 1;
}
static void dc_update_lighting_colors(int enable_lighting_flag) {
    if (!g_dc_col || !g_dc_col_base || !g_dc_nrm || g_dc_vcount == 0) return;

    static int   last_enabled = -1;
    static float last_yaw = 1e30f, last_pitch = 1e30f;
    static uint32_t last_epoch = 0;

    /* New model loaded? force recompute */
    if (last_epoch != g_dc_epoch) {
        last_epoch = g_dc_epoch;
        last_enabled = -1;
        last_yaw = 1e30f;
        last_pitch = 1e30f;
    }

    if (!enable_lighting_flag) {
        if (last_enabled != 0) {
            memcpy(g_dc_col, g_dc_col_base, (size_t)g_dc_vcount * 4u);
            last_enabled = 0;
        }
        return;
    }

    if (last_enabled == 1 &&
        fabsf(g_light_yaw_deg - last_yaw) < 0.0001f &&
        fabsf(g_light_pitch_deg - last_pitch) < 0.0001f) {
        return;
    }

    last_enabled = 1;
    last_yaw = g_light_yaw_deg;
    last_pitch = g_light_pitch_deg;

    const float amb = 0.35f;
    const float dif = 0.85f;

    const float yaw   = g_light_yaw_deg   * (float)M_PI / 180.0f;
    const float pitch = g_light_pitch_deg * (float)M_PI / 180.0f;

    /* Unit-length already */
    const float cx = cosf(pitch) * cosf(yaw);
    const float cy = sinf(pitch);
    const float cz = cosf(pitch) * sinf(yaw);

    /* Directional vector (and fold your Y/Z render flips into the light once) */
    float lx = -cx;
    float ly = -cy;
    float lz = -cz;
    ly = -ly;
    lz = -lz;

    for (uint32_t i = 0; i < g_dc_vcount; ++i) {
        const float nx = g_dc_nrm[i*3u + 0];
        const float ny = g_dc_nrm[i*3u + 1];
        const float nz = g_dc_nrm[i*3u + 2];

        float ndotl = nx*lx + ny*ly + nz*lz;
        if (ndotl < 0.0f) ndotl = 0.0f;

        float I = amb + dif * ndotl;
        if (I > 1.0f) I = 1.0f;

        /* fixed-point scale for faster per-channel multiply */
        const uint32_t scale = (uint32_t)(I * 256.0f + 0.5f); /* 0..256 */

        const uint8_t br = g_dc_col_base[i*4u + 0];
        const uint8_t bg = g_dc_col_base[i*4u + 1];
        const uint8_t bb = g_dc_col_base[i*4u + 2];
        const uint8_t ba = g_dc_col_base[i*4u + 3];

        g_dc_col[i*4u + 0] = (uint8_t)((br * scale + 128u) >> 8);
        g_dc_col[i*4u + 1] = (uint8_t)((bg * scale + 128u) >> 8);
        g_dc_col[i*4u + 2] = (uint8_t)((bb * scale + 128u) >> 8);
        g_dc_col[i*4u + 3] = ba;
    }
}
#endif


static void set_perspective(float fovy_deg, float aspect, float znear, float zfar) {
    const float PI = 3.14159265358979323846f;
    float fovy_rad = fovy_deg * (PI / 180.0f);
    float f = 1.0f / tanf(fovy_rad * 0.5f);
    float top = znear / f;
    float bottom = -top;
    float right = top * aspect;
    float left = -right;

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glFrustum(left, right, bottom, top, znear, zfar);
    glMatrixMode(GL_MODELVIEW);
}

static float aabb_radius_rcm_units(const rcm_header_t *h) {
    float minx = (float)h->aabb_min[0] / VERTEX_SCALE;
    float miny = (float)h->aabb_min[1] / VERTEX_SCALE;
    float minz = (float)h->aabb_min[2] / VERTEX_SCALE;
    float maxx = (float)h->aabb_max[0] / VERTEX_SCALE;
    float maxy = (float)h->aabb_max[1] / VERTEX_SCALE;
    float maxz = (float)h->aabb_max[2] / VERTEX_SCALE;
    float dx = maxx - minx, dy = maxy - miny, dz = maxz - minz;
    return 0.5f * sqrtf(dx*dx + dy*dy + dz*dz);
}

static const atlas_pos_t *atlas_pos_for_texture_id(uint16_t texture_id) {
    if (texture_id < g_model_atlas_pos_count) {
        return &g_model_atlas_pos[texture_id];
    }

    /* Special cases copied from rsc-c; keep for compatibility and to avoid unused warnings */
    if (texture_id == 0xFFFEu) return &g_white_model_atlas_pos;
    if (texture_id == 0xFFFDu) return &g_transparent_model_atlas_pos;

    return NULL;
}

static void apply_q15_texcoord_transform(uint16_t texture_id, int flip_v_local) {
    glMatrixMode(GL_TEXTURE);
    glLoadIdentity();

    const float s = 1.0f / UV_Q15_DIVISOR;

    if (!flip_v_local) {
        glScalef(s, s, 1.0f);
    } else {
        float c = 1.0f;
        const atlas_pos_t *p = atlas_pos_for_texture_id(texture_id);
        if (p) c = p->v0 + p->v1;

        glTranslatef(0.0f, c, 0.0f);
        glScalef(s, -s, 1.0f);
    }

    glMatrixMode(GL_MODELVIEW);
}


static void reset_tex_matrix(void) {
    glMatrixMode(GL_TEXTURE);
    glLoadIdentity();
    glMatrixMode(GL_MODELVIEW);
}

static void setup_lighting(const rcm_model_t *m) {
    glEnable(GL_LIGHTING);
    glEnable(GL_LIGHT0);

    /* Convert yaw/pitch -> directional vector (world space) */
    const float yaw   = g_light_yaw_deg   * (float)M_PI / 180.0f;
    const float pitch = g_light_pitch_deg * (float)M_PI / 180.0f;

    const float cx = cosf(pitch) * cosf(yaw);
    const float cy = sinf(pitch);
    const float cz = cosf(pitch) * sinf(yaw);

    /* Directional light: w = 0 */
    GLfloat dir[4] = { -cx, -cy, -cz, 0.0f };

    const GLfloat amb[4] = { 0.35f, 0.35f, 0.35f, 1.0f };
    const GLfloat dif[4] = { 0.85f, 0.85f, 0.85f, 1.0f };

    glLightfv(GL_LIGHT0, GL_AMBIENT, amb);
    glLightfv(GL_LIGHT0, GL_DIFFUSE, dif);

    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    /* Match the same axis/sign convention used when drawing the model */
    {
        const float inv = (m && m->fmt == RCM_FMT_RCM2) ? VERTEX_SCALE : 1.0f;
        glScalef(inv, -inv, -inv);
    }
    glLightfv(GL_LIGHT0, GL_POSITION, dir);
    glPopMatrix();

    glEnable(GL_COLOR_MATERIAL);
    glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);

    glEnable(GL_NORMALIZE);
}


static void disable_lighting(void) {
    glDisable(GL_LIGHTING);
    glDisable(GL_LIGHT0);
    glDisable(GL_NORMALIZE);
    glDisable(GL_COLOR_MATERIAL);
}

static int file_exists(const char *p) {
    FILE *f = fopen(p, "rb");
    if (!f) return 0;
    fclose(f);
    return 1;
}

static const char *pick_default_atlas(void) {
    static const char *candidates[] = {
        "textures/model_textures.png",
        "cache/textures/model_textures.png",
        "./cache/textures/model_textures.png",
        NULL
    };
    for (int i = 0; candidates[i]; i++) {
        if (file_exists(candidates[i])) return candidates[i];
    }
    return NULL;
}

static void draw_model(const rcm_model_t *m, GLuint atlas_tex,
                       int enable_blend, int enable_lighting_flag, int wireframe) {
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_CULL_FACE);
    glCullFace(GL_BACK);
    glFrontFace(GL_CCW);

#ifndef _arch_dreamcast
    glPolygonMode(GL_FRONT_AND_BACK, wireframe ? GL_LINE : GL_FILL);
#else
    (void)wireframe;
#endif

    if (enable_lighting_flag) setup_lighting(m);
    else                      disable_lighting();

    glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE);

    int has_uv = 0; /* per-model UV presence (also gates texturing) */


#ifdef _arch_dreamcast
    /* GLdc fixed-function lighting is unreliable; do lighting in the color array instead. */
    glDisable(GL_LIGHTING);
    glDisable(GL_LIGHT0);
    glDisable(GL_COLOR_MATERIAL);
    glDisable(GL_NORMALIZE);

    dc_update_lighting_colors(enable_lighting_flag);

    has_uv = g_dc_has_uv;


    if (!g_dc_pos || !g_dc_col || !g_dc_idx) {
        return;
    }

    glEnableClientState(GL_VERTEX_ARRAY);
    glVertexPointer(3, GL_FLOAT, 0, (const void*)g_dc_pos);

    if (g_dc_has_uv && g_dc_uv) {
        glEnableClientState(GL_TEXTURE_COORD_ARRAY);
        glTexCoordPointer(2, GL_FLOAT, 0, (const void*)g_dc_uv);
    } else {
        glDisableClientState(GL_TEXTURE_COORD_ARRAY);
    }

    glEnableClientState(GL_COLOR_ARRAY);
    glColorPointer(4, GL_UNSIGNED_BYTE, 0, (const void*)g_dc_col);

#else
    has_uv = model_has_uv(m);
    const GLsizei stride = (GLsizei)m->hdr.vertex_stride;

    glEnableClientState(GL_VERTEX_ARRAY);
    glEnableClientState(GL_NORMAL_ARRAY);

    if (m->fmt == RCM_FMT_RCL1) {
        glEnableClientState(GL_COLOR_ARRAY);
        glColorPointer(4, GL_UNSIGNED_BYTE, stride,
                       (const void *)(m->vtx + offsetof(rcl_vtxf_cuv_t, rgba)));

        glVertexPointer(3, GL_FLOAT, stride,
                        (const void *)(m->vtx + offsetof(rcl_vtxf_cuv_t, x)));
        glNormalPointer(GL_BYTE, stride,
                        (const void *)(m->vtx + offsetof(rcl_vtxf_cuv_t, nx)));
    } else if (m->fmt == RCM_FMT_RCM1 || m->fmt == RCM_FMT_RCW1) {
        glDisableClientState(GL_COLOR_ARRAY);
        glVertexPointer(3, GL_FLOAT, stride,
                        (const void *)(m->vtx + offsetof(rcm_vtxf_t, x)));
        glNormalPointer(GL_BYTE, stride,
                        (const void *)(m->vtx + offsetof(rcm_vtxf_t, nx)));
    } else if (m->fmt == RCM_FMT_RCM3) {
        glDisableClientState(GL_COLOR_ARRAY);
        glVertexPointer(3, GL_FLOAT, stride,
                        (const void *)(m->vtx + offsetof(rcm_vtxf_n_t, x)));
        glNormalPointer(GL_BYTE, stride,
                        (const void *)(m->vtx + offsetof(rcm_vtxf_n_t, nx)));
    } else { /* RCM2 */
        glDisableClientState(GL_COLOR_ARRAY);
        glVertexPointer(3, GL_SHORT, stride,
                        (const void *)(m->vtx + offsetof(rcm_vtx16_t, x)));
        glNormalPointer(GL_BYTE, stride,
                        (const void *)(m->vtx + offsetof(rcm_vtx16_t, nx)));
    }

    if (has_uv) {
        glEnableClientState(GL_TEXTURE_COORD_ARRAY);
        if (m->fmt == RCM_FMT_RCL1) {
            glTexCoordPointer(2, GL_SHORT, stride,
                              (const void *)(m->vtx + offsetof(rcl_vtxf_cuv_t, u)));
        } else if (m->fmt == RCM_FMT_RCM1 || m->fmt == RCM_FMT_RCW1) {
            glTexCoordPointer(2, GL_SHORT, stride,
                              (const void *)(m->vtx + offsetof(rcm_vtxf_t, u)));
        } else { /* RCM2 */
            glTexCoordPointer(2, GL_SHORT, stride,
                              (const void *)(m->vtx + offsetof(rcm_vtx16_t, u)));
        }
    }
    else {
        glDisableClientState(GL_TEXTURE_COORD_ARRAY);
    }
#endif

    /* State cache */
    int last_tex_on = -1;
    GLuint last_tex = 0;
    int last_alpha = -1;
    int last_blend = -1;
    GLenum last_shade = (GLenum)-1;


    for (uint32_t si = 0; si < m->hdr.submesh_count; si++) {
        const rcm_submesh_t *sm = &m->sub[si];
        const int textured = (sm->texture_id != 0xFFFFu) && (atlas_tex != 0) && has_uv;
        const GLenum shade = (sm->flags & RCM_SM_GOURAUD) ? GL_SMOOTH : GL_FLAT;
        if (shade != last_shade) { glShadeModel(shade); last_shade = shade; }

#ifndef _arch_dreamcast
        /* For RCL1, vertex color drives tint; don't overwrite it per submesh. */
        if (m->fmt != RCM_FMT_RCL1) {
            uint8_t a = argb1555_a(sm->color_argb1555);
            uint8_t r = argb1555_r(sm->color_argb1555);
            uint8_t g = argb1555_g(sm->color_argb1555);
            uint8_t b = argb1555_b(sm->color_argb1555);
            if (textured) glColor4ub(r, g, b, 255);
            else          glColor4ub(r, g, b, a);
        }

        if (textured && has_uv) {
            glEnableClientState(GL_TEXTURE_COORD_ARRAY);

            if (m->fmt == RCM_FMT_RCL1) {
                glTexCoordPointer(2, GL_SHORT, (GLsizei)m->hdr.vertex_stride,
                                  (const void *)(m->vtx + offsetof(rcl_vtxf_cuv_t, u)));
            } else if (m->fmt == RCM_FMT_RCM1 || m->fmt == RCM_FMT_RCW1) {
                glTexCoordPointer(2, GL_SHORT, (GLsizei)sizeof(rcm_vtxf_t),
                                  (const void *)&((const rcm_vtxf_t *)m->vtx)[0].u);
            } else { /* RCM2 */
                glTexCoordPointer(2, GL_SHORT, (GLsizei)sizeof(rcm_vtx16_t),
                                  (const void *)&((const rcm_vtx16_t *)m->vtx)[0].u);
            }
        } else {
            glDisableClientState(GL_TEXTURE_COORD_ARRAY);
        }
#endif

        /* Texture state */
        if (textured) {
            if (last_tex_on != 1) { glEnable(GL_TEXTURE_2D); last_tex_on = 1; }
            if (last_tex != atlas_tex) { glBindTexture(GL_TEXTURE_2D, atlas_tex); last_tex = atlas_tex; }

            apply_q15_texcoord_transform(sm->texture_id, g_flip_v_local);

            /* Keep behavior identical to current viewer: alpha test on for textured. */
            if (last_alpha != 1) { glEnable(GL_ALPHA_TEST); last_alpha = 1; }
            glAlphaFunc(GL_GREATER, 0.5f);
        } else {
            if (last_tex_on != 0) { glDisable(GL_TEXTURE_2D); last_tex_on = 0; }
#ifndef _arch_dreamcast
            glDisableClientState(GL_TEXTURE_COORD_ARRAY);
#endif
            reset_tex_matrix();
            if (last_alpha != 0) { glDisable(GL_ALPHA_TEST); last_alpha = 0; }
        }

        /* Blend state */
        {
            const int want_blend = (enable_blend && textured && (sm->flags & RCM_SM_ALPHA)) ? 1 : 0;
            if (want_blend != last_blend) {
                if (want_blend) {
                    glEnable(GL_BLEND);
                    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
                } else {
                    glDisable(GL_BLEND);
                }
                last_blend = want_blend;
            }
        }

#ifdef _arch_dreamcast
        const uint16_t *idx = g_dc_idx + sm->first_index;
#else
        const uint16_t *idx = m->idx + sm->first_index;
#endif
        glDrawElements(GL_TRIANGLES, (GLsizei)sm->index_count, GL_UNSIGNED_SHORT, idx);
    }

    reset_tex_matrix();

    glDisable(GL_ALPHA_TEST);
    glDisable(GL_BLEND);

#ifdef _arch_dreamcast
    glDisableClientState(GL_COLOR_ARRAY);
    glDisableClientState(GL_TEXTURE_COORD_ARRAY);
    glDisableClientState(GL_VERTEX_ARRAY);
#else
    glDisableClientState(GL_TEXTURE_COORD_ARRAY);
    glDisableClientState(GL_NORMAL_ARRAY);
    glDisableClientState(GL_VERTEX_ARRAY);

    /* Important cleanup: don’t leave UI in wireframe */
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    glDisableClientState(GL_COLOR_ARRAY);
#endif
}

static void update_window_title(SDL_Window *win, const char *path, size_t idx, size_t total) {
    char title[512];
    const char *base = path_basename(path);
    if (total > 1) {
        snprintf(title, sizeof(title), "RCM Viewer (GL1.1) - %s [%zu/%zu]", base, idx + 1, total);
    } else {
        snprintf(title, sizeof(title), "RCM Viewer (GL1.1) - %s", base);
    }
    SDL_SetWindowTitle(win, title);
}

static int load_model_by_index(rcm_model_t *model, const char *path,
                               float *out_dist, SDL_Window *win,
                               size_t idx, size_t total) {
    rcm_free(model);

    if (!rcm_load(path, model)) {
        fprintf(stderr, "Failed to load %s\n", path);
        return 0;
    }

#ifdef _arch_dreamcast
    if (!dc_build_draw_buffers(model)) {
        fprintf(stderr, "Failed to build Dreamcast draw buffers for %s\n", path);
        rcm_free(model);
        return 0;
    }
#endif

    {
        float radius = aabb_radius_rcm_units(&model->hdr);
        float d = fmaxf(1.5f, radius * 2.5f);
        if (out_dist) *out_dist = d;
    }

    update_window_title(win, path, idx, total);

    printf("Loaded %s\n", path);
    printf("  format=%s vertices=%lu indices=%lu submeshes=%lu\n",
           rcm_format_name(model->fmt),
           (unsigned long)model->hdr.vertex_count,
           (unsigned long)model->hdr.index_count,
           (unsigned long)model->hdr.submesh_count);

    return 1;
}


static void usage(const char *argv0) {
    fprintf(stderr,
        "Usage:\n"
        "  %s <model.rcm|directory> [--atlas PNG|<path>]\n\n"
        "Modes:\n"
        "  --atlas PNG    : use textures/model_textures.png + textures/sprites.png\n"
        "  --atlas <path> : load that file for the model atlas\n"
        "                  (PC: .png only; sprites uses textures/sprites.png)\n"
#ifdef _arch_dreamcast
        "                  (Dreamcast: .dt supported; sprites follows .dt/.png)\n"
#endif
        "\n"
        "Examples:\n"
        "  %s out_rcm\n"
        "  %s out_rcm --atlas PNG\n"
        "  %s out_rcm --atlas textures/model_textures.png\n",
        argv0, argv0, argv0, argv0);
#ifdef _arch_dreamcast
    fprintf(stderr,
        "  %s out_rcm --atlas DTEX\n"
        "  %s out_rcm --atlas textures/model_textures.dt\n",
        argv0, argv0);
#endif
}

int main(int argc, char **argv) {
#ifndef _arch_dreamcast
    if (argc < 2) { usage(argv[0]); return 1; }
#endif

    SDL_SetMainReady(); /* Safe/standard when building with SDL_MAIN_HANDLED */

    const char *input_path = NULL;
    const char *atlas_path = NULL;
    const char *sprites_path = NULL;

#ifdef _arch_dreamcast
    /* -------- Dreamcast path resolution --------
       We may be running from:
         - /cd  (disc / CDI)
         - /pc  (dcload + dc-tool -c host dir)
       So pick a root that actually contains our data folders. */

    const char *root = NULL;
    {
        const char *candidates[] = { "/cd", "/pc" };
        for (size_t i = 0; i < sizeof(candidates)/sizeof(candidates[0]); ++i) {
            const char *r = candidates[i];

            char test_out[256];
            char test_tex[256];
            snprintf(test_out, sizeof(test_out), "%s/out_rcm", r);
            snprintf(test_tex, sizeof(test_tex), "%s/textures", r);

            if (is_dir_path(test_out) || is_dir_path(test_tex)) {
                root = r;
                break;
            }
        }
        if (!root) root = "/cd";
    }

    /* Try to make relative filesystem usage sane */
    (void)chdir(root);

    /* Input path: accept argv[1] if present, else default "out_rcm".
       If argv[1] is relative, resolve under the chosen root. */
    {
        const char *arg_input =
            (argc >= 2 && argv && argv[1] && argv[1][0]) ? argv[1] : "out_rcm";

        static char input_buf[512];

        if (arg_input[0] == '/') {
            snprintf(input_buf, sizeof(input_buf), "%s", arg_input);
        } else {
            snprintf(input_buf, sizeof(input_buf), "%s/%s", root, arg_input);
        }
        input_path = input_buf;
    }

    {
        static char atlas_buf[512];
        static char sprites_buf[512];

        char cand_a[512];
        char cand_s[512];

        snprintf(cand_a, sizeof(cand_a), "%s/textures/model_textures.dt", root);
        snprintf(cand_s, sizeof(cand_s), "%s/textures/sprites.dt", root);

        if (is_regular_file(cand_a)) {
            snprintf(atlas_buf, sizeof(atlas_buf), "%s", cand_a);
        } else {
            snprintf(atlas_buf, sizeof(atlas_buf), "%s/model_textures.dt", root);
        }

        if (is_regular_file(cand_s)) {
            snprintf(sprites_buf, sizeof(sprites_buf), "%s", cand_s);
        } else {
            snprintf(sprites_buf, sizeof(sprites_buf), "%s/sprites.dt", root);
        }

        atlas_path   = atlas_buf;
        sprites_path = sprites_buf;
    }

    printf("DC data root : %s\n", root);
    printf("DC input     : %s\n", input_path);
    printf("DC atlas     : %s\n", atlas_path);
    printf("DC sprites   : %s\n", sprites_path);

#else
    /* -------- PC argument parsing -------- */
    input_path = argv[1];

    for (int i = 2; i < argc; ++i) {
        if (strcmp(argv[i], "--atlas") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "--atlas requires an argument\n");
                usage(argv[0]);
                return 1;
            }

            const char *arg = argv[++i];

            if (streq_icase(arg, "PNG")) {
                atlas_path   = "textures/model_textures.png";
                sprites_path = "textures/sprites.png";
                continue;
            }

            /* PC: DTEX is not supported (keyword or .dt path). */
            if (streq_icase(arg, "DTEX") || has_ext(arg, ".dt")) {
                fprintf(stderr,
                        "%s: DTEX is Dreamcast-only. Use --atlas PNG or a .png atlas path on PC.\n",
                        arg);
                return 1;
            }

            /* Explicit atlas file path */
            atlas_path   = arg;
            sprites_path = sprites_path_for_atlas(atlas_path);
            continue;
        }

        fprintf(stderr, "Unknown arg: %s\n", argv[i]);
        usage(argv[0]);
        return 1;
    }

    /* Defaults (PC) */
    if (!atlas_path) atlas_path = pick_default_atlas(); /* PNG default */
    if (!sprites_path) sprites_path = sprites_path_for_atlas(atlas_path);
#endif

    /* -------- Build model list -------- */
    char **model_paths = NULL;
    size_t model_count = 0;
    size_t current_idx = 0;

    if (is_dir_path(input_path)) {
        if (!list_rcm_files(input_path, &model_paths, &model_count)) {
            fprintf(stderr, "No .rcm files found in directory: %s\n", input_path);
            return 1;
        }
        current_idx = 0;
    } else {
        char *dir = path_dirname_dup(input_path);
        if (list_rcm_files(dir, &model_paths, &model_count)) {
            const char *want = path_basename(input_path);
            for (size_t i = 0; i < model_count; i++) {
                if (strcmp(path_basename(model_paths[i]), want) == 0) {
                    current_idx = i;
                    break;
                }
            }
        } else {
            model_paths = (char**)malloc(sizeof(char*));
            if (!model_paths) die("out of memory");
            model_paths[0] = strdup_safe(input_path);
            model_count = 1;
            current_idx = 0;
        }
        free(dir);
    }

    if (SDL_Init(SDL_INIT_VIDEO) != 0) die(SDL_GetError());

#ifdef _arch_dreamcast
    SDL_Window *win = SDL_CreateWindow("RCM Viewer (Dreamcast)", 0, 0, 640, 480, 0);
    if (!win) die(SDL_GetError());

    glKosInit();
#else
    SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
    SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 24);

    SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_COMPATIBILITY);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 2);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 1);

    SDL_Window *win = SDL_CreateWindow("RCM Viewer (GL path)",
                                       SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
                                       900, 700,
                                       SDL_WINDOW_OPENGL | SDL_WINDOW_RESIZABLE);
    if (!win) die(SDL_GetError());

    SDL_GLContext ctx = SDL_GL_CreateContext(win);
    if (!ctx) die(SDL_GetError());

    SDL_GL_SetSwapInterval(1);
#endif

    /* -------- Texture loads -------- */
    GLuint atlas_tex = 0;
    int atlas_w = 0, atlas_h = 0;
    if (atlas_path) {
        atlas_tex = load_texture_auto(atlas_path, 0, &atlas_w, &atlas_h);
        if (atlas_tex) printf("Loaded atlas: %s (%dx%d)\n", atlas_path, atlas_w, atlas_h);
        else printf("WARNING: could not load atlas: %s\n", atlas_path);
    }

    GLuint sprites_tex = 0;
    int sprites_w = 0, sprites_h = 0;
    sprites_tex = load_texture_auto(sprites_path, 0, &sprites_w, &sprites_h);
    if (sprites_tex) {
        printf("Loaded sprites font atlas: %s (%dx%d)\n", sprites_path, sprites_w, sprites_h);
    } else {
        printf("WARNING: could not load %s (font overlay disabled)\n", sprites_path);
    }

    /* -------- Model + camera -------- */
    rcm_model_t model;
    memset(&model, 0, sizeof(model));

    float yaw = 35.0f, pitch = 20.0f;
    float dist = 5.0f;
    float dist_default = dist;

    if (!load_model_by_index(&model, model_paths[current_idx], &dist, win, current_idx, model_count)) {
        fprintf(stderr, "Failed to load initial model.\n");
        if (sprites_tex) glDeleteTextures(1, &sprites_tex);
        if (atlas_tex) glDeleteTextures(1, &atlas_tex);
#ifdef _arch_dreamcast
        /* no SDL GL context to delete */
#else
        SDL_GL_DeleteContext(ctx);
#endif
        SDL_DestroyWindow(win);
        SDL_Quit();
        free_path_list(model_paths, model_count);
        return 1;
    }
    dist_default = dist;

    viewer_controls ctl = {0};
    viewer_fps_counter fpsc;
    viewer_fps_init(&fpsc);

    int wireframe = 0;
    int enable_light = 0;
    int enable_blend = 0;

    int running = 1;
    while (running) {
        viewer_fps_tick(&fpsc);

        SDL_Event e;
        while (SDL_PollEvent(&e)) {
            viewer_handle_event(&e, &running, &ctl,
                                &yaw, &pitch, &dist, &dist_default,
                                &wireframe, &enable_light, &enable_blend,
                                &current_idx, model_count, model_paths,
                                &model, win);
        }

#ifdef _arch_dreamcast
        viewer_handle_dc_controller(&running,
                                   &yaw, &pitch,
                                   &dist, &dist_default,
                                   &enable_light,
                                   &current_idx, model_count, model_paths,
                                   &model, win);
#endif


#ifdef _arch_dreamcast
        int w = 640, h = 480;
#else
        int w = 0, h = 0;
        SDL_GetWindowSize(win, &w, &h);
#endif
        glViewport(0, 0, w, h);

        glClearColor(0.08f, 0.08f, 0.10f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        set_perspective(60.0f, (h > 0) ? (float)w / (float)h : 1.0f, 0.05f, 500.0f);

        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();

        glTranslatef(0.0f, 0.0f, -dist);
        glRotatef(pitch, 1.0f, 0.0f, 0.0f);
        glRotatef(yaw,   0.0f, 1.0f, 0.0f);

        {
            const float s = rcm_draw_scale_to_world(&model);
            glScalef(s, -s, -s);
        }

        {
            const float cs = rcm_center_units_scale(&model);
            float cx = ((float)model.hdr.aabb_min[0] + (float)model.hdr.aabb_max[0]) * 0.5f * cs;
            float cy = ((float)model.hdr.aabb_min[1] + (float)model.hdr.aabb_max[1]) * 0.5f * cs;
            float cz = ((float)model.hdr.aabb_min[2] + (float)model.hdr.aabb_max[2]) * 0.5f * cs;
            glTranslatef(-cx, -cy, -cz);
        }

        draw_model(&model, atlas_tex, enable_blend, enable_light, wireframe);

        if (sprites_tex) {
            font_begin(sprites_tex, w, h);

            const size_t cur   = current_idx;
            const size_t total = model_count;
            const char *cur_path = (model_paths && cur < total) ? model_paths[cur] : NULL;

            viewer_draw_model_overlay(sprites_tex, sprites_w, sprites_h, w, h,
                                      cur_path, cur, total, &model);

            viewer_draw_fps_top_right_left(sprites_tex, sprites_w, sprites_h, w, h, &fpsc);
            viewer_draw_controls_hint_bottom_center(sprites_tex, sprites_w, sprites_h, w, h);
            font_end();
        }

#ifdef _arch_dreamcast
        glKosSwapBuffers();
#else
        SDL_GL_SwapWindow(win);
#endif
    }

    if (sprites_tex) glDeleteTextures(1, &sprites_tex);
    if (atlas_tex) glDeleteTextures(1, &atlas_tex);
    rcm_free(&model);
    free_path_list(model_paths, model_count);

#ifdef _arch_dreamcast
    /* no SDL GL context to delete */
#else
    SDL_GL_DeleteContext(ctx);
#endif
    SDL_DestroyWindow(win);
    SDL_Quit();
    return 0;
}
