// rcm_font.c - Font Overlay/UI Handling, based on RSC-C's atlas of jagex.jag
// Meant for rcm_viewer, but modularized enough to port to anything.
#include "rcm_font.h"
#include "rcm_viewer.h"   /* temporary: for atlas_pos_t (we’ll decouple later) */
#include <math.h>         /* floorf */

/* --- FONT_BOLD_12 atlas rects from rsc-c fonts.c (index 1 in gl_font_atlas_positions) --- */
const atlas_pos_t g_font_bold12_atlas_pos[95] = {
        {0.546875f, 0.553711f, 0.875977f, 0.884766f},
        {0.567383f, 0.574219f, 0.802734f, 0.811523f},
        {0.558594f, 0.565430f, 0.811523f, 0.820313f},
        {0.542969f, 0.549805f, 0.842773f, 0.851563f},
        {0.528320f, 0.534180f, 0.960938f, 0.969727f},
        {0.351563f, 0.356445f, 0.906250f, 0.915039f},
        {0.540039f, 0.546875f, 0.875977f, 0.884766f},
        {0.537109f, 0.543945f, 0.915039f, 0.923828f},
        {0.849609f, 0.851563f, 0.802734f, 0.811523f},
        {0.516602f, 0.522461f, 0.989258f, 0.998047f},
        {0.420898f, 0.427734f, 0.886719f, 0.895508f},
        {0.497070f, 0.502930f, 0.991211f, 1.000000f},
        {0.536133f, 0.544922f, 0.945313f, 0.954102f},
        {0.364258f, 0.371094f, 0.886719f, 0.895508f},
        {0.537109f, 0.544922f, 0.989258f, 0.998047f},
        {0.545898f, 0.552734f, 0.792969f, 0.801758f},
        {0.532227f, 0.540039f, 0.812500f, 0.822266f},
        {0.516602f, 0.524414f, 0.945313f, 0.954102f},
        {0.379883f, 0.386719f, 0.906250f, 0.915039f},
        {0.427734f, 0.433594f, 0.886719f, 0.895508f},
        {0.514648f, 0.521484f, 0.784180f, 0.792969f},
        {0.472656f, 0.480469f, 0.991211f, 1.000000f},
        {0.245117f, 0.256836f, 0.892578f, 0.901367f},
        {0.552734f, 0.559570f, 0.960938f, 0.969727f},
        {0.540039f, 0.546875f, 0.929688f, 0.938477f},
        {0.544922f, 0.550781f, 0.989258f, 0.998047f},
        {0.554688f, 0.560547f, 0.908203f, 0.915039f},
        {0.534180f, 0.540039f, 0.765625f, 0.774414f},
        {0.479492f, 0.485352f, 0.822266f, 0.829102f},
        {0.523438f, 0.529297f, 0.792969f, 0.801758f},
        {0.346680f, 0.352539f, 0.993164f, 1.000000f},
        {0.519531f, 0.523438f, 0.915039f, 0.923828f},
        {0.510742f, 0.516602f, 0.971680f, 0.980469f},
        {0.545898f, 0.551758f, 0.884766f, 0.893555f},
        {0.260742f, 0.262695f, 0.855469f, 0.864258f},
        {0.791992f, 0.793945f, 0.666016f, 0.676758f},
        {0.333008f, 0.338867f, 0.746094f, 0.754883f},
        {0.849609f, 0.851563f, 0.802734f, 0.811523f},
        {0.944336f, 0.954102f, 0.770508f, 0.777344f},
        {0.300781f, 0.306641f, 0.434570f, 0.441406f},
        {0.447266f, 0.453125f, 0.852539f, 0.859375f},
        {0.371094f, 0.376953f, 0.886719f, 0.895508f},
        {0.528320f, 0.534180f, 0.895508f, 0.904297f},
        {0.027344f, 0.031250f, 0.778320f, 0.785156f},
        {0.522461f, 0.528320f, 0.954102f, 0.960938f},
        {0.357422f, 0.360352f, 0.819336f, 0.828125f},
        {0.284180f, 0.290039f, 0.434570f, 0.441406f},
        {0.441406f, 0.447266f, 0.852539f, 0.859375f},
        {0.745117f, 0.753906f, 0.165039f, 0.171875f},
        {0.428711f, 0.434570f, 0.981445f, 0.988281f},
        {0.558594f, 0.564453f, 0.929688f, 0.938477f},
        {0.290039f, 0.294922f, 0.434570f, 0.441406f},
        {0.532227f, 0.538086f, 0.885742f, 0.894531f},
        {0.553711f, 0.557617f, 0.980469f, 0.989258f},
        {0.540039f, 0.545898f, 0.792969f, 0.801758f},
        {0.529297f, 0.535156f, 0.792969f, 0.801758f},
        {0.555664f, 0.561523f, 0.829102f, 0.837891f},
        {0.552734f, 0.558594f, 0.765625f, 0.774414f},
        {0.558594f, 0.564453f, 0.792969f, 0.801758f},
        {0.357422f, 0.363281f, 0.695313f, 0.704102f},
        {0.546875f, 0.552734f, 0.784180f, 0.792969f},
        {0.398438f, 0.404297f, 0.906250f, 0.915039f},
        {0.421875f, 0.423828f, 0.731445f, 0.740234f},
        {0.971680f, 0.976563f, 0.322266f, 0.325195f},
        {0.418945f, 0.425781f, 0.906250f, 0.915039f},
        {0.370117f, 0.375000f, 0.895508f, 0.906250f},
        {0.532227f, 0.540039f, 0.774414f, 0.783203f},
        {0.895508f, 0.901367f, 0.848633f, 0.853516f},
        {0.545898f, 0.553711f, 0.980469f, 0.989258f},
        {0.319336f, 0.322266f, 0.615234f, 0.619141f},
        {0.796875f, 0.799805f, 0.415039f, 0.425781f},
        {0.799805f, 0.802734f, 0.415039f, 0.425781f},
        {0.860352f, 0.863281f, 0.225586f, 0.226563f},
        {0.559570f, 0.566406f, 0.466797f, 0.467773f},
        {0.608398f, 0.614258f, 0.717773f, 0.720703f},
        {0.838867f, 0.844727f, 0.438477f, 0.443359f},
        {0.717773f, 0.720703f, 0.413086f, 0.423828f},
        {0.414063f, 0.418945f, 0.945313f, 0.956055f},
        {0.905273f, 0.908203f, 0.423828f, 0.434570f},
        {0.504883f, 0.508789f, 0.661133f, 0.671875f},
        {0.280273f, 0.282227f, 0.736328f, 0.745117f},
        {0.825195f, 0.827148f, 0.682617f, 0.689453f},
        {0.847656f, 0.853516f, 0.323242f, 0.324219f},
        {0.360352f, 0.371094f, 0.915039f, 0.926758f},
        {0.565430f, 0.571289f, 0.811523f, 0.820313f},
        {0.813477f, 0.819336f, 0.274414f, 0.276367f},
        {0.958008f, 0.959961f, 0.715820f, 0.719727f},
        {0.516602f, 0.522461f, 0.954102f, 0.960938f},
        {0.868164f, 0.870117f, 0.496094f, 0.498047f},
        {0.753906f, 0.759766f, 0.165039f, 0.171875f},
        {0.997070f, 1.000000f, 0.285156f, 0.293945f},
        {0.552734f, 0.558594f, 0.792969f, 0.801758f},
        {0.542969f, 0.545898f, 0.884766f, 0.893555f},
        {0.647461f, 0.648438f, 0.244141f, 0.254883f},
        {0.001953f, 0.002930f, 0.999023f, 1.000000f},
};

const atlas_pos_t g_font_bold12_shadow_atlas_pos[95] = {
        {0.476563f, 0.484375f, 0.929688f, 0.939453f},
        {0.454102f, 0.461914f, 0.875977f, 0.885742f},
        {0.461914f, 0.469727f, 0.981445f, 0.991211f},
        {0.499023f, 0.506836f, 0.784180f, 0.793945f},
        {0.473633f, 0.480469f, 0.945313f, 0.955078f},
        {0.482422f, 0.488281f, 0.875977f, 0.885742f},
        {0.508789f, 0.516602f, 0.929688f, 0.939453f},
        {0.471680f, 0.479492f, 0.802734f, 0.812500f},
        {0.460938f, 0.463867f, 0.802734f, 0.812500f},
        {0.500977f, 0.507813f, 0.905273f, 0.915039f},
        {0.465820f, 0.473633f, 0.842773f, 0.852539f},
        {0.444336f, 0.451172f, 0.842773f, 0.852539f},
        {0.442383f, 0.452148f, 0.945313f, 0.955078f},
        {0.452148f, 0.459961f, 0.971680f, 0.981445f},
        {0.439453f, 0.448242f, 0.859375f, 0.869141f},
        {0.470703f, 0.478516f, 0.829102f, 0.838867f},
        {0.419922f, 0.428711f, 0.977539f, 0.988281f},
        {0.495117f, 0.503906f, 0.915039f, 0.924805f},
        {0.463867f, 0.471680f, 0.802734f, 0.812500f},
        {0.507813f, 0.514648f, 0.842773f, 0.852539f},
        {0.537109f, 0.544922f, 0.802734f, 0.812500f},
        {0.422852f, 0.431641f, 0.929688f, 0.939453f},
        {0.540039f, 0.552734f, 0.625977f, 0.635742f},
        {0.434570f, 0.442383f, 0.945313f, 0.955078f},
        {0.452148f, 0.459961f, 0.915039f, 0.924805f},
        {0.441406f, 0.448242f, 0.929688f, 0.939453f},
        {0.583008f, 0.589844f, 0.929688f, 0.937500f},
        {0.434570f, 0.441406f, 0.981445f, 0.991211f},
        {0.567383f, 0.574219f, 0.979492f, 0.987305f},
        {0.498047f, 0.504883f, 0.802734f, 0.812500f},
        {0.578125f, 0.584961f, 0.987305f, 0.995117f},
        {0.448242f, 0.453125f, 0.929688f, 0.939453f},
        {0.441406f, 0.448242f, 0.885742f, 0.895508f},
        {0.474609f, 0.481445f, 0.785156f, 0.794922f},
        {0.499023f, 0.501953f, 0.895508f, 0.905273f},
        {0.602539f, 0.605469f, 0.366211f, 0.377930f},
        {0.470703f, 0.477539f, 0.859375f, 0.869141f},
        {0.460938f, 0.463867f, 0.802734f, 0.812500f},
        {0.103516f, 0.114258f, 0.854492f, 0.862305f},
        {0.128906f, 0.135742f, 0.854492f, 0.862305f},
        {0.536133f, 0.542969f, 0.851563f, 0.859375f},
        {0.487305f, 0.494141f, 0.945313f, 0.955078f},
        {0.477539f, 0.484375f, 0.859375f, 0.869141f},
        {0.570313f, 0.575195f, 0.915039f, 0.922852f},
        {0.568359f, 0.575195f, 0.937500f, 0.945313f},
        {0.473633f, 0.477539f, 0.915039f, 0.924805f},
        {0.556641f, 0.563477f, 0.859375f, 0.867188f},
        {0.574219f, 0.581055f, 0.971680f, 0.979492f},
        {0.114258f, 0.124023f, 0.854492f, 0.862305f},
        {0.564453f, 0.571289f, 0.987305f, 0.995117f},
        {0.466797f, 0.473633f, 0.915039f, 0.924805f},
        {0.595703f, 0.601563f, 0.987305f, 0.995117f},
        {0.481445f, 0.488281f, 0.981445f, 0.991211f},
        {0.448242f, 0.453125f, 0.859375f, 0.869141f},
        {0.448242f, 0.455078f, 0.885742f, 0.895508f},
        {0.523438f, 0.530273f, 0.859375f, 0.869141f},
        {0.459961f, 0.466797f, 0.915039f, 0.924805f},
        {0.505859f, 0.512695f, 0.829102f, 0.838867f},
        {0.501953f, 0.508789f, 0.875977f, 0.885742f},
        {0.494141f, 0.500977f, 0.905273f, 0.915039f},
        {0.480469f, 0.487305f, 0.905273f, 0.915039f},
        {0.487305f, 0.494141f, 0.905273f, 0.915039f},
        {0.489258f, 0.492188f, 0.885742f, 0.895508f},
        {0.840820f, 0.846680f, 0.742188f, 0.746094f},
        {0.473633f, 0.481445f, 0.842773f, 0.852539f},
        {0.326172f, 0.332031f, 0.945313f, 0.957031f},
        {0.479492f, 0.488281f, 0.812500f, 0.822266f},
        {0.156250f, 0.163086f, 0.874023f, 0.879883f},
        {0.514648f, 0.523438f, 0.859375f, 0.869141f},
        {0.864258f, 0.868164f, 0.923828f, 0.928711f},
        {0.585938f, 0.589844f, 0.635742f, 0.647461f},
        {0.482422f, 0.486328f, 0.744141f, 0.755859f},
        {0.808594f, 0.812500f, 0.123047f, 0.125000f},
        {0.601563f, 0.609375f, 0.520508f, 0.522461f},
        {0.706055f, 0.712891f, 0.808594f, 0.812500f},
        {0.971680f, 0.978516f, 0.863281f, 0.869141f},
        {0.845703f, 0.849609f, 0.846680f, 0.858398f},
        {0.296875f, 0.302734f, 0.983398f, 0.995117f},
        {0.496094f, 0.500000f, 0.083008f, 0.094727f},
        {0.335938f, 0.340820f, 0.983398f, 0.995117f},
        {0.451172f, 0.454102f, 0.875977f, 0.885742f},
        {0.856445f, 0.859375f, 0.292969f, 0.300781f},
        {0.214844f, 0.217773f, 0.807617f, 0.811523f},
        {0.222656f, 0.234375f, 0.958008f, 0.970703f},
        {0.473633f, 0.480469f, 0.971680f, 0.981445f},
        {0.421875f, 0.428711f, 0.541992f, 0.544922f},
        {0.211914f, 0.214844f, 0.833008f, 0.837891f},
        {0.576172f, 0.583008f, 0.929688f, 0.937500f},
        {0.297852f, 0.300781f, 0.800781f, 0.803711f},
        {0.583984f, 0.590820f, 0.960938f, 0.968750f},
        {0.464844f, 0.468750f, 0.895508f, 0.905273f},
        {0.496094f, 0.502930f, 0.885742f, 0.895508f},
        {0.510742f, 0.514648f, 0.859375f, 0.869141f},
        {0.692383f, 0.694336f, 0.624023f, 0.635742f},
        {0.807617f, 0.808594f, 0.140625f, 0.141602f},
};

/* ---------------- UI begin/end (one per frame) ---------------- */

static int   g_ui_active     = 0;
static GLuint g_ui_bound_tex = 0;

void font_begin(GLuint tex, int win_w, int win_h) {
    if (g_ui_active) return;
    g_ui_active = 1;

    /* Avoid glPushAttrib/glPopAttrib: GLdc may not implement attrib stack fully. */
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_LIGHTING);
    glDisable(GL_CULL_FACE);
    glDisable(GL_SCISSOR_TEST);
    glDisable(GL_ALPHA_TEST);

    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, tex);
    g_ui_bound_tex = tex;

    glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE);

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    /* Texture matrix: identity */
    glMatrixMode(GL_TEXTURE);
    glPushMatrix();
    glLoadIdentity();

    /* Ortho projection in pixel coords (origin top-left), using float matrix
       to avoid double-heavy glOrtho on SH-4. */
    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    {
        const float w = (win_w > 0) ? (float)win_w : 1.0f;
        const float h = (win_h > 0) ? (float)win_h : 1.0f;

        const GLfloat ortho[16] = {
            2.0f / w, 0,         0, 0,
            0,       -2.0f / h,  0, 0,
            0,        0,        -1, 0,
           -1,        1,         0, 1
        };

        glLoadMatrixf(ortho);
    }

    /* Modelview identity */
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();
}

static void rcm_font_bind_texture(GLuint tex) {
    if (!g_ui_active) return;
    if (g_ui_bound_tex == tex) return;
    glBindTexture(GL_TEXTURE_2D, tex);
    g_ui_bound_tex = tex;
}

void font_end(void) {
    if (!g_ui_active) return;

    glMatrixMode(GL_MODELVIEW);
    glPopMatrix();

    glMatrixMode(GL_PROJECTION);
    glPopMatrix();

    glMatrixMode(GL_TEXTURE);
    glPopMatrix();

    glMatrixMode(GL_MODELVIEW);

    /* Return to a reasonable baseline (viewer will set what it needs anyway). */
    glDisable(GL_BLEND);
    glDisable(GL_ALPHA_TEST);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_CULL_FACE);

    g_ui_bound_tex = 0;
    g_ui_active = 0;
}

/* ---------------- Font/UI cached metrics ---------------- */

#define FONT_GLYPH_COUNT 95
#define FONT_IDX_INVALID 0xFF

static const unsigned char g_char_set[FONT_GLYPH_COUNT + 1] =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
    "!\"\xA3$%^&*()-_=+[{]};:'@#~,<.>/?\\| ";

static uint8_t g_font_char_to_idx[256];
static int     g_font_gw[FONT_GLYPH_COUNT];
static int     g_font_gh[FONT_GLYPH_COUNT];
static int     g_font_adv[256];
static int     g_font_line_h = 12;
static int     g_font_cache_w = 0;
static int     g_font_cache_h = 0;
static int     g_font_cache_inited = 0;

static int atlas_px_w(const atlas_pos_t *r, int tex_w) {
    float fw = (r->u1 - r->u0) * (float)tex_w;
    int w = (int)floorf(fw + 0.5f);
    return (w < 1) ? 1 : w;
}

static int atlas_px_h(const atlas_pos_t *r, int tex_h) {
    float fh = (r->v1 - r->v0) * (float)tex_h;
    int h = (int)floorf(fh + 0.5f);
    return (h < 1) ? 1 : h;
}

static void font_bold12_cache_init(int sprites_w, int sprites_h) {
    if (g_font_cache_inited && g_font_cache_w == sprites_w && g_font_cache_h == sprites_h)
        return;

    for (int i = 0; i < 256; ++i) g_font_char_to_idx[i] = FONT_IDX_INVALID;

    for (int i = 0; i < FONT_GLYPH_COUNT; ++i) {
        unsigned char ch = g_char_set[i];
        g_font_char_to_idx[ch] = (uint8_t)i;
    }

    int max_h = 1;
    for (int i = 0; i < FONT_GLYPH_COUNT; ++i) {
        g_font_gw[i] = atlas_px_w(&g_font_bold12_atlas_pos[i], sprites_w);
        g_font_gh[i] = atlas_px_h(&g_font_bold12_atlas_pos[i], sprites_h);
        if (g_font_gh[i] > max_h) max_h = g_font_gh[i];
    }
    g_font_line_h = (max_h < 1) ? 12 : max_h;

    for (int c = 0; c < 256; ++c) {
        if (c == ' ') {
            g_font_adv[c] = 4;
            continue;
        }

        uint8_t idx = g_font_char_to_idx[c];
        if (idx == FONT_IDX_INVALID) {
            /* treat unknown chars as a small gap (prevents OOB + weird glyphs) */
            g_font_adv[c] = 4;
            continue;
        }

        int w = g_font_gw[idx];
        if (w < 2) w = 2;
        g_font_adv[c] = w + 1; /* 1px spacing */
    }

    g_font_cache_w = sprites_w;
    g_font_cache_h = sprites_h;
    g_font_cache_inited = 1;
}

static uint8_t g_charset_lut[256];
static uint8_t g_charset_space_idx = 0;
static int g_charset_inited = 0;

static void charset_init_once(void) {
    if (g_charset_inited) return;

    for (int i = 0; i < 256; ++i) g_charset_lut[i] = FONT_IDX_INVALID;

    for (uint8_t i = 0; i < FONT_GLYPH_COUNT; ++i) {
        const unsigned char ch = g_char_set[i];
        g_charset_lut[ch] = i;
    }

    /* Default unknown chars -> space */
    const uint8_t si = g_charset_lut[(unsigned char)' '];
    g_charset_space_idx = (si != FONT_IDX_INVALID) ? si : (FONT_GLYPH_COUNT - 1);

    g_charset_inited = 1;
}

/* Keep the 3-arg signature so existing callsites compile. */
static int charset_index_for_char(unsigned char ch, int sprites_w, int sprites_h) {
    (void)sprites_w;
    (void)sprites_h;

    charset_init_once();

    uint8_t idx = g_charset_lut[ch];
    if (idx == FONT_IDX_INVALID) idx = g_charset_space_idx;
    return (int)idx;
}

int font_bold12_text_width_px(const char *s, int sprites_w, int sprites_h) {
    font_bold12_cache_init(sprites_w, sprites_h);
    int x = 0;
    for (; *s; ++s) x += g_font_adv[(unsigned char)*s];
    if (x > 0) x -= 1;
    return x;
}

int font_bold12_line_height_px(int sprites_w, int sprites_h) {
    font_bold12_cache_init(sprites_w, sprites_h);
    return g_font_line_h;
}

static int bold12_descender_adjust_px(int c) {
    switch (c) {
        case 'Q':
        case 'g':
        case 'j':
        case 'p':
        case 'q':
        case 'y':
            return 2;
        default:
            return 0;
    }
}

void font_draw_bold12(GLuint sprites_tex,
                             int sprites_w, int sprites_h,
                             int win_w, int win_h,
                             int x0, int top_y,
                             const char *text) {
    if (!sprites_tex || !text || !*text) return;

    int did_begin = 0;
    if (!g_ui_active) {
        font_begin(sprites_tex, win_w, win_h);
        did_begin = 1;
    } else {
        rcm_font_bind_texture(sprites_tex);
    }

    font_bold12_cache_init(sprites_w, sprites_h);

    if (x0 < 0) x0 = 8;
    const int baseline_y = top_y + g_font_line_h;

    /* Shadow pass */
    glColor4ub(0, 0, 0, 255);
    glBegin(GL_TRIANGLES);
    {
        int pen_x = x0;
        for (const char *p = text; *p; ++p) {
            unsigned char ch = (unsigned char)*p;

            if (ch == ' ') {
                pen_x += g_font_adv[ch];
                continue;
            }

            int idx = charset_index_for_char(ch, sprites_w, sprites_h);
            if (idx < 0) {
                pen_x += g_font_adv[ch];
                continue;
            }

            const atlas_pos_t *uv = &g_font_bold12_shadow_atlas_pos[idx];
            const int gw = g_font_gw[idx];
            const int gh = g_font_gh[idx];

            const int y = baseline_y - gh + bold12_descender_adjust_px((int)ch);

            const float x  = (float)(pen_x + 1);
            const float yy = (float)(y + 1);

            glTexCoord2f(uv->u0, uv->v0); glVertex2f(x,       yy);
            glTexCoord2f(uv->u1, uv->v0); glVertex2f(x + gw,  yy);
            glTexCoord2f(uv->u1, uv->v1); glVertex2f(x + gw,  yy + gh);

            glTexCoord2f(uv->u0, uv->v0); glVertex2f(x,       yy);
            glTexCoord2f(uv->u1, uv->v1); glVertex2f(x + gw,  yy + gh);
            glTexCoord2f(uv->u0, uv->v1); glVertex2f(x,       yy + gh);


            pen_x += g_font_adv[ch];
        }
    }
    glEnd();

    /* Main pass */
    glColor4ub(255, 255, 255, 255);
    glBegin(GL_TRIANGLES);
    {
        int pen_x = x0;
        for (const char *p = text; *p; ++p) {
            unsigned char ch = (unsigned char)*p;

            if (ch == ' ') {
                pen_x += g_font_adv[ch];
                continue;
            }

            int idx = charset_index_for_char(ch, sprites_w, sprites_h);
            if (idx < 0) {
                pen_x += g_font_adv[ch];
                continue;
            }

            const atlas_pos_t *uv = &g_font_bold12_atlas_pos[idx];
            const int gw = g_font_gw[idx];
            const int gh = g_font_gh[idx];

            const int y = baseline_y - gh + bold12_descender_adjust_px((int)ch);

            const float x  = (float)pen_x;
            const float yy = (float)y;

            glTexCoord2f(uv->u0, uv->v0); glVertex2f(x,       yy);
            glTexCoord2f(uv->u1, uv->v0); glVertex2f(x + gw,  yy);
            glTexCoord2f(uv->u1, uv->v1); glVertex2f(x + gw,  yy + gh);

            glTexCoord2f(uv->u0, uv->v0); glVertex2f(x,       yy);
            glTexCoord2f(uv->u1, uv->v1); glVertex2f(x + gw,  yy + gh);
            glTexCoord2f(uv->u0, uv->v1); glVertex2f(x,       yy + gh);

            pen_x += g_font_adv[ch];
        }
    }
    glEnd();

    if (did_begin) {
        font_end();
    }
}
