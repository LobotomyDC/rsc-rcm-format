# Makefile - RCM Viewer (Dreamcast/KallistiOS + SDL2 + GLdc)

TARGET      := rcm_viewer.elf
SRCS        := rcm_viewer.c rcm_atlas.c rcm_font.c dtex.c
OBJS        := $(SRCS:.c=.o)

# KOS ports path (matches your RuneCast layout)
KOS_PORTS   ?= $(KOS_BASE)/../kos-ports

# -----------------------------
# Build configuration
# -----------------------------
# Use SDL2 from kos-ports (Dreamcast build)
SDL2_INC    := $(KOS_PORTS)/include/SDL2
SDL2_LIBDIR := $(KOS_PORTS)/lib/dreamcast

# Common flags
CFLAGS      += -std=gnu99 -fwrapv -DSDL_MAIN_HANDLED
CFLAGS      += -DGLDC_GL
CFLAGS      += -I. -I$(KOS_BASE)/include -I$(KOS_BASE)/include/GL -I$(SDL2_INC)

# Warnings: tune as desired (kept reasonable for KOS builds)
CFLAGS      += -Wall -Wextra -Wno-unused-function -Wno-unused-variable -Wno-sign-compare -Wno-strict-aliasing

# Debug vs release
ifdef DEBUG
CFLAGS      += -O0 -g
LDFLAGS     +=
else
CFLAGS      += -O3 -ffast-math
LDFLAGS     += -s
endif

# Linker flags / libs
LDFLAGS     += -L$(SDL2_LIBDIR) -lSDL2
LDFLAGS     += -lGL -lm

# -----------------------------
# KOS rules
# -----------------------------
include $(KOS_BASE)/Makefile.rules

all: rm-elf $(TARGET)

rm-elf:
	-rm -f $(TARGET)

clean: rm-elf
	-rm -f $(OBJS)

%.o: %.c
	kos-cc $(CFLAGS) -c $< -o $@

$(TARGET): $(OBJS)
	@echo "Linking $(TARGET)..."
	kos-cc $(CFLAGS) -o $(TARGET) $(OBJS) $(LDFLAGS)

run: $(TARGET)
	$(KOS_LOADER) $(TARGET)

# -----------------------------
# Optional CDI image generation
# -----------------------------
CDI_OUTPUT   ?= RCM_Viewer.cdi
CDI_DATA_DIR ?= ./cache

ifdef CDI
POST_BUILD_CMD = mkdcdisc -e $(TARGET) -d $(CDI_DATA_DIR) -o $(CDI_OUTPUT) -N
else
POST_BUILD_CMD = @echo "CDI=1 not set, skipping CDI generation."
endif

cdi: $(TARGET)
	$(POST_BUILD_CMD)

