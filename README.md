### rsc-rcm-format - RuneCast Mesh/Runescape Classic Mesh ###

A suite of converters that take RuneScape Classic OB3, HEI, and DAT data, from their original JAG/MEM archives, and recreates serialized models for them, in the .RCM binary blob format. For rsc-c/RuneCast.

### Features: ### 

Converters for Landscape (`RCL`), Scenery/Object/RS2 Beta item (`RCM`), and Wall/Roof (`RCW`) models.
The tool is capable of generating an accurate, to-scale, reproduction of (theoretically) every world chunk in RuneScape Classic, and the items therein.

The RCM format is designed to be optimized for the Sega Dreamcast, but should also be able to replace the original game's models, using an OpenGL renderer.
Output models are named after their entries in a JAG/MEM archive.

`rcm_viewer` is an OpenGL 1.1 RCM model viewer for Linux and Dreamcast. The Dreamcast build requires GLdc.
This viewer also provides a readout of important stats for every model. Polygon/Vertex counts, submesh counts, UV coordinates,and model format Magic.

### RCM ### 

RCM is the typical Scenery/Object/3D Ground Items model format, made to replace `models36.jag`. It is split into three types:

RCM1: This type is for textured or partially-textured meshes. Vertices are stored as Floating-Point values.

RCM2: More or less the above, but vertices are stored with Integer-Precision, instead.

RCM3: This is for untextured meshes, stored with Floating-Point precision. Only untextured meshes are generated with `rsc2rcm`, and models with Texture/UV data are discarded in the conversion process. Use a combination of RCM1 and RCM3 to save memory and I/O, if used in RuneCast.

### RCL ### 

RCL is the Landscape variant of RCM. The end product of an RCL model is an accurate recreation of a 48x48 RuneScape Classic landscape chunk that seamlessly fits with its neighbors. 
Accuracy to the original game has one exception-- blended vertex colors on the ground, versus one-color-tint-per-tile. All of the tile colors in the landscape, are now blended smoothly, similarly to RuneScape 2.

Currently, the only RCL type is RCL1, which is Textured, with Floating-Point Vertex Precision. 

RCL also contains the per-chunk tilemap and heightfield metadata from `land63.jag`/`land63.mem`, thus rendering the loading and allocation of these JAG archives unnecessary, in RuneCast. All tile overlays are baked straight into the landscape. This includes 45 degree half-tile slices, roads, bridges, water, lava, holes, et al.

### RCW ### 

RCW is the variant of RCM for baking walls and roofs into static meshes. This also contains the collision and "are you in a building" flag metadata used by `maps63` archives, eliminating the need to load/allocate for them.
