### rsc-rcm-format ###

A suite of converters that take RuneScape Classic OB3, HEI, and DAT data, from their original JAG/MEM archives, and recreates serialized models for them, in the .RCM binary blob format. For rsc-c/RuneCast.

### Features: ### 

Converters for Landscape (`RCL`), Scenery/Object/RS2 Beta item (`RCM`), and Wall/Roof (`RCW`) models.
The tool is capable of generating an accurate, to-scale, reproduction of (theoretically) every world chunk in RuneScape Classic, and the items therein.

The RCM format is designed to be optimized for the Sega Dreamcast, but should also be able to replace the original game's models, using an OpenGL renderer.
Output models are named after their entries in a JAG/MEM archive.

`rcm_viewer` is an OpenGL 1.1 RCM model viewer for Linux and Dreamcast. The Dreamcast build requires GLdc.
This viewer also provides a readout of important stats for every model. Polygon/Vertex counts, submesh counts, UV coordinates,and model format Magic.
