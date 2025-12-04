# XAC to GLB Converter

A standalone tool to batch convert XAC (Actor) files to GLB (GLTF 2.0 binary) format, with support for XSM skeletal animations.

## Installation

1. Make sure you have Python 3.8+ installed
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

### Basic Usage

1. Place your files in the `input` folder (preserving IPF folder structure)
2. Run the converter:
   ```
   python convert.py
   ```
3. Find the converted `.glb` files in the `output` folder

The converter automatically finds animations and textures from parallel IPF folder structures.

### With Custom Texture Paths

If your textures are in different locations:
```
python convert.py --texture-paths "C:/game/textures" "D:/assets/textures"
```

### Without Animations

To convert models without including animations:
```
python convert.py --no-animations
```

### Custom Input/Output Directories

```
python convert.py --input "C:/my/xac/files" --output "C:/my/glb/output"
```

## Features

- Recursive conversion of all `.xac` files
- Automatic detection of `.xsm` skeletal animations from IPF folder structure
- Automatic texture discovery from parallel texture IPF folders
- Preserves directory structure in output
- Embeds textures into GLB files
- Supports skinned meshes with skeleton data
- Case-insensitive texture file matching

## IPF Folder Structure

The converter understands the standard IPF folder structure. Place files maintaining their original paths:

```
input/
├── char_hi.ipf/
│   └── monster/
│       └── monster_popolion_set.xac      # Model file
├── animation.ipf/
│   └── monster/
│       └── popolion/
│           ├── popolion_idle.xsm         # Animations (auto-detected)
│           ├── popolion_run.xsm
│           ├── popolion_atk1.xsm
│           └── popolion_hit.xsm
└── char_texture.ipf/
    └── monster/
        └── popolion/
            └── popolion.dds              # Texture (auto-detected)
```

The converter will:
1. Find the `.xac` model in `char_hi.ipf/monster/`
2. Automatically discover animations in `animation.ipf/monster/popolion/`
3. Automatically discover textures in `char_texture.ipf/monster/popolion/`

Output will preserve the model's folder structure:
```
output/
└── char_hi.ipf/
    └── monster/
        └── monster_popolion_set.glb      # Converted with embedded textures & animations
```

## Project Structure

```
XacToGlb/
├── convert.py         # Main batch conversion script
├── gltf_exporter.py   # GLTF/GLB export with animation support
├── xac_parser.py      # XAC and XSM file format parsers
├── binary_reader.py   # Binary file reading utilities
├── requirements.txt   # Python dependencies
├── input/             # Place IPF folder structure here
└── output/            # Converted .glb files appear here
```

## Supported Formats

### Input
- `.xac` - Actor/model files (mesh, skeleton, materials)
- `.xsm` - Skeletal motion/animation files
- `.dds`, `.png`, `.tga` - Texture files

### Output
- `.glb` - GLTF 2.0 binary format with embedded textures and animations
