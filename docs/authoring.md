# Asset authoring

SourceIO provides two supported authoring workflows:

- **CS2 Workshop static asset** writes editable CS2 content and can run
  Resource Compiler.
- **Source 1 asset** writes editable SMD/QC/VMT/texture sources for static
  props or an opt-in skeletal reference model and can run VTEX and StudioMDL.

Both workflows use the same non-destructive Blender extraction, preflight,
atomic staging, manifest, diagnostics, and recompile model. Unsupported inputs
fail before success is reported.

These workflows author new assets. The separate `export-modeldoc` and
`export-hammer` commands reconstruct limited editable Source 2 files from
Blender data and cannot recover information absent from compiled resources.

## Shared behavior

- Selected meshes are the default scope; active collection and scene scopes are
  also available.
- Objects can be combined into one asset or exported independently in batch
  mode.
- Evaluated modifiers are exported without modifying source datablocks,
  selection, mode, transforms, or render settings.
- Names and paths are normalized and checked for traversal, reserved names,
  collisions, and target-root containment.
- Existing editable files are protected unless **Overwrite** or `--overwrite`
  is explicit.
- Compilation is opt-in. SourceIO invokes official tools and verifies fresh
  expected outputs instead of writing custom compiled binaries.
- Every successful staging operation writes a deterministic manifest and
  preflight report.
- Recompile validates the target and every authored artifact's size and SHA-256
  before launching a compiler.
- Compiler failures preserve editable files and a bounded diagnostic log.

## CS2 Workshop static assets

Use *File > Export > CS2 Workshop static asset* or:

```text
blender -c sourceio export-cs2 --install-root ROOT --addon ADDON --asset NAME [--scope selected|collection|scene] [--batch] [--overwrite] [--compile]
blender -c sourceio recompile-cs2 --install-root ROOT --addon ADDON MANIFEST
```

### Target

- **CS2 installation** is the root containing both `content` and `game`, such
  as `<steam>/common/Counter-Strike Global Offensive`.
- **Workshop addon** is a lowercase addon name mirrored below
  `content/csgo_addons` and `game/csgo_addons`.
- **Asset name** is a lowercase path-safe model name.

Default authored paths are below `models/sourceio`, `materials/sourceio`, and
`materials/sourceio/textures` inside the content addon.

`SOURCEIO_CS2_WORKSHOP_ROOT` can replace `--install-root`:

```powershell
$env:SOURCEIO_CS2_WORKSHOP_ROOT = "<steam>\common\Counter-Strike Global Offensive"
blender -c sourceio export-cs2 --addon my_addon --asset my_prop --compile
blender -c sourceio recompile-cs2 --addon my_addon sourceio/manifests/my_prop.sourceio.json
```

### Outputs

The workflow emits:

- FBX geometry;
- ModelDoc 36 VMDL;
- `csgo_complex.vfx` VMAT files;
- 32-bit TGA source textures;
- `sourceio/manifests/<asset>.sourceio.json`;
- `sourceio/reports/<asset>.preflight.json`;
- a bounded compile log when compilation fails.

### Supported material graph

An imported or explicitly assigned existing `.vmat` is reused first. Otherwise
the material must have exactly one active Material Output connected directly to
one Principled BSDF.

Supported inputs are:

- **Base Color:** constant color or one directly connected Image Texture.
- **Normal:** unlinked, or an Image Texture Color output through a
  default-strength tangent-space Normal Map node. SourceIO writes
  `TextureNormal` and converts Blender's green-channel handedness.
- **Roughness:** constant or directly connected grayscale Image Texture.
  SourceIO writes linear `TextureRoughness`; Resource Compiler packs it into the
  normal texture's alpha.
- **Metallic:** constant or directly connected grayscale Image Texture.
  SourceIO writes `TextureMetalness` with `F_METALNESS_TEXTURE`. Resource
  Compiler packs opaque metalness into color alpha, or emits `g_tMetalness` when
  alpha testing occupies that channel.
- **Alpha test:** Blender's **Dithered** surface method, using a constant alpha
  or the Alpha output of the Base Color image. SourceIO writes `F_ALPHA_TEST`,
  the Blender threshold, and linear `TextureTranslucency`.

True blended transparency, emission, other texture roles, and procedural graphs
remain blocked until their mappings have official-tool contracts.

### Compilation

Compilation requires CS2 Workshop Tools and
`game/bin/win64/resourcecompiler.exe`. SourceIO verifies mirrored compiled
model, material, and texture outputs. A failed compile leaves the editable
content and writes `sourceio/logs/<asset>.compile.log`.

This is a CS2 static-asset workflow, not a Dota exporter, general Source 2
binary exporter, skeletal exporter, or map-authoring tool.

## Source 1 assets

Use *File > Export > Source 1 asset* or:

```text
blender -c sourceio export-source1 --game-root GAME --content-root CONTENT --asset NAME [--scope selected|collection|scene] [--batch] [--skeletal] [--overwrite] [--compile]
blender -c sourceio recompile-source1 --game-root GAME --content-root CONTENT MANIFEST
```

### Target

- **Game root:** mod or game directory containing `gameinfo.txt`, such as
  `<TF2>/tf`.
- **SDK content root:** directory below which SourceIO may create `modelsrc`,
  `materialsrc`, and `sourceio` metadata folders.
- **Compiler root:** optional directory containing `vtex.exe` and
  `studiomdl.exe`. When omitted, SourceIO checks the normal adjacent `bin`
  directory and then `SOURCEIO_SOURCE1_TOOL_ROOT`.
- **Asset name:** path-safe model name. Default compiled paths are below
  `models/sourceio/<asset>` and `materials/models/sourceio/<asset>`.

The roots can be supplied through environment variables:

```powershell
$env:SOURCEIO_SOURCE1_GAME_ROOT = "<steam>\common\Team Fortress 2\tf"
$env:SOURCEIO_SOURCE1_CONTENT_ROOT = "<steam>\common\Team Fortress 2"
$env:SOURCEIO_SOURCE1_TOOL_ROOT = "<steam>\common\Team Fortress 2\bin"
blender -c sourceio export-source1 --asset my_prop --compile
blender -c sourceio recompile-source1 sourceio/manifests/my_prop.sourceio.json
```

### Outputs

The workflow emits:

- a static or skeletal reference SMD, bodygroup-choice SMDs, and optional
  static collision and LOD SMDs;
- deterministic QC with typed bodygroups, skin families, attachments, and
  hitboxes;
- `VertexLitGeneric` VMT files;
- 32-bit TGA source textures and VTEX settings;
- deterministic manifest and preflight report;
- bounded compile log when compilation fails.

### Collision, LOD, and bodygroup collections

Source 1 assets can use tagged Blender collections for collision geometry, any
number of LODs, and bodygroup choices. Untagged scenes retain the existing
reference-only behavior.

Select a collection and open *Collection Properties > SourceIO Source 1
Authoring*, then assign one of these roles:

- **Reference:** render geometry for the base model.
- **Collision:** static collision geometry. Only one collision collection is
  allowed. Mass zero asks StudioMDL to calculate the mass; **Allow concave**
  permits multiple convex pieces.
- **LOD:** replacement render geometry at the configured switch distance in
  Hammer units. Each LOD needs a unique positive distance.
- **Bodygroup choice:** geometry for one named choice in a bodygroup, or an
  explicit blank choice. Collections in the same bodygroup use the same
  **Bodygroup** value and unique lowercase **Choice** values.

Roles apply only to mesh objects inside the export operator's selected scope.
When at least one collection is explicitly **Reference**, untagged objects are
excluded from the reference model. Without an explicit reference collection,
all scoped meshes not assigned to collision or LOD collections remain
reference geometry.

Tagged collections must not overlap or nest in a way that assigns the same
object more than one role. Collision, LOD, and bodygroup authoring require
**One combined model** grouping. Every LOD must use exactly the same material
identities as the reference model. SourceIO evaluates modifiers and transforms
non-destructively for every role and writes deterministic
`<model>_collision.smd`, `<model>_lodN.smd`, and bodygroup-choice SMD files.

### Skeletal reference models

Enable **Skeletal reference** in the export dialog or pass `--skeletal`.
Reference meshes must resolve to exactly one armature. SourceIO captures its
rest-pose hierarchy and bind transforms, then maps evaluated vertex groups to
exported bones by name. Every exported vertex must have at least one positive
weight for a bone in that armature.

LOD and bodygroup geometry use the same skeleton and weight rules. SourceIO
temporarily evaluates the armature in its rest pose and restores its previous
pose state even when extraction fails. Skeletal collision is deliberately
blocked: this slice does not silently turn jointed physics into static
collision.

This is a reference-model vertical slice. Animation clips, flex/morph
authoring, VTA generation from Blender shape keys, DMX export, and jointed
physics are not included.

### Skins, attachments, and hitboxes

Add a `sourceio_source1_skins` custom property to a Blender material to define
named skin replacements. Its value is a mapping from family names to bare
replacement VMT names:

```python
material["sourceio_source1_skins"] = {
    "blue": "crate_blue",
    "red": "crate_red",
}
```

SourceIO emits the authored materials as skin family zero, then emits named
families deterministically. A material without a replacement in a family keeps
its base name. Replacement VMTs are referenced by name; they must already
exist or be authored separately.

Tag scene objects in *Object Properties > SourceIO Source 1 Authoring*:

- **Attachment** emits the object's transform relative to its target bone.
- **Hitbox** emits the object's bounds and non-negative **Hitbox group**.
- **Bone** overrides the target bone. Direct bone parenting takes precedence;
  otherwise static assets use `root` and skeletal assets use the first
  exported bone.

An object cannot be both an attachment and a hitbox. An Empty used as a hitbox
is treated as a cube with its display size; other object types use their
bounding box.

### Supported materials

- An imported Source 1 VMT or
  `sourceio_export_source1_material_path` is reused through a local Patch VMT.
- A direct Principled graph can author base color/tint, alpha test or blending,
  a default-strength tangent-space normal map, and two-sided state.
- Explicit phong uses `sourceio_source1_phong`; optional custom properties are
  `sourceio_source1_phong_exponent`, `sourceio_source1_phong_boost`,
  `sourceio_source1_phong_tint`, and
  `sourceio_source1_phong_fresnel_ranges`. Albedo-tinted highlights and rim
  lighting use `sourceio_source1_phong_albedo_tint`,
  `sourceio_source1_rimlight`, `sourceio_source1_rimlight_exponent`, and
  `sourceio_source1_rimlight_boost`.
- `sourceio_source1_selfillum` and `sourceio_source1_selfillum_tint` author
  self-illumination. `sourceio_source1_envmap`, `sourceio_source1_envmap_tint`,
  `sourceio_source1_envmap_contrast`, and
  `sourceio_source1_envmap_saturation` author typed envmap controls.
- `sourceio_source1_pbr_to_phong = True` explicitly converts scalar Principled
  roughness and metallic values into deterministic phong exponent, boost, and
  albedo-tint controls. Optional
  `sourceio_source1_pbr_max_exponent` and
  `sourceio_source1_pbr_dielectric_boost` values tune its bounded conversion.
  Explicit phong and PBR conversion are mutually exclusive.
- Roughness and metallic textures are not silently baked or reduced to
  scalars; the current PBR conversion blocks them.
- Unsupported procedural graphs and unrelated PBR slots block export.

### Compilation

SourceIO runs `vtex.exe` and `studiomdl.exe`, sets the `VPROJECT` environment
required by Valve's VTEX, and verifies VTF, MDL, VVD, and DX90 VTX outputs.
Assets with collision geometry additionally require a fresh `.phy` output.
Compiler failures leave editable files and write
`sourceio/logs/<asset>.compile.log` below the SDK content root.

The installed TF2 gate and both Source SDK Base 2013 profiles verify positions,
UVs, winding, normals, material identity, and texture data after official
compilation and SourceIO re-import. The SDK 2013 gate compiles into disposable
mod/content roots; it never writes generated files into Valve's shipped game
directories.

The Source 1 workflow supports static props and opt-in skeletal reference
models. Animation clips, flexes, jointed physics, texture-driven PBR
conversion, and map authoring remain outside the current slice.

## Static Source 2 reconstruction

`export-modeldoc` writes ModelDoc plus static DMX from selected Blender meshes.
`export-hammer` writes a static Hammer VMAP from the active imported map
collection. Both include provenance and loss reports and stage outputs
atomically.

These commands do not promise a lossless compiled-to-authored round trip.
Dynamic rigs, complete animation authoring, original brush topology, and data
not present in compiled resources are not fabricated.

## Legacy export nodes

The old **SourceIO model definition** node tree is deprecated. Its node and
socket types remain registered so existing `.blend` files load, but it no longer
appears in the Add menu and its partial QC evaluator is disabled. Use *File >
Export* or the matching command-line workflow.
