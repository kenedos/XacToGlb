import math
import struct
import os
import bisect
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
import numpy as np
import pyrr

from binary_reader import BinaryReader


# ==========================================================================
# 1. ENUMS and Helper Classes
# ==========================================================================
class SharedChunk:
    MotionEventTable = 50
    Timestamp = 51


class XacChunk:
    XacChunkNode = 0
    XacChunkMesh = 1
    XacChunkSkinninginfo = 2
    XacChunkStdmaterial = 3
    XacChunkStdmateriallayer = 4
    XacChunkFxmaterial = 5
    XacLimit = 6
    XacChunkInfo = 7
    XacChunkMeshlodlevels = 8
    XacChunkStdprogmorphtarget = 9
    XacChunkNodegroups = 10
    XacChunkNodes = 11
    XacChunkStdpmorphtargets = 12
    XacChunkMaterialinfo = 13
    XacChunkNodemotionsources = 14
    XacChunkAttachmentnodes = 15
    XacForce32bit = 0xFFFFFFFF


class XsmChunk:
    XsmChunkInfo = 201  # 0xC9
    XsmChunkSkeletalmotion = 202  # 0xCA


class XacAttribute:
    Positions = 0
    Normals = 1
    Tangents = 2
    Uvcoords = 3
    Colors32 = 4
    AttribOrgvtxnumbers = 5
    Colors128 = 6
    Bitangents = 7


@dataclass
class FileChunk:
    chunk_id: int
    size_in_bytes: int
    version: int


@dataclass
class TimestampData:
    year: int
    month: int
    day: int
    hours: int
    minutes: int
    seconds: int


# ==============================================================================
# 2. XAC (Actor) Data Structures
# ==============================================================================
@dataclass
class XacHeader:
    fourcc: int
    hi_version: int
    lo_version: int
    endian_type: int
    mul_order: int


@dataclass
class XacInfo:
    version: int
    num_lods: Optional[int] = None
    trajectory_node_index: Optional[int] = None
    motion_extraction_node_index: Optional[int] = None
    repositioning_mask: Optional[int] = None
    repositioning_node_index: Optional[int] = None
    motion_extraction_mask: Optional[int] = None
    exporter_high_version: int = 0
    exporter_low_version: int = 0
    retarget_root_offset: Optional[float] = None
    source_app: str = ""
    original_filename: str = ""
    compilation_date: str = ""
    actor_name: str = ""


@dataclass
class XacNode:
    version: int
    local_quat: Tuple
    scale_rot: Tuple
    local_pos: Tuple
    local_scale: Tuple
    shear: Tuple
    skeletal_lods: int
    parent_index: int
    motion_lods: Optional[int] = None
    num_children: Optional[int] = None
    node_flags: Optional[int] = None
    obb: Optional[List[float]] = None
    importance_factor: Optional[float] = None
    node_name: str = ""


@dataclass
class XacSkinInfluence:
    weight: float
    node_number: int


@dataclass
class XacSkinningInfoTableEntry:
    start_index: int
    num_elements: int


@dataclass
class XacSkinningInfo:
    version: int
    node_index: int
    is_for_collision_mesh: int
    lod: Optional[int] = None
    num_local_bones: Optional[int] = None
    num_total_influences: Optional[int] = None
    influences: Optional[List[XacSkinInfluence]] = None
    table: Optional[List[XacSkinningInfoTableEntry]] = None


@dataclass
class XACStandardMaterialLayer:
    version: int
    amount: float
    u_offset: float
    v_offset: float
    u_tiling: float
    v_tiling: float
    rotation_radians: float
    material_number: int
    map_type: int
    blend_mode: Optional[int] = None
    texture_name: str = ""


@dataclass
class XacStandardMaterial:
    version: int
    ambient: Tuple
    diffuse: Tuple
    specular: Tuple
    emissive: Tuple
    shine: float
    shine_strength: float
    opacity: float
    ior: float
    double_sided: int
    wireframe: int
    transparency_type: int
    lod: Optional[int] = None
    num_layers: Optional[int] = None
    material_name: str = ""
    layers: List[XACStandardMaterialLayer] = field(default_factory=list)


@dataclass
class XACVertexAttributeLayer:
    layer_type_id: int
    attrib_size_in_bytes: int
    enable_deformations: int
    is_scale: int
    mesh_data: bytes = b""


@dataclass
class XACSubMesh:
    num_indices: int
    num_verts: int
    material_index: int
    num_bones: int
    indices: List[int] = field(default_factory=list)
    bones: List[int] = field(default_factory=list)


@dataclass
class XACMesh:
    version: int
    node_index: int
    num_org_verts: int
    total_verts: int
    total_indices: int
    num_sub_meshes: int
    num_layers: int
    is_collision_mesh: int
    lod: Optional[int] = None
    vertex_attribute_layers: List[XACVertexAttributeLayer] = field(default_factory=list)
    sub_meshes: List[XACSubMesh] = field(default_factory=list)


@dataclass
class XACMaterialInfo:
    version: int
    num_total_materials: int
    num_standard_materials: int
    num_fx_materials: int
    lod: Optional[int] = None


@dataclass
class XACNodeGroup:
    num_nodes: int
    disabled_on_default: int
    name: str
    data: List[int] = field(default_factory=list)


@dataclass
class XACNodes:
    num_nodes: int
    num_root_nodes: int
    nodes: List[XacNode] = field(default_factory=list)


@dataclass
class XACLimit:
    version: int
    translation_min: Tuple
    translation_max: Tuple
    rotation_min: Tuple
    rotation_max: Tuple
    scale_min: Tuple
    scale_max: Tuple
    limit_flags: List[int]
    node_number: int


@dataclass
class XACFXBitmapParameter:
    name: str
    value_name: str


@dataclass
class XacFxMaterial:
    version: int
    name: str = ""
    effect_file: str = ""
    shader_technique: str = ""
    bitmap_parameters: List[XACFXBitmapParameter] = field(default_factory=list)


# ==============================================================================
# 3. XAC (Actor) Parser
# ==============================================================================
class XACParser:
    def __init__(self, filepath):
        self.filepath = filepath
        self.reader = None
        self.header = None
        self.chunks = []
        self.mesh_chunks_by_node = {}

    def parse(self):
        if not os.path.exists(self.filepath):
            raise FileNotFoundError(f"File not found: {self.filepath}")
        with open(self.filepath, "rb") as f:
            header_bytes = f.read(8)
            if len(header_bytes) < 8:
                raise ValueError("File is too small to be a valid XAC file.")
            endian_char = "<" if header_bytes[6] == 0 else ">"
            f.seek(0)
            self.reader = BinaryReader(f, endian=endian_char)
            self._read_header()
            self._read_all_chunks()
        return self.chunks

    def _read_header(self):
        self.header = XacHeader(
            fourcc=self.reader.read_u32(),
            hi_version=self.reader.read_u8(),
            lo_version=self.reader.read_u8(),
            endian_type=self.reader.read_u8(),
            mul_order=self.reader.read_u8(),
        )

    def _read_all_chunks(self):
        while not self.reader.is_eof():
            try:
                chunk_header = FileChunk(
                    self.reader.read_u32(),
                    self.reader.read_u32(),
                    self.reader.read_u32(),
                )
            except EOFError:
                break
            current_pos = self.reader.tell()
            chunk_id = chunk_header.chunk_id
            handler = {
                XacChunk.XacChunkInfo: self._read_info,
                XacChunk.XacChunkNode: self._read_node,
                XacChunk.XacChunkNodes: self._read_nodes_chunk,
                XacChunk.XacChunkMaterialinfo: self._read_material_info,
                XacChunk.XacChunkStdmaterial: self._read_std_material,
                XacChunk.XacChunkMesh: self._read_mesh,
                XacChunk.XacChunkSkinninginfo: self._read_skinning_info,
                XacChunk.XacChunkNodegroups: self._read_node_group,
                XacChunk.XacLimit: self._read_limit,
                XacChunk.XacChunkFxmaterial: self._read_fx_material,
                SharedChunk.Timestamp: self._read_timestamp_chunk,
            }.get(chunk_id, self._skip_chunk)
            parsed_data = handler(chunk_header)
            if parsed_data:
                self.chunks.append(parsed_data)
                if chunk_id == XacChunk.XacChunkMesh:
                    self.mesh_chunks_by_node[parsed_data.node_index] = parsed_data
            self.reader.seek(current_pos + chunk_header.size_in_bytes)

    def _skip_chunk(self, header: FileChunk):
        return None

    def _read_timestamp_chunk(self, header: FileChunk) -> TimestampData:
        return TimestampData(
            year=self.reader.read_u16(),
            month=self.reader.read_i8(),
            day=self.reader.read_i8(),
            hours=self.reader.read_i8(),
            minutes=self.reader.read_i8(),
            seconds=self.reader.read_i8(),
        )

    def _read_info(self, header: FileChunk) -> XacInfo:
        info = XacInfo(version=header.version)
        if header.version == 1:
            info.repositioning_mask = self.reader.read_u32()
            info.repositioning_node_index = self.reader.read_u32()
            (
                info.exporter_high_version,
                info.exporter_low_version,
            ) = self.reader.read_struct("BB", 2)
            self.reader.read_u16()
        elif header.version == 2:
            info.repositioning_mask = self.reader.read_u32()
            info.repositioning_node_index = self.reader.read_u32()
            (
                info.exporter_high_version,
                info.exporter_low_version,
            ) = self.reader.read_struct("BB", 2)
            info.retarget_root_offset = self.reader.read_f32()
            self.reader.read_u16()
        elif header.version == 3:
            info.trajectory_node_index = self.reader.read_u32()
            info.motion_extraction_node_index = self.reader.read_u32()
            info.motion_extraction_mask = self.reader.read_u32()
            (
                info.exporter_high_version,
                info.exporter_low_version,
            ) = self.reader.read_struct("BB", 2)
            info.retarget_root_offset = self.reader.read_f32()
            self.reader.read_u16()
        elif header.version == 4:
            info.num_lods = self.reader.read_u32()
            info.trajectory_node_index = self.reader.read_u32()
            info.motion_extraction_node_index = self.reader.read_u32()
            (
                info.exporter_high_version,
                info.exporter_low_version,
            ) = self.reader.read_struct("BB", 2)
            info.retarget_root_offset = self.reader.read_f32()
            self.reader.read_u16()
        info.source_app = self.reader.read_string()
        info.original_filename = self.reader.read_string()
        info.compilation_date = self.reader.read_string()
        info.actor_name = self.reader.read_string()
        return info

    def _read_node(self, header: FileChunk) -> XacNode:
        node = XacNode(
            version=header.version,
            local_quat=self.reader.read_quat(),
            scale_rot=self.reader.read_quat(),
            local_pos=self.reader.read_vec3(),
            local_scale=self.reader.read_vec3(),
            shear=self.reader.read_vec3(),
            skeletal_lods=self.reader.read_u32(),
            parent_index=-1,
        )
        if header.version == 4:
            node.motion_lods = self.reader.read_u32()
        node.parent_index = self.reader.read_u32()
        if header.version == 4:
            node.num_children = self.reader.read_u32()
        if header.version >= 2:
            node.node_flags = self.reader.read_u8()
        if header.version >= 3:
            node.obb = self.reader.read_struct("16f", 64)
        if header.version == 4:
            node.importance_factor = self.reader.read_f32()
        self.reader.read_bytes(3)
        node.node_name = self.reader.read_string()
        return node

    def _read_nodes_chunk(self, header: FileChunk) -> XACNodes:
        nodes = XACNodes(
            num_nodes=self.reader.read_u32(), num_root_nodes=self.reader.read_u32()
        )
        for _ in range(nodes.num_nodes):
            nodes.nodes.append(self._read_node(FileChunk(XacChunk.XacChunkNode, 0, 4)))
        return nodes

    def _read_material_info(self, header: FileChunk) -> XACMaterialInfo:
        info = XACMaterialInfo(
            version=header.version,
            num_total_materials=0,
            num_standard_materials=0,
            num_fx_materials=0,
        )
        if header.version == 2:
            info.lod = self.reader.read_u32()
        info.num_total_materials = self.reader.read_u32()
        info.num_standard_materials = self.reader.read_u32()
        info.num_fx_materials = self.reader.read_u32()
        return info

    def _read_std_material(self, header: FileChunk) -> XacStandardMaterial:
        mat = XacStandardMaterial(
            version=header.version,
            ambient=None,
            diffuse=None,
            specular=None,
            emissive=None,
            shine=0,
            shine_strength=0,
            opacity=0,
            ior=0,
            double_sided=0,
            wireframe=0,
            transparency_type=0,
        )
        if header.version == 3:
            mat.lod = self.reader.read_u32()
        mat.ambient = self.reader.read_color()
        mat.diffuse = self.reader.read_color()
        mat.specular = self.reader.read_color()
        mat.emissive = self.reader.read_color()
        mat.shine, mat.shine_strength, mat.opacity, mat.ior = self.reader.read_struct(
            "ffff", 16
        )
        (
            mat.double_sided,
            mat.wireframe,
            mat.transparency_type,
        ) = self.reader.read_struct("BBB", 3)
        if header.version >= 2:
            mat.num_layers = self.reader.read_u8()
        else:
            self.reader.read_u8()
        mat.material_name = self.reader.read_string()
        if header.version >= 2 and mat.num_layers > 0:
            for _ in range(mat.num_layers):
                mat.layers.append(
                    self._read_std_material_layer(
                        FileChunk(XacChunk.XacChunkStdmateriallayer, 0, 2)
                    )
                )
        return mat

    def _read_std_material_layer(self, header: FileChunk) -> XACStandardMaterialLayer:
        layer = XACStandardMaterialLayer(
            version=header.version,
            amount=0,
            u_offset=0,
            v_offset=0,
            u_tiling=0,
            v_tiling=0,
            rotation_radians=0,
            material_number=0,
            map_type=0,
        )
        (
            layer.amount,
            layer.u_offset,
            layer.v_offset,
            layer.u_tiling,
            layer.v_tiling,
            layer.rotation_radians,
        ) = self.reader.read_struct("ffffff", 24)
        layer.material_number = self.reader.read_u16()
        layer.map_type = self.reader.read_u8()
        if header.version == 2:
            layer.blend_mode = self.reader.read_u8()
        else:
            self.reader.read_u8()
        layer.texture_name = self.reader.read_string()
        return layer

    def _read_mesh(self, header: FileChunk) -> XACMesh:
        mesh = XACMesh(
            version=header.version,
            node_index=0,
            num_org_verts=0,
            total_verts=0,
            total_indices=0,
            num_sub_meshes=0,
            num_layers=0,
            is_collision_mesh=0,
        )
        mesh.node_index = self.reader.read_u32()
        if header.version == 2:
            mesh.lod = self.reader.read_u32()
        (
            mesh.num_org_verts,
            mesh.total_verts,
            mesh.total_indices,
            mesh.num_sub_meshes,
            mesh.num_layers,
        ) = self.reader.read_struct("IIIII", 20)
        mesh.is_collision_mesh = self.reader.read_u8()
        self.reader.read_bytes(3)
        for _ in range(mesh.num_layers):
            layer = XACVertexAttributeLayer(
                layer_type_id=self.reader.read_u32(),
                attrib_size_in_bytes=self.reader.read_u32(),
                enable_deformations=self.reader.read_u8(),
                is_scale=self.reader.read_u8(),
            )
            self.reader.read_bytes(2)
            layer.mesh_data = self.reader.read_bytes(
                layer.attrib_size_in_bytes * mesh.total_verts
            )
            mesh.vertex_attribute_layers.append(layer)
        for _ in range(mesh.num_sub_meshes):
            submesh = XACSubMesh(
                num_indices=self.reader.read_u32(),
                num_verts=self.reader.read_u32(),
                material_index=self.reader.read_u32(),
                num_bones=self.reader.read_u32(),
            )
            submesh.indices = list(
                self.reader.read_struct(
                    f"{submesh.num_indices}I", submesh.num_indices * 4
                )
            )
            submesh.bones = list(
                self.reader.read_struct(f"{submesh.num_bones}I", submesh.num_bones * 4)
            )
            mesh.sub_meshes.append(submesh)
        return mesh

    def _read_skinning_info(self, header: FileChunk) -> XacSkinningInfo:
        skin = XacSkinningInfo(
            version=header.version, node_index=0, is_for_collision_mesh=0
        )
        skin.node_index = self.reader.read_u32()
        if header.version == 4:
            skin.lod = self.reader.read_u32()
        if header.version >= 3:
            skin.num_local_bones = self.reader.read_u32()
        if header.version >= 2:
            skin.num_total_influences = self.reader.read_u32()
        skin.is_for_collision_mesh = self.reader.read_u8()
        self.reader.read_bytes(3)
        if header.version >= 2:
            skin.influences = [
                XacSkinInfluence(self.reader.read_f32(), self.reader.read_u32())
                for _ in range(skin.num_total_influences)
            ]
            mesh_chunk = self.mesh_chunks_by_node.get(skin.node_index)
            if not mesh_chunk:
                return skin
            skin.table = [
                XacSkinningInfoTableEntry(
                    self.reader.read_u32(), self.reader.read_u32()
                )
                for _ in range(mesh_chunk.num_org_verts)
            ]
        return skin

    def _read_node_group(self, header: FileChunk) -> XACNodeGroup:
        group = XACNodeGroup(
            num_nodes=self.reader.read_u16(),
            disabled_on_default=self.reader.read_u8(),
            name="",
        )
        self.reader.read_bytes(1)
        group.name = self.reader.read_string()
        group.data = list(
            self.reader.read_struct(f"{group.num_nodes}H", group.num_nodes * 2)
        )
        return group

    def _read_limit(self, header: FileChunk) -> XACLimit:
        return XACLimit(
            version=header.version,
            translation_min=self.reader.read_vec3(),
            translation_max=self.reader.read_vec3(),
            rotation_min=self.reader.read_vec3(),
            rotation_max=self.reader.read_vec3(),
            scale_min=self.reader.read_vec3(),
            scale_max=self.reader.read_vec3(),
            limit_flags=list(self.reader.read_struct("9B", 9)),
            node_number=self.reader.read_u32(),
        )

    def _read_fx_material(self, header: FileChunk) -> Optional[XacFxMaterial]:
        fx = XacFxMaterial(version=header.version)
        if header.version < 2:
            return None

        if header.version == 3:
            _ = self.reader.read_u32()
        num_int_params = self.reader.read_u32()
        num_float_params = self.reader.read_u32()
        num_color_params = self.reader.read_u32()
        num_bool_params = self.reader.read_u32()
        num_vector3_params = self.reader.read_u32()
        num_bitmap_params = self.reader.read_u32()
        fx.name = self.reader.read_string()
        fx.effect_file = self.reader.read_string()
        fx.shader_technique = self.reader.read_string()
        for _ in range(num_int_params):
            self.reader.read_i32()
            self.reader.read_string()
        for _ in range(num_float_params):
            self.reader.read_f32()
            self.reader.read_string()
        for _ in range(num_color_params):
            self.reader.read_color()
            self.reader.read_string()
        for _ in range(num_bool_params):
            self.reader.read_u8()
            self.reader.read_string()
        for _ in range(num_vector3_params):
            self.reader.read_vec3()
            self.reader.read_string()
        for _ in range(num_bitmap_params):
            param_name = self.reader.read_string()
            texture_filename = self.reader.read_string()
            fx.bitmap_parameters.append(
                XACFXBitmapParameter(name=param_name, value_name=texture_filename)
            )
        return fx


# ==============================================================================
# 4. EXTRACTION LOGIC
# ==============================================================================
@dataclass
class SkinningData:
    """Holds bone IDs and weights for each vertex."""
    bone_ids: np.ndarray
    bone_weights: np.ndarray


@dataclass
class RenderableMesh:
    """Represents a single mesh with its own vertex data and list of sub-meshes for drawing."""
    vertices: np.ndarray
    normals: Optional[np.ndarray]
    uvs: Optional[np.ndarray]
    skinning_data: Optional[SkinningData]
    sub_meshes: List[dict] = field(default_factory=list)
    obb_matrix: Optional[pyrr.Matrix44] = None
    obb_owner_node_index: Optional[int] = None
    is_collision: bool = False


@dataclass
class SkeletonData:
    """Holds all data related to the model's skeleton."""
    bind_pose_transforms: List[pyrr.Matrix44] = field(default_factory=list)
    inverse_bind_matrices: List[pyrr.Matrix44] = field(default_factory=list)
    nodes: List[XacNode] = field(default_factory=list)
    bones: List[Tuple[np.ndarray, np.ndarray]] = field(default_factory=list)
    mul_order: int = 1
    root_transform: pyrr.Matrix44 = field(default_factory=pyrr.Matrix44.identity)


def get_local_transform(node: XacNode, mul_order: int) -> pyrr.Matrix44:
    """Calculates the local transformation matrix for a given node, applying Noesis correction."""
    pos_x, pos_y, pos_z = node.local_pos
    transformed_pos = pyrr.Vector3([-pos_x, pos_z, pos_y])

    quat_x, quat_y, quat_z, quat_w = node.local_quat
    transformed_quat = pyrr.Quaternion([-quat_x, quat_z, quat_y, -quat_w])

    if pyrr.vector.length(transformed_quat) > 0:
        transformed_quat = pyrr.quaternion.normalize(transformed_quat)

    translation = pyrr.matrix44.create_from_translation(transformed_pos)
    rotation = pyrr.matrix44.create_from_quaternion(transformed_quat)
    scale = pyrr.matrix44.create_from_scale(node.local_scale)

    if mul_order == 0:
        return scale @ rotation @ translation
    else:
        return rotation @ scale @ translation


def extract_renderable_data(
        xac_filepath: str, existing_skeleton: Optional[SkeletonData] = None
) -> (List[RenderableMesh], Optional[SkeletonData], List):
    """
    Parses an XAC file and extracts all data required for rendering and animation.
    """
    parser = XACParser(xac_filepath)
    all_chunks = parser.parse()

    root_transform = pyrr.matrix44.create_from_x_rotation(math.radians(90.0))

    skeleton_data = None
    if existing_skeleton:
        skeleton_data = existing_skeleton
    else:
        nodes_chunk = next((c for c in all_chunks if isinstance(c, XACNodes)), None)
        if nodes_chunk:
            skeleton_data = SkeletonData(
                nodes=nodes_chunk.nodes, mul_order=parser.header.mul_order
            )
            skeleton_data.root_transform = root_transform

            num_nodes = nodes_chunk.num_nodes
            nodes = nodes_chunk.nodes
            world_transforms = [pyrr.Matrix44.identity() for _ in range(num_nodes)]

            processed_nodes = [False] * num_nodes

            def build_world_transforms(node_index):
                if processed_nodes[node_index]:
                    return world_transforms[node_index]

                node = nodes[node_index]
                local_transform = get_local_transform(node, parser.header.mul_order)

                parent_id = int(node.parent_index)
                if parent_id != 4294967295 and parent_id < num_nodes:
                    parent_transform = build_world_transforms(parent_id)
                    world_transforms[node_index] = local_transform @ parent_transform
                else:
                    world_transforms[node_index] = local_transform @ root_transform

                processed_nodes[node_index] = True
                return world_transforms[node_index]

            for i in range(num_nodes):
                if not processed_nodes[i]:
                    build_world_transforms(i)

            skeleton_data.bind_pose_transforms = world_transforms
            skeleton_data.inverse_bind_matrices = [
                pyrr.matrix44.inverse(m)
                if np.linalg.det(m) != 0
                else pyrr.Matrix44.identity()
                for m in world_transforms
            ]

    # --- Build Material Map ---
    material_index_map = {}
    current_material_index = 0
    for chunk in all_chunks:
        if isinstance(chunk, XacStandardMaterial):
            texture_name = None
            if chunk.layers:
                diffuse_layer = next((l for l in chunk.layers if l.map_type == 2), None)
                if not diffuse_layer:
                    diffuse_layer = chunk.layers[0]
                texture_name = diffuse_layer.texture_name
            material_index_map[current_material_index] = {
                "name": chunk.material_name,
                "texture": texture_name,
            }
            current_material_index += 1
        elif isinstance(chunk, XacFxMaterial):
            diffuse_tex_param = next(
                (p for p in chunk.bitmap_parameters if "diffuse" in p.name.lower()),
                None,
            )
            if not diffuse_tex_param and chunk.bitmap_parameters:
                diffuse_tex_param = chunk.bitmap_parameters[0]
            material_index_map[current_material_index] = {
                "name": chunk.name,
                "texture": diffuse_tex_param.value_name if diffuse_tex_param else None,
            }
            current_material_index += 1

    # --- Build Skinning Info Map ---
    skinning_map = {
        c.node_index: c for c in all_chunks if isinstance(c, XacSkinningInfo)
    }

    # --- Extract Meshes and Skinning Data ---
    renderable_meshes = []
    mesh_chunks = [c for c in all_chunks if isinstance(c, XACMesh)]
    for mesh_chunk in mesh_chunks:
        pos_layer = next(
            (
                l
                for l in mesh_chunk.vertex_attribute_layers
                if l.layer_type_id == XacAttribute.Positions
            ),
            None,
        )
        norm_layer = next(
            (
                l
                for l in mesh_chunk.vertex_attribute_layers
                if l.layer_type_id == XacAttribute.Normals
            ),
            None,
        )
        uv_layer = next(
            (
                l
                for l in mesh_chunk.vertex_attribute_layers
                if l.layer_type_id == XacAttribute.Uvcoords
            ),
            None,
        )
        org_vtx_layer = next(
            (
                l
                for l in mesh_chunk.vertex_attribute_layers
                if l.layer_type_id == XacAttribute.AttribOrgvtxnumbers
            ),
            None,
        )

        if not pos_layer or not org_vtx_layer:
            continue

        positions = (
            np.frombuffer(pos_layer.mesh_data, dtype=np.float32).reshape(-1, 3).copy()
        )
        normals = (
            np.frombuffer(norm_layer.mesh_data, dtype=np.float32).reshape(-1, 3).copy()
            if norm_layer
            else None
        )
        uvs = (
            np.frombuffer(uv_layer.mesh_data, dtype=np.float32).reshape(-1, 2).copy()
            if uv_layer
            else None
        )
        if uvs is not None:
            uvs[:, 1] = 1.0 - uvs[:, 1]

        positions_orig = positions.copy()
        positions[:, 0] = -positions_orig[:, 0]
        positions[:, 1] = positions_orig[:, 2]
        positions[:, 2] = positions_orig[:, 1]

        if normals is not None:
            normals_orig = normals.copy()
            normals[:, 0] = -normals_orig[:, 0]
            normals[:, 1] = normals_orig[:, 2]
            normals[:, 2] = normals_orig[:, 1]

        if positions.size > 0:
            pos_h = np.hstack([positions, np.ones((positions.shape[0], 1))])
            positions = (pos_h @ root_transform)[:, :3]

        if normals is not None and normals.size > 0:
            normal_transform_3x3 = root_transform[:3, :3]
            normals = normals @ normal_transform_3x3

        org_vtx_numbers = np.frombuffer(org_vtx_layer.mesh_data, dtype=np.uint32)

        skinning_data = None
        skin_info = skinning_map.get(mesh_chunk.node_index)

        if skeleton_data and skin_info and skin_info.table and skin_info.influences:
            num_verts = mesh_chunk.total_verts
            bone_ids = np.zeros((num_verts, 4), dtype=np.int32)
            bone_weights = np.zeros((num_verts, 4), dtype=np.float32)

            max_node_index = (
                len(skeleton_data.nodes) - 1
                if skeleton_data and skeleton_data.nodes
                else -1
            )

            for i in range(num_verts):
                original_vertex_index = org_vtx_numbers[i]
                if original_vertex_index < len(skin_info.table):
                    table_entry = skin_info.table[original_vertex_index]
                    for j in range(min(4, table_entry.num_elements)):
                        influence = skin_info.influences[table_entry.start_index + j]

                        if (
                                max_node_index != -1
                                and influence.node_number > max_node_index
                        ):
                            bone_ids[i, j] = 0
                            bone_weights[i, j] = 0.0
                        else:
                            bone_ids[i, j] = influence.node_number
                            bone_weights[i, j] = influence.weight

            weight_sums = np.sum(bone_weights, axis=1, keepdims=True)
            weight_sums[weight_sums == 0] = 1.0
            bone_weights /= weight_sums
            skinning_data = SkinningData(bone_ids=bone_ids, bone_weights=bone_weights)

        new_renderable_mesh = RenderableMesh(
            vertices=positions,
            normals=normals,
            uvs=uvs,
            skinning_data=skinning_data,
            sub_meshes=[],
            is_collision=bool(mesh_chunk.is_collision_mesh),
        )

        SWIZZLE_MATRIX = pyrr.Matrix44(
            [
                [-1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )

        if skeleton_data and mesh_chunk.node_index < len(skeleton_data.nodes):
            node_for_mesh = skeleton_data.nodes[mesh_chunk.node_index]
            if node_for_mesh.obb:
                obb_from_file = pyrr.Matrix44(node_for_mesh.obb)
                t, r, s = pyrr.matrix44.decompose(obb_from_file)

                swizzled_t = pyrr.Vector3([-t[0], t[2], t[1]])
                swizzled_r = pyrr.Quaternion([-r[0], r[2], r[1], -r[3]])
                if pyrr.vector.length(swizzled_r) > 0:
                    swizzled_r = pyrr.quaternion.normalize(swizzled_r)

                t_mat = pyrr.matrix44.create_from_translation(swizzled_t)
                r_mat = pyrr.matrix44.create_from_quaternion(swizzled_r)
                s_mat = pyrr.matrix44.create_from_scale(s)

                new_renderable_mesh.obb_matrix = (
                        SWIZZLE_MATRIX @ obb_from_file @ SWIZZLE_MATRIX
                )
                new_renderable_mesh.obb_owner_node_index = mesh_chunk.node_index

        vertex_offset = 0
        for submesh in mesh_chunk.sub_meshes:
            material_info = material_index_map.get(submesh.material_index, {})
            if submesh.num_indices > 0:
                absolute_indices = (
                        np.array(submesh.indices, dtype=np.uint32) + vertex_offset
                )
                new_renderable_mesh.sub_meshes.append(
                    {
                        "indices": absolute_indices,
                        "material_name": material_info.get(
                            "name", f"Material_{submesh.material_index}"
                        ),
                        "texture_name": material_info.get("texture"),
                        "index_count": submesh.num_indices,
                    }
                )
            vertex_offset += submesh.num_verts

        if new_renderable_mesh.sub_meshes:
            renderable_meshes.append(new_renderable_mesh)

    # --- Finalize Skeleton Data (Bones for drawing) ---
    if skeleton_data:
        nodes = skeleton_data.nodes
        world_transforms = skeleton_data.bind_pose_transforms
        num_nodes = len(nodes)
        for i, node in enumerate(nodes):
            parent_id = int(node.parent_index)
            if parent_id != 4294967295 and parent_id < num_nodes:
                parent_pos = world_transforms[parent_id][3, :3]
                child_pos = world_transforms[i][3, :3]
                skeleton_data.bones.append((parent_pos, child_pos))

    return renderable_meshes, skeleton_data, all_chunks


# ==============================================================================
# 5. XSM (Skeletal Motion) Parser and Animation Data
# ==============================================================================
@dataclass
class XsmHeader:
    fourcc: int
    hi_version: int
    lo_version: int
    endian_type: int
    padding: int


@dataclass
class XsmSkeletalSubMotion:
    pose_rot: Tuple
    bind_pose_rot: Tuple
    pose_scale_rot: Tuple
    bind_pose_scale_rot: Tuple
    pose_pos: Tuple
    pose_scale: Tuple
    bind_pose_pos: Tuple
    bind_pose_scale: Tuple
    num_pos_keys: int
    num_rot_keys: int
    num_scale_keys: int
    num_scale_rot_keys: int
    max_error: float
    node_name: str
    pos_keys: List[Tuple[float, Tuple]] = field(default_factory=list)
    rot_keys: List[Tuple[float, Tuple]] = field(default_factory=list)
    scale_keys: List[Tuple[float, Tuple]] = field(default_factory=list)


@dataclass
class AnimationTrack:
    """Holds all keyframes for a single node's animation."""
    pos_keys: List[Tuple[float, pyrr.Vector3]] = field(default_factory=list)
    rot_keys: List[Tuple[float, pyrr.Quaternion]] = field(default_factory=list)
    scale_keys: List[Tuple[float, pyrr.Vector3]] = field(default_factory=list)

    def get_interpolated_value(self, keys, time):
        if not keys:
            return None
        if time <= keys[0][0]:
            return keys[0][1]
        if time >= keys[-1][0]:
            return keys[-1][1]

        timestamps = [k[0] for k in keys]
        next_idx = bisect.bisect_right(timestamps, time)
        prev_idx = next_idx - 1

        prev_key_time, prev_key_val = keys[prev_idx]
        next_key_time, next_key_val = keys[next_idx]

        time_diff = next_key_time - prev_key_time
        factor = (time - prev_key_time) / time_diff if time_diff > 0 else 0

        if isinstance(prev_key_val, pyrr.Quaternion):
            return pyrr.quaternion.slerp(prev_key_val, next_key_val, factor)
        else:
            return prev_key_val * (1.0 - factor) + next_key_val * factor


@dataclass
class AnimationData:
    """Contains all animation tracks for a skeleton, keyed by node name."""
    duration: float = 0.0
    tracks: Dict[str, AnimationTrack] = field(default_factory=dict)
    tracks_by_index: Dict[int, AnimationTrack] = field(default_factory=dict)
    mul_order: int = 1  # Default to 3DS Max style

    def map_tracks_to_node_indices(self, nodes: List[XacNode]):
        """Create a mapping from node index to track for faster lookups."""
        node_map = {node.node_name: i for i, node in enumerate(nodes)}
        self.tracks_by_index = {
            node_map[name]: track for name, track in self.tracks.items() if name in node_map
        }

    def get_local_transform(self, node_index: int, node: XacNode, time: float) -> pyrr.Matrix44:
        """Calculate the local transform matrix for a node at a given time."""
        track = self.tracks_by_index.get(node_index)

        pos = self.get_value(track.pos_keys, time, pyrr.Vector3(node.local_pos)) if track else pyrr.Vector3(
            node.local_pos)
        rot = self.get_value(track.rot_keys, time, pyrr.Quaternion(node.local_quat)) if track else pyrr.Quaternion(
            node.local_quat)
        scale = self.get_value(track.scale_keys, time, pyrr.Vector3(node.local_scale)) if track else pyrr.Vector3(
            node.local_scale)

        # Apply the Noesis swizzle to the final (potentially animated) values
        transformed_pos = pyrr.Vector3([-pos[0], pos[2], pos[1]])
        transformed_rot = pyrr.Quaternion([-rot[0], rot[2], rot[1], -rot[3]])

        if pyrr.vector.length(transformed_rot) > 0:
            transformed_rot = pyrr.quaternion.normalize(transformed_rot)

        pos_mat = pyrr.matrix44.create_from_translation(transformed_pos)
        rot_mat = pyrr.matrix44.create_from_quaternion(transformed_rot)
        # Use the original scale, as it's not swizzled
        scale_mat = pyrr.matrix44.create_from_scale(scale)

        if self.mul_order == 0:
            return scale_mat @ rot_mat @ pos_mat
        else:
            return rot_mat @ scale_mat @ pos_mat

    def get_value(self, keys, time, default_value):
        """Helper to get interpolated value or default if track is missing."""
        if not keys:
            return default_value
        return AnimationTrack().get_interpolated_value(keys, time)


class XSMParser:
    def __init__(self, filepath):
        self.filepath = filepath
        self.reader = None
        self.header = None

    def parse(self) -> AnimationData:
        if not os.path.exists(self.filepath):
            raise FileNotFoundError(f"File not found: {self.filepath}")

        with open(self.filepath, "rb") as f:
            header_bytes = f.read(8)
            if len(header_bytes) < 8:
                raise ValueError("File is too small to be a valid XSM file.")
            endian_char = "<" if header_bytes[6] == 0 else ">"
            f.seek(0)
            self.reader = BinaryReader(f, endian=endian_char)
            self._read_header()
            return self._read_all_chunks()

    def _read_header(self):
        self.header = XsmHeader(
            fourcc=self.reader.read_u32(),
            hi_version=self.reader.read_u8(),
            lo_version=self.reader.read_u8(),
            endian_type=self.reader.read_u8(),
            padding=self.reader.read_u8(),
        )

    def _read_all_chunks(self) -> AnimationData:
        animation_data = AnimationData()

        while not self.reader.is_eof():
            try:
                chunk_header = FileChunk(
                    chunk_id=self.reader.read_u32(),
                    size_in_bytes=self.reader.read_u32(),
                    version=self.reader.read_u32(),
                )
            except EOFError:
                break

            current_pos = self.reader.tell()
            chunk_id = chunk_header.chunk_id
            handler = {
                XsmChunk.XsmChunkSkeletalmotion: self._read_skeletal_motion_chunk,
                SharedChunk.Timestamp: self._read_timestamp_chunk
            }.get(chunk_id, self._skip_chunk)

            handler(chunk_header, animation_data)

            self.reader.seek(current_pos + chunk_header.size_in_bytes)

        animation_data.duration = max((k[0] for track in animation_data.tracks.values() for k in track.pos_keys),
                                      default=0.0)
        return animation_data

    def _skip_chunk(self, header: FileChunk, anim_data: AnimationData):
        pass

    def _read_timestamp_chunk(self, header: FileChunk, anim_data: AnimationData):
        # Timestamp data - just skip it
        self.reader.read_u16()  # year
        self.reader.read_i8()   # month
        self.reader.read_i8()   # day
        self.reader.read_i8()   # hours
        self.reader.read_i8()   # minutes
        self.reader.read_i8()   # seconds

    def _read_skeletal_motion_chunk(self, header: FileChunk, anim_data: AnimationData):
        num_sub_motions = self.reader.read_u32()
        for _ in range(num_sub_motions):
            sub_motion = self._read_skeletal_sub_motion()
            track = AnimationTrack()
            for time, pos in sub_motion.pos_keys:
                track.pos_keys.append((time, pyrr.Vector3(pos)))
            for time, rot in sub_motion.rot_keys:
                track.rot_keys.append((time, pyrr.Quaternion(rot)))
            for time, scale in sub_motion.scale_keys:
                track.scale_keys.append((time, pyrr.Vector3(scale)))
            anim_data.tracks[sub_motion.node_name] = track

    def _read_skeletal_sub_motion(self) -> XsmSkeletalSubMotion:
        sub_motion = XsmSkeletalSubMotion(
            pose_rot=self.reader.read_quat16(),
            bind_pose_rot=self.reader.read_quat16(),
            pose_scale_rot=self.reader.read_quat16(),
            bind_pose_scale_rot=self.reader.read_quat16(),
            pose_pos=self.reader.read_vec3(),
            pose_scale=self.reader.read_vec3(),
            bind_pose_pos=self.reader.read_vec3(),
            bind_pose_scale=self.reader.read_vec3(),
            num_pos_keys=self.reader.read_u32(),
            num_rot_keys=self.reader.read_u32(),
            num_scale_keys=self.reader.read_u32(),
            num_scale_rot_keys=self.reader.read_u32(),
            max_error=self.reader.read_f32(),
            node_name=self.reader.read_string(),
        )
        for _ in range(sub_motion.num_pos_keys):
            pos = self.reader.read_vec3()
            time = self.reader.read_f32()
            sub_motion.pos_keys.append((time, pos))

        for _ in range(sub_motion.num_rot_keys):
            rot = self.reader.read_quat16()
            time = self.reader.read_f32()
            sub_motion.rot_keys.append((time, rot))

        for _ in range(sub_motion.num_scale_keys):
            scale = self.reader.read_vec3()
            time = self.reader.read_f32()
            sub_motion.scale_keys.append((time, scale))

        # Skip ScaleRotKeys as they are not typically used
        for _ in range(sub_motion.num_scale_rot_keys):
            self.reader.read_quat16()
            self.reader.read_f32()

        return sub_motion
