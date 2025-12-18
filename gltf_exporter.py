import io
import math
import os
from typing import Optional, List

import imageio.v2 as imageio
import numpy as np
import pyrr
from pygltflib import (
    GLTF2, Scene, Node, Mesh, Buffer, BufferView, Accessor, Animation, Skin, Primitive,
    FLOAT, VEC2, VEC3, VEC4, SCALAR, MAT4, UNSIGNED_SHORT, UNSIGNED_INT,
    ARRAY_BUFFER, ELEMENT_ARRAY_BUFFER,
    Image as GLTFImage, Texture, Material, TextureInfo, PbrMetallicRoughness, Sampler,
    AnimationChannel, AnimationSampler, AnimationChannelTarget
)

from xac_parser import extract_renderable_data, get_local_transform, XSMParser, AnimationData


def find_texture_path(texture_name: str, search_paths: List[str]) -> Optional[str]:
    """Find texture file in search paths (case-insensitive)."""
    if not texture_name:
        return None

    # Common image extensions to try
    extensions = ['', '.png', '.dds', '.tga', '.jpg', '.jpeg', '.bmp']

    for search_path in search_paths:
        if not os.path.isdir(search_path):
            continue

        # Get all files in the directory
        try:
            files_in_dir = os.listdir(search_path)
        except OSError:
            continue

        # Create case-insensitive lookup
        lower_to_actual = {f.lower(): f for f in files_in_dir}

        for ext in extensions:
            test_name = texture_name + ext if ext else texture_name
            test_lower = test_name.lower()

            if test_lower in lower_to_actual:
                return os.path.join(search_path, lower_to_actual[test_lower])

            # Also try without extension if texture_name already has one
            base_name = os.path.splitext(texture_name)[0]
            if ext:
                test_name = base_name + ext
                test_lower = test_name.lower()
                if test_lower in lower_to_actual:
                    return os.path.join(search_path, lower_to_actual[test_lower])

    return None


def load_image_bytes(tex_path: str) -> Optional[bytes]:
    """Load an image file, flip it vertically, and convert to PNG bytes."""
    try:
        if not os.path.exists(tex_path):
            return None
        image_data = np.flipud(imageio.imread(tex_path))
        byte_arr = io.BytesIO()
        imageio.imwrite(byte_arr, image_data, format='png')
        return byte_arr.getvalue()
    except Exception as e:
        print(f"  Warning: Failed to load texture '{tex_path}': {e}")
        return None


def _add_accessor(gltf, blob, data, component_type, accessor_type, target=None):
    """Add an accessor and buffer view to the GLTF, ensuring proper alignment."""
    if data is None or data.size == 0:
        return -1

    byte_offset = len(blob)
    padding = (4 - (byte_offset % 4)) % 4
    blob.extend(b'\x00' * padding)
    byte_offset += padding

    data_bytes = data.tobytes()
    bv = BufferView(buffer=0, byteOffset=byte_offset, byteLength=len(data_bytes), target=target)
    gltf.bufferViews.append(bv)
    bv_idx = len(gltf.bufferViews) - 1

    accessor = Accessor(
        bufferView=bv_idx,
        componentType=component_type,
        count=len(data),
        type=accessor_type
    )

    if accessor_type in [VEC2, VEC3, VEC4]:
        accessor.min = data.min(axis=0).tolist()
        accessor.max = data.max(axis=0).tolist()
    elif accessor_type == SCALAR:
        if component_type == FLOAT:
            accessor.min = [float(data.min())]
            accessor.max = [float(data.max())]
        else:
            accessor.min = [int(data.min())]
            accessor.max = [int(data.max())]

    gltf.accessors.append(accessor)
    blob.extend(data_bytes)
    return len(gltf.accessors) - 1


def export_to_gltf(xac_path: str, output_gltf_path: str, texture_search_paths: List[str],
                   xsm_paths: Optional[List[str]] = None):
    """Export an XAC model and optional XSM animations to GLTF 2.0 format (GLB binary)."""
    meshes, skeleton_data, _ = extract_renderable_data(xac_path)

    if not meshes:
        raise ValueError(f"No renderable meshes found in {xac_path}")

    # Check if any mesh has actual skinning data - if not, this is a billboard/static model
    has_skinned_meshes = any(m.skinning_data is not None for m in meshes if not m.is_collision)
    if not has_skinned_meshes:
        skeleton_data = None  # Treat as static geometry

    gltf = GLTF2(asset={"version": "2.0", "generator": "XacToGlb-Converter"})
    blob = bytearray()
    gltf.buffers = [Buffer(byteLength=0)]
    gltf.samplers = [Sampler()]
    sampler_idx = 0

    # Texture/material embedding
    material_map = {}
    for mesh in meshes:
        if mesh.is_collision:
            continue
        for sub in mesh.sub_meshes:
            mat_name = sub['material_name']
            tex_name = sub['texture_name']
            if mat_name in material_map:
                continue

            mat_obj = Material(name=mat_name, doubleSided=True)
            mat_obj.pbrMetallicRoughness = PbrMetallicRoughness(
                baseColorFactor=[1, 1, 1, 1],
                metallicFactor=0.0,
                roughnessFactor=0.9
            )
            mat_obj.alphaMode = "OPAQUE"
            mat_obj.alphaCutoff = None

            if tex_name:
                tex_path = find_texture_path(tex_name, texture_search_paths)
                image_bytes = load_image_bytes(tex_path) if tex_path else None
                if image_bytes:
                    img_offset = len(blob)
                    blob.extend(image_bytes)
                    bv = BufferView(buffer=0, byteOffset=img_offset, byteLength=len(image_bytes))
                    gltf.bufferViews.append(bv)
                    img = GLTFImage(bufferView=len(gltf.bufferViews) - 1, mimeType="image/png")
                    gltf.images.append(img)
                    tex_obj = Texture(sampler=sampler_idx, source=len(gltf.images) - 1)
                    gltf.textures.append(tex_obj)
                    mat_obj.pbrMetallicRoughness.baseColorTexture = TextureInfo(index=len(gltf.textures) - 1)
                    mat_obj.alphaMode = "MASK"
                    mat_obj.alphaCutoff = 0.1

            gltf.materials.append(mat_obj)
            material_map[mat_name] = len(gltf.materials) - 1

    # Process Skeleton
    node_map, skeleton_root_indices = {}, []
    final_skeleton_roots = []
    if skeleton_data:
        for i, node in enumerate(skeleton_data.nodes):
            # Apply swizzle directly to TRS values (don't use matrix decomposition)
            # This preserves the exact values for round-trip conversion
            pos_x, pos_y, pos_z = node.local_pos
            swizzled_pos = [-pos_x, pos_z, pos_y]

            quat_x, quat_y, quat_z, quat_w = node.local_quat
            swizzled_quat = [-quat_x, quat_z, quat_y, -quat_w]
            # Normalize quaternion
            q_len = (swizzled_quat[0]**2 + swizzled_quat[1]**2 + swizzled_quat[2]**2 + swizzled_quat[3]**2) ** 0.5
            if q_len > 0:
                swizzled_quat = [q / q_len for q in swizzled_quat]

            # Scale is not swizzled
            swizzled_scale = list(node.local_scale)

            gltf_node = Node(name=node.node_name, translation=swizzled_pos, rotation=swizzled_quat, scale=swizzled_scale)
            gltf.nodes.append(gltf_node)
            node_map[i] = len(gltf.nodes) - 1

        for i, node in enumerate(skeleton_data.nodes):
            p_idx = int(node.parent_index)
            if p_idx != 4294967295 and p_idx in node_map:
                parent_node = gltf.nodes[node_map[p_idx]]
                if not parent_node.children:
                    parent_node.children = []
                parent_node.children.append(node_map[i])
            else:
                skeleton_root_indices.append(node_map[i])

        skin_skeleton_prop_idx = None
        if len(skeleton_root_indices) > 1:
            common_root_node = Node(name="SkeletonRoot", children=skeleton_root_indices)
            gltf.nodes.append(common_root_node)
            common_root_idx = len(gltf.nodes) - 1
            final_skeleton_roots.append(common_root_idx)
            skin_skeleton_prop_idx = common_root_idx
        elif len(skeleton_root_indices) == 1:
            final_skeleton_roots = skeleton_root_indices
            skin_skeleton_prop_idx = skeleton_root_indices[0]

        # Clean inverse bind matrices for GLTF validation
        ibms = np.array(skeleton_data.inverse_bind_matrices, dtype=np.float32)
        ibms[:, :3, 3] = 0.0
        ibms[:, 3, 3] = 1.0

        ibm_accessor = _add_accessor(gltf, blob, ibms, FLOAT, MAT4)
        skin = Skin(
            inverseBindMatrices=ibm_accessor,
            joints=list(node_map.values()),
            skeleton=skin_skeleton_prop_idx
        )
        gltf.skins = [skin]

    # Process Meshes
    mesh_nodes = []
    for i, r_mesh in enumerate(meshes):
        if r_mesh.is_collision:
            continue

        positions = r_mesh.vertices.astype(np.float32)

        normals = None
        if r_mesh.normals is not None and r_mesh.normals.size > 0:
            normals = r_mesh.normals.copy().astype(np.float32)
            # For static/billboard meshes, flip normals to match flipped triangle winding
            if not skeleton_data:
                normals = -normals
            with np.errstate(divide='ignore', invalid='ignore'):
                norm_lengths = np.linalg.norm(normals, axis=1, keepdims=True)
                non_zero_mask = norm_lengths > 1e-6
                np.divide(normals, norm_lengths, out=normals, where=non_zero_mask)

        uvs = r_mesh.uvs.astype(np.float32) if r_mesh.uvs is not None and r_mesh.uvs.size > 0 else None

        pos_idx = _add_accessor(gltf, blob, positions, FLOAT, VEC3, ARRAY_BUFFER)
        norm_idx = _add_accessor(gltf, blob, normals, FLOAT, VEC3, ARRAY_BUFFER) if normals is not None else -1
        uv_idx = _add_accessor(gltf, blob, uvs, FLOAT, VEC2, ARRAY_BUFFER) if uvs is not None else -1

        joints_idx, weights_idx = -1, -1
        if r_mesh.skinning_data:
            joints = r_mesh.skinning_data.bone_ids.astype(np.uint16)
            weights = r_mesh.skinning_data.bone_weights.astype(np.float32)
            joints_idx = _add_accessor(gltf, blob, joints, UNSIGNED_SHORT, VEC4, ARRAY_BUFFER)
            weights_idx = _add_accessor(gltf, blob, weights, FLOAT, VEC4, ARRAY_BUFFER)

        primitives = []
        for sub in r_mesh.sub_meshes:
            indices = sub['indices'].copy()
            # For static/billboard meshes, flip triangle winding to fix face direction
            if not skeleton_data:
                indices = indices.reshape(-1, 3)
                indices = indices[:, [0, 2, 1]].flatten()
            idx_idx = _add_accessor(gltf, blob, indices, UNSIGNED_INT, SCALAR, ELEMENT_ARRAY_BUFFER)
            attrs = {"POSITION": pos_idx}
            if norm_idx != -1:
                attrs["NORMAL"] = norm_idx
            if uv_idx != -1:
                attrs["TEXCOORD_0"] = uv_idx
            if joints_idx != -1:
                attrs.update({"JOINTS_0": joints_idx, "WEIGHTS_0": weights_idx})
            primitives.append(Primitive(
                attributes=attrs,
                indices=idx_idx,
                material=material_map.get(sub['material_name'], 0)
            ))

        gltf_mesh = Mesh(primitives=primitives)
        gltf.meshes.append(gltf_mesh)
        mesh_node = Node(name=f"Mesh_{i}", mesh=len(gltf.meshes) - 1)
        if skeleton_data and gltf.skins:
            mesh_node.skin = 0
        gltf.nodes.append(mesh_node)
        mesh_nodes.append(len(gltf.nodes) - 1)

    # Process Animations
    if xsm_paths and skeleton_data:
        gltf.animations = []
        for xsm_path in xsm_paths:
            try:
                animation_data = XSMParser(xsm_path).parse()
            except Exception as e:
                print(f"  Warning: Failed to parse animation '{xsm_path}': {e}")
                continue

            animation_data.map_tracks_to_node_indices(skeleton_data.nodes)
            anim_samplers, anim_channels = [], []

            # Iterate through all skeleton nodes, not just tracks in the file
            for node_idx, skel_node in enumerate(skeleton_data.nodes):
                gltf_node_idx = node_map.get(node_idx)
                if gltf_node_idx is None:
                    continue

                track = animation_data.tracks_by_index.get(node_idx)

                for path in ["translation", "rotation", "scale"]:
                    keys = []
                    if track:
                        keys = getattr(track,
                                       f"{'pos' if path == 'translation' else 'rot' if path == 'rotation' else 'scale'}_keys",
                                       [])

                    if not keys:
                        anim_duration = animation_data.duration if animation_data.duration > 0 else 1.0
                        if path == "translation":
                            default_value = pyrr.Vector3(skel_node.local_pos)
                        elif path == "rotation":
                            default_value = pyrr.Quaternion(skel_node.local_quat)
                        else:
                            default_value = pyrr.Vector3(skel_node.local_scale)
                        keys = [(0.0, default_value), (anim_duration, default_value)]

                    times = np.array([t for t, _ in keys], dtype=np.float32)
                    raw_values = [v for _, v in keys]

                    if path == "translation":
                        values = np.array([[-v[0], v[2], v[1]] for v in raw_values], dtype=np.float32)
                    elif path == "rotation":
                        swizzled_list = []
                        for v in raw_values:
                            q = pyrr.Quaternion([-v[0], v[2], v[1], v[3]])
                            if pyrr.vector.length(q) > 0:
                                q = pyrr.quaternion.normalize(q)
                            swizzled_list.append(q.tolist())
                        values = np.array(swizzled_list, dtype=np.float32)
                    else:
                        values = np.array([v.tolist() if hasattr(v, 'tolist') else list(v) for v in raw_values], dtype=np.float32)

                    if times.size == 0 or values.size == 0:
                        continue
                    time_acc = _add_accessor(gltf, blob, times, FLOAT, SCALAR)
                    val_acc = _add_accessor(gltf, blob, values, FLOAT, VEC3 if path != "rotation" else VEC4)
                    if time_acc == -1 or val_acc == -1:
                        continue

                    sampler_idx_in_anim = len(anim_samplers)
                    anim_samplers.append(AnimationSampler(input=time_acc, output=val_acc, interpolation="LINEAR"))
                    anim_channels.append(AnimationChannel(
                        sampler=sampler_idx_in_anim,
                        target=AnimationChannelTarget(node=gltf_node_idx, path=path)
                    ))

            if anim_channels:
                anim_name = os.path.splitext(os.path.basename(xsm_path))[0]
                gltf.animations.append(Animation(name=anim_name, channels=anim_channels, samplers=anim_samplers))

    # Create a master root node to orient the model correctly
    model_root_node = Node(name="ModelRoot")
    if skeleton_data:
        # For skinned models, apply -90 deg X rotation
        root_rotation_matrix = pyrr.matrix44.create_from_x_rotation(math.radians(-90.0))
        _, r, _ = pyrr.matrix44.decompose(root_rotation_matrix)
        model_root_node.rotation = r.tolist()
        model_root_node.children = mesh_nodes + final_skeleton_roots
    else:
        # For static/billboard models, apply -90 deg Z rotation
        root_rotation_matrix = pyrr.matrix44.create_from_z_rotation(math.radians(-90.0))
        _, r, _ = pyrr.matrix44.decompose(root_rotation_matrix)
        model_root_node.rotation = r.tolist()
        model_root_node.children = mesh_nodes
    gltf.nodes.append(model_root_node)
    model_root_node_idx = len(gltf.nodes) - 1

    gltf.scenes = [Scene(nodes=[model_root_node_idx])]
    gltf.scene = 0
    gltf.buffers[0].byteLength = len(blob)

    # Save as GLB
    gltf.set_binary_blob(blob)
    gltf.save_binary(output_gltf_path)
