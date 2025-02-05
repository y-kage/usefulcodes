# A simple script that uses blender to render views of a single object
# by rotation the camera around it.
# Also produces depth map at the same time.

import json
import os
from math import radians

import bpy
import numpy as np

DEBUG = False

RESULTS_PATH = "output/test_1"
COLOR_DEPTH = 8
FORMAT = "PNG"
RESOLUTION_X = 8192
RESOLUTION_Y = 8192
RESOLUTION_PERCENTAGE = 100
CAMERA_TYPE = "PERSP"  # 'PANO', 'ORTHO' or 'PERSP'
CAMERA_NAME = None

DEPTH_IN_BLENDER_MIN = 0.0
DEPTH_IN_BLENDER_MAX = 1000.0
DEPTH_OUTPUT_MIN = 1.0  # white
DEPTH_OUTPUT_MAX = 0.0  # black
BACKGROUND_TRANSPARENT = False
FILE_DIGITS = 6

# If USE_ANIMATION is True, VIEWS and MAX_ANGLE will be ignored
USE_ANIMATION = False
if USE_ANIMATION is False:
    VIEWS = 5  # 4000
    MAX_ANGLE = 360.0
    DYNAMIC = False
    UPPER_VIEWS = True
    CIRCLE_FIXED_START = (0, 0, 0)
    CIRCLE_FIXED_END = (0.7, 0, 0)

max_cache_frame = 1000
skip_to = 0

# Automatically select the camera when CAMERA_NAME is None
if CAMERA_NAME is None:
    cameras = [obj for obj in bpy.data.objects if obj.type == "CAMERA"]
    if len(cameras) == 1:
        cam = cameras[0]
        CAMERA_NAME = cam.name
    elif len(cameras) > 1:
        raise ValueError(
            f"Multiple cameras found ({[cam.name for cam in cameras]}). "
            "Please manually set CAMERA_NAME."
        )
    else:
        raise ValueError("No camera found in the scene. Please add a camera.")
else:
    cam = bpy.data.objects.get(CAMERA_NAME)
    if cam is None or cam.type != "CAMERA":
        raise ValueError(f"Camera with name '{CAMERA_NAME}' not found or is not a valid camera.")

fp = bpy.path.abspath(f"//{RESULTS_PATH}")


def listify_matrix(matrix):
    matrix_list = []
    for row in matrix:
        matrix_list.append(list(row))
    return matrix_list


if not os.path.exists(fp):
    os.makedirs(fp)

# Data to store in JSON file
out_data = {
    "camera_angle_x": bpy.data.objects[CAMERA_NAME].data.angle_x,
}

# Render Optimizations
bpy.context.scene.render.use_persistent_data = True


# Set up rendering of depth map.
bpy.context.scene.use_nodes = True
tree = bpy.context.scene.node_tree
links = tree.links

# Add passes for additionally dumping albedo and normals.
try:
    layer = bpy.context.scene.view_layers["ViewLayer"]
except (AttributeError, KeyError):
    layer = bpy.context.scene.view_layers["RenderLayer"]

layer.use_pass_normal = True
bpy.context.scene.render.image_settings.file_format = str(FORMAT)
bpy.context.scene.render.image_settings.color_depth = str(COLOR_DEPTH)

bpy.context.view_layer.use_pass_z = True
bpy.context.view_layer.use_pass_normal = True

if "Custom Outputs" in tree.nodes:

    def remove_all_links(node, previous_node=None):
        for output in node.outputs:
            if output.is_linked:
                for link in output.links:
                    if link.to_node != previous_node:
                        remove_all_links(link.to_node, previous_node=node)

        for input in node.inputs:
            if input.is_linked:
                for link in input.links:
                    if link.from_node != previous_node:
                        remove_all_links(link.from_node, previous_node=node)
        tree.nodes.remove(node)

    remove_all_links(tree.nodes["Custom Outputs"])

# Create input render layer node.
render_layers = tree.nodes.new("CompositorNodeRLayers")
render_layers.label = "Custom Outputs"
render_layers.name = "Custom Outputs"

depth_file_output = tree.nodes.new(type="CompositorNodeOutputFile")
depth_file_output.label = "Depth Output"
depth_file_output.name = "Depth Output"
if FORMAT == "OPEN_EXR":
    links.new(render_layers.outputs["Depth"], depth_file_output.inputs[0])
else:
    # Remap as other types can not represent the full range of depth.
    map = tree.nodes.new(type="CompositorNodeMapRange")
    # Size is chosen kind of arbitrarily,
    # try out until you're satisfied with resulting depth map.
    map.inputs["From Min"].default_value = DEPTH_IN_BLENDER_MIN
    map.inputs["From Max"].default_value = DEPTH_IN_BLENDER_MAX
    map.inputs["To Min"].default_value = DEPTH_OUTPUT_MIN
    map.inputs["To Max"].default_value = DEPTH_OUTPUT_MAX
    links.new(render_layers.outputs["Depth"], map.inputs[0])

    links.new(map.outputs[0], depth_file_output.inputs[0])

normal_file_output = tree.nodes.new(type="CompositorNodeOutputFile")
normal_file_output.label = "Normal Output"
normal_file_output.name = "Normal Output"
links.new(render_layers.outputs["Normal"], normal_file_output.inputs[0])

# Background
bpy.context.scene.render.dither_intensity = 0.0
bpy.context.scene.render.film_transparent = BACKGROUND_TRANSPARENT

# Create collection for objects not to render with background
for ob in bpy.context.scene.objects:
    if ob.type in ("EMPTY") and "Empty" in ob.name:
        bpy.data.objects.remove(ob)


def parent_obj_to_camera(b_camera):
    origin = (0, 0, 0)
    b_empty = bpy.data.objects.new("Empty", None)
    b_empty.location = origin
    b_camera.parent = b_empty  # setup parenting

    scn = bpy.context.scene
    scn.collection.objects.link(b_empty)
    bpy.context.view_layer.objects.active = b_empty
    # scn.objects.active = b_empty
    return b_empty


scene = bpy.context.scene
scene.render.resolution_x = RESOLUTION_X
scene.render.resolution_y = RESOLUTION_Y
scene.render.resolution_percentage = RESOLUTION_PERCENTAGE
scene.render.use_motion_blur = False
# scene.render.use_motion_blur = True
# scene.render.motion_blur_shutter = 0.5

# for animation
# frame_num = 0
# bpy.context.scene.frame_set(frame_num)
# b_empty.keyframe_insert(data_path = "rotation_euler", index = -1)
# obj.keyframe_insert(data_path = "location", index = -1)

# 10.0 # low position
# bpy.ops.wm.save_mainfile()
# obj.location.z
# TechnicPulleyLarge.001
# TechnicPulleyLarge.002
# PinSmoothWithoutFrictionRidges_01*02.005
# PinSmoothWithoutFrictionRidges_01*02.006

cam = scene.objects[CAMERA_NAME]
cam.data.type = CAMERA_TYPE
if cam.data.type == "PANO":
    cam.data.panorama_type = "EQUIRECTANGULAR"

if USE_ANIMATION is False:
    cam.location = (0, 4.0, 0.5)
    cam_constraint = cam.constraints.new(type="TRACK_TO")
    cam_constraint.track_axis = "TRACK_NEGATIVE_Z"
    cam_constraint.up_axis = "UP_Y"
    b_empty = parent_obj_to_camera(cam)
    cam_constraint.target = b_empty

scene.render.image_settings.file_format = "PNG"  # set output format to .png

# for output_node in [tree.nodes["Depth Output"], tree.nodes["Normal Output"]]:
#     output_node.base_path = fp
tree.nodes["Depth Output"].base_path = os.path.join(fp, "depth")
tree.nodes["Normal Output"].base_path = os.path.join(fp, "normal")

out_data["frames"] = []


if USE_ANIMATION:
    print("Using animation")
else:
    print("Not using animation")

    stepsize = MAX_ANGLE / VIEWS
    vertical_diff = CIRCLE_FIXED_END[0] - CIRCLE_FIXED_START[0]
    rotation_mode = "XYZ"

    b_empty.rotation_euler = CIRCLE_FIXED_START
    b_empty.rotation_euler[0] = CIRCLE_FIXED_START[0] + vertical_diff

    if DYNAMIC:
        # bpy.ops.wm.open_mainfile(filepath=blendfilepath)
        obj = bpy.data.objects.get("Controlpanel_Arm")
        obj.location.z = 4
        # precalculate the locations
        z_max = 10.0
        # interpolate ths distance
        z_loc = np.linspace(
            start=obj.location.z, stop=z_max, num=VIEWS // 4, endpoint=False
        )
        z_loc_flip = np.flip(z_loc)
        z_loc_full = np.concatenate([z_loc, z_loc_flip, z_loc, z_loc_flip], axis=0)

json_num = 0
START = 0
if USE_ANIMATION is True:
    START = scene.frame_start
    VIEWS = scene.frame_end + 1
    _file_format = ""
    for i in range(FILE_DIGITS):
        _file_format += "#"
    tree.nodes["Depth Output"].file_slots[0].path = _file_format
    tree.nodes["Normal Output"].file_slots[0].path = _file_format

for i in range(START, VIEWS):
    if USE_ANIMATION is True:
        scene.frame_set(i)
    elif i > 0:
        b_empty.rotation_euler[0] = (
            CIRCLE_FIXED_START[0]
            + (np.cos(radians(stepsize * i)) + 1) / 2 * vertical_diff
        )
        b_empty.rotation_euler[2] += radians(2 * stepsize)
        if DYNAMIC:
            obj.location.z = z_loc_full[i]

    if DEBUG:
        i = np.random.randint(0, VIEWS)
        b_empty.rotation_euler[0] = (
            CIRCLE_FIXED_START[0]
            + (np.cos(radians(stepsize * i)) + 1) / 2 * vertical_diff
        )
        b_empty.rotation_euler[2] += radians(2 * stepsize * i)

    file_name = str(i).zfill(FILE_DIGITS)
    relative_filepath = f"rgb/{file_name}"

    scene.render.filepath = os.path.join(fp, relative_filepath)
    if USE_ANIMATION is False:
        tree.nodes["Depth Output"].file_slots[0].path = file_name
        tree.nodes["Normal Output"].file_slots[0].path = file_name

    if DEBUG:
        break
    elif i >= skip_to:
        bpy.ops.render.render(write_still=True)

    frame_data = {
        "file_path": relative_filepath,
        # "depth_path": depth_path,
        # "normal_path": normal_path,
        # "rotation": radians(stepsize * i),
        # "time": i ,
        "transform_matrix": listify_matrix(cam.matrix_world),
    }

    out_data["frames"].append(frame_data)

    if (i + 1) % max_cache_frame == 0:
        json_num = ((i + 1) // max_cache_frame) - 1
        if not DEBUG:
            with open(fp + "/" + f"transforms_{json_num}.json", "w") as out_file:
                json.dump(out_data, out_file, indent=4)
            out_data["frames"] = []


if not DEBUG:
    json_num = VIEWS // max_cache_frame
    if len(out_data["frames"]) > 0:
        with open(fp + "/" + f"transforms_{json_num}.json", "w") as out_file:
            json.dump(out_data, out_file, indent=4)
        out_data["frames"] = []
        json_num = json_num + 1
    for i in range(json_num):
        with open(fp + "/" + f"transforms_{i}.json", "r") as out_file:
            d = json.load(out_file)
            for f_data in d["frames"]:
                out_data["frames"].append(f_data)
    with open(fp + "/" + "transforms.json", "w") as out_file:
        json.dump(out_data, out_file, indent=4)
