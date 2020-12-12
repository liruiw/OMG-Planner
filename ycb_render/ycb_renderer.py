# --------------------------------------------------------
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
 
import sys
import ctypes

from pprint import pprint
from PIL import Image
import glutils.glcontext as glcontext
import OpenGL.GL as GL
import cv2
import numpy as np
from pyassimp import *
from glutils.meshutil import *
from glutils.voxel import *
from transforms3d.quaternions import axangle2quat, mat2quat, qmult, qinverse
from transforms3d.euler import quat2euler, mat2euler, euler2quat, euler2mat
import CppYCBRenderer
from numpy.linalg import inv, norm
import numpy.random as npr
import IPython
import subprocess
import multiprocessing

import threading
import platform

PYTHON2 = True
if platform.python_version().startswith("3"):
    PYTHON2 = False
try:
    from .get_available_devices import *
except:
    from get_available_devices import *
import scipy.io as sio

MAX_NUM_OBJECTS = 3
from glutils.utils import *
from glutils.trackball import Trackball
import time
import random


def load_mesh_single(param):
    """
    Load a single mesh. Used in multiprocessing.
    """
    path, scale, offset_map = param
    scene = load(path)
    mesh_file = path.strip().split("/")[-1]  # for offset the robot mesh
    offset = np.zeros(3)
    if offset_map and mesh_file in offset_map:
        offset = offset_map[mesh_file]
    return recursive_load(scene.rootnode, [], [], [], [], offset, scale, [[], [], []])


def load_texture_single(param):
    obj_path, texture_paths = param
    textures, is_colors, is_textures = [], [], []
    for texture_path in texture_paths:
        # print('obj_path:', obj_path, texture_path)
        is_texture = False
        is_color = False
        if texture_path == "":
            texture = texture_path
        elif texture_path == "color":
            is_color = True
            texture = texture_path
        else:
            texture_path = os.path.join(
                "/".join(obj_path.split("/")[:-1]), texture_path
            )
            texture = loadTexture2(texture_path)
            is_texture = True
        textures.append(texture)
        is_colors.append(is_color)
        is_textures.append(is_texture)
    return [textures, is_colors, is_textures]


def recursive_load(
    node,
    vertices,
    faces,
    materials,
    texture_paths,
    offset,
    scale=1,
    repeated=[[], [], []],
):
    """
    Applying transform to recursively load vertices, normals, faces
    """
    if node.meshes:
        transform = node.transformation
        for idx, mesh in enumerate(node.meshes):
            if mesh.faces.shape[-1] != 3:  # ignore Line Set
                continue
            mat = mesh.material
            texture_path = False
            if hasattr(mat, "properties"):
                file = ("file", long(1)) if PYTHON2 else ("file", 1)
                if file in mat.properties:
                    texture_paths.append(mat.properties[file])  # .encode("utf-8")
                    texture_path = True
                else:
                    texture_paths.append("")
            mat_diffuse = np.array(mat.properties["diffuse"])[:3]
            mat_specular = np.array(mat.properties["specular"])[:3]
            mat_ambient = np.array(mat.properties["ambient"])[:3]  # phong shader
            if "shininess" in mat.properties:
                mat_shininess = max(
                    mat.properties["shininess"], 1
                )  # avoid the 0 shininess
            else:
                mat_shininess = 1

            mesh_vertex = (
                homotrans(transform, mesh.vertices) - offset
            )  # subtract the offset
            if mesh.normals.shape[0] > 0:
                mesh_normals = (
                    transform[:3, :3].dot(mesh.normals.transpose()).transpose()
                )  # normal stays the same
            else:
                mesh_normals = np.zeros_like(mesh_vertex)
                mesh_normals[:, -1] = 1
            if texture_path:
                vertices.append(
                    np.concatenate(
                        [
                            mesh_vertex * scale,
                            mesh_normals,
                            mesh.texturecoords[0, :, :2],
                        ],
                        axis=-1,
                    )
                )
            elif mesh.colors is not None and len(mesh.colors.shape) > 2:
                vertices.append(
                    np.concatenate(
                        [mesh_vertex * scale, mesh_normals, mesh.colors[0, :, :3]],
                        axis=-1,
                    )
                )  #
                texture_paths[-1] = "color"
            else:
                vertices.append(
                    np.concatenate([mesh_vertex * scale, mesh_normals], axis=-1)
                )
            faces.append(mesh.faces)
            materials.append(
                np.hstack([mat_diffuse, mat_specular, mat_ambient, mat_shininess])
            )
    for child in node.children:
        recursive_load(
            child, vertices, faces, materials, texture_paths, offset, scale, repeated
        )
    return vertices, faces, materials, texture_paths


def loadTexture2(path):
    """
    Load texture file
    """

    img = Image.open(path).transpose(Image.FLIP_TOP_BOTTOM)
    img_data = np.fromstring(img.tobytes(), np.uint8)
    width, height = img.size
    return img, img_data


def loadTexture(path):
    """
    Load texture file
    """
    img = Image.open(path).transpose(Image.FLIP_TOP_BOTTOM)
    img_data = np.fromstring(img.tobytes(), np.uint8)
    width, height = img.size

    texture = GL.glGenTextures(1)
    GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 1)
    GL.glBindTexture(GL.GL_TEXTURE_2D, texture)
    GL.glTexParameterf(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
    GL.glTexParameterf(
        GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR_MIPMAP_LINEAR
    )
    GL.glTexParameterf(
        GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_REPEAT
    )  # .GL_CLAMP_TO_EDGE GL_REPEAT
    GL.glTexParameterf(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_REPEAT)
    if img.mode == "RGBA":
        GL.glTexImage2D(
            GL.GL_TEXTURE_2D,
            0,
            GL.GL_RGBA,
            width,
            height,
            0,
            GL.GL_RGBA,
            GL.GL_UNSIGNED_BYTE,
            img_data,
        )
    else:
        GL.glTexImage2D(
            GL.GL_TEXTURE_2D,
            0,
            GL.GL_RGB,
            width,
            height,
            0,
            GL.GL_RGB,
            GL.GL_UNSIGNED_BYTE,
            img_data,
        )
    GL.glGenerateMipmap(GL.GL_TEXTURE_2D)
    return texture


def bindTexture(imgs):
    """
    Load texture file
    """
    all_textures = []

    for img in imgs:
        textures = []
        for item in img:
            if len(item) < 2:
                textures.append([])
                continue
            (img, img_data) = item
            width, height = img.size

            texture = GL.glGenTextures(1)
            GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 1)
            GL.glBindTexture(GL.GL_TEXTURE_2D, texture)
            GL.glTexParameterf(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
            GL.glTexParameterf(
                GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR_MIPMAP_LINEAR
            )
            GL.glTexParameterf(
                GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_REPEAT
            )  # .GL_CLAMP_TO_EDGE GL_REPEAT
            GL.glTexParameterf(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_REPEAT)
            if img.mode == "RGBA":
                GL.glTexImage2D(
                    GL.GL_TEXTURE_2D,
                    0,
                    GL.GL_RGBA,
                    width,
                    height,
                    0,
                    GL.GL_RGBA,
                    GL.GL_UNSIGNED_BYTE,
                    img_data,
                )
            else:
                GL.glTexImage2D(
                    GL.GL_TEXTURE_2D,
                    0,
                    GL.GL_RGB,
                    width,
                    height,
                    0,
                    GL.GL_RGB,
                    GL.GL_UNSIGNED_BYTE,
                    img_data,
                )
            GL.glGenerateMipmap(GL.GL_TEXTURE_2D)
            textures.append(texture)
        all_textures.append(textures)
    return all_textures


class YCBRenderer:
    def __init__(
        self,
        width=512,
        height=512,
        gpu_id=0,
        render_marker=False,
        robot="panda_arm",
        offset=True,
        reinit=False,
    ):
        self.render_marker = render_marker
        self.VAOs = []
        self.VBOs = []
        self.materials = []
        self.textures = []
        self.is_textured = []
        self.is_materialed = []
        self.is_colored = []
        self.objects = []
        self.texUnitUniform = None
        self.width = width
        self.height = height
        self.faces = []
        self.poses_trans = []
        self.poses_rot = []
        self.instances = []
        self.parallel_textures = False
        self.parallel_load_mesh = False

        self.robot = robot
        self._offset_map = None
        if robot == "panda_arm" or robot == "baxter" and offset:
            self._offset_map = self.load_offset()
        if gpu_id == -1:
            from gibson2.core.render.mesh_renderer import CppMeshRenderer

            self.r = CppMeshRenderer.CppMeshRenderer(width, height, 0)
        else:
            self.r = CppYCBRenderer.CppYCBRenderer(
                width, height, get_available_devices()[gpu_id]
            )
        self.r.init()
        self.glstring = GL.glGetString(GL.GL_VERSION)
        from OpenGL.GL import shaders

        self.shaders = shaders
        self.colors = []
        self.lightcolor = [1, 1, 1]
        self.worldlight = [[0.2, 0, 0.2], [0.2, 0.2, 0], [0, 0.5, 1], [0.5, 0, 1]]
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        vertexShader = self.shaders.compileShader(
            open(os.path.join(cur_dir, "shaders/vert.shader")).readlines(),
            GL.GL_VERTEX_SHADER,
        )

        fragmentShader = self.shaders.compileShader(
            open(os.path.join(cur_dir, "shaders/frag.shader")).readlines(),
            GL.GL_FRAGMENT_SHADER,
        )

        vertexShader_textureMat = self.shaders.compileShader(
            open(os.path.join(cur_dir, "shaders/vert_blinnphong.shader")).readlines(),
            GL.GL_VERTEX_SHADER,
        )

        fragmentShader_textureMat = self.shaders.compileShader(
            open(os.path.join(cur_dir, "shaders/frag_blinnphong.shader")).readlines(),
            GL.GL_FRAGMENT_SHADER,
        )

        vertexShader_textureless = self.shaders.compileShader(
            open(os.path.join(cur_dir, "shaders/vert_textureless.shader")).readlines(),
            GL.GL_VERTEX_SHADER,
        )

        fragmentShader_textureless = self.shaders.compileShader(
            open(os.path.join(cur_dir, "shaders/frag_textureless.shader")).readlines(),
            GL.GL_FRAGMENT_SHADER,
        )

        vertexShader_material = self.shaders.compileShader(
            open(os.path.join(cur_dir, "shaders/vert_mat.shader")).readlines(),
            GL.GL_VERTEX_SHADER,
        )

        fragmentShader_material = self.shaders.compileShader(
            open(os.path.join(cur_dir, "shaders/frag_mat.shader")).readlines(),
            GL.GL_FRAGMENT_SHADER,
        )

        vertexShader_simple = self.shaders.compileShader(
            open(os.path.join(cur_dir, "shaders/vert_simple.shader")).readlines(),
            GL.GL_VERTEX_SHADER,
        )

        fragmentShader_simple = self.shaders.compileShader(
            open(os.path.join(cur_dir, "shaders/frag_simple.shader")).readlines(),
            GL.GL_FRAGMENT_SHADER,
        )

        vertexShader_simple_color = self.shaders.compileShader(
            open(os.path.join(cur_dir, "shaders/vert_simple_color.shader")).readlines(),
            GL.GL_VERTEX_SHADER,
        )

        fragmentShader_simple_color = self.shaders.compileShader(
            open(os.path.join(cur_dir, "shaders/frag_simple_color.shader")).readlines(),
            GL.GL_FRAGMENT_SHADER,
        )

        self.shaderProgram = self.shaders.compileProgram(vertexShader, fragmentShader)
        self.shaderProgram_textureless = self.shaders.compileProgram(
            vertexShader_textureless, fragmentShader_textureless
        )
        self.shaderProgram_simple = self.shaders.compileProgram(
            vertexShader_simple, fragmentShader_simple
        )
        self.shaderProgram_simple_color = self.shaders.compileProgram(
            vertexShader_simple_color, fragmentShader_simple_color
        )

        self.shaderProgram_material = self.shaders.compileProgram(
            vertexShader_material, fragmentShader_material
        )
        self.shaderProgram_textureMat = self.shaders.compileProgram(
            vertexShader_textureMat, fragmentShader_textureMat
        )

        vertexShader_textureMatNormal = self.shaders.compileShader(
            open(
                os.path.join(cur_dir, "shaders/vert_blinnphong_normal.shader")
            ).readlines(),
            GL.GL_VERTEX_SHADER,
        )

        geometryShader_textureMatNormal = self.shaders.compileShader(
            open(
                os.path.join(cur_dir, "shaders/geo_blinnphong_normal.shader")
            ).readlines(),
            GL.GL_GEOMETRY_SHADER,
        )

        fragmentShader_textureMatNormal = self.shaders.compileShader(
            open(
                os.path.join(cur_dir, "shaders/frag_blinnphong_normal.shader")
            ).readlines(),
            GL.GL_FRAGMENT_SHADER,
        )
        self.shaderProgram_textureMatNormal = self.shaders.compileProgram(
            vertexShader_textureMatNormal,
            geometryShader_textureMatNormal,
            fragmentShader_textureMatNormal,
        )

        self.texUnitUniform_textureMat = GL.glGetUniformLocation(
            self.shaderProgram_textureMat, "texUnit"
        )

        self.bind_texture_buffer()
        self.lightpos = [0, 0, 0]
        self.fov = 20
        self.camera = [1, 0, 0]
        self.target = [0, 0, 0]
        self.up = [0, 0, 1]
        self.lineVAOs = []
        self.pointVAOs = []
        self.coordVAOs = []
        P = perspective(self.fov, float(self.width) / float(self.height), 0.01, 100)
        V = lookat(self.camera, self.target, up=self.up)

        self.V = np.ascontiguousarray(V, np.float32)
        self.P = np.ascontiguousarray(P, np.float32)
        self.grid = self.generate_grid()

    def generate_grid(self):
        """
        Generate a grid as the plane background
        """
        VAO = GL.glGenVertexArrays(1)
        GL.glBindVertexArray(VAO)

        vertexData = []
        for i in np.arange(-1, 1, 0.05):
            vertexData.append([i, 0, -1, 0, 0, 0, 0, 0])
            vertexData.append([i, 0, 1, 0, 0, 0, 0, 0])
            vertexData.append([1, 0, i, 0, 0, 0, 0, 0])
            vertexData.append([-1, 0, i, 0, 0, 0, 0, 0])

        vertexData = np.array(vertexData).astype(np.float32) * 3
        # Need VBO for triangle vertices and texture UV coordinates
        VBO = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, VBO)
        GL.glBufferData(
            GL.GL_ARRAY_BUFFER, vertexData.nbytes, vertexData, GL.GL_STATIC_DRAW
        )

        # enable array and set up data
        positionAttrib = GL.glGetAttribLocation(self.shaderProgram_simple, "position")
        GL.glEnableVertexAttribArray(0)
        GL.glVertexAttribPointer(positionAttrib, 3, GL.GL_FLOAT, GL.GL_FALSE, 32, None)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)
        GL.glBindVertexArray(0)
        return VAO

    def generate_coordinate_axis(self, thickness=5, length=0.2, coordVAOs=None):
        lines = []
        basis_line = [np.zeros([3, 3]), np.eye(3) * length]
        line_colors = [
            np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]])
        ] * self.get_num_objects()
        thicknesses = [thickness] * self.get_num_objects()

        # lines [3 x n, 3 x n]
        for i in range(len(self.poses_rot)):
            linea = (
                self.poses_rot[i][:3, :3].dot(basis_line[0])
                + self.poses_trans[i][[3], :3].T
            )
            lineb = (
                self.poses_rot[i][:3, :3].dot(basis_line[1])
                + self.poses_trans[i][[3], :3].T
            )
            lines.append([linea.copy(), lineb.copy()])

        return self.generate_lines((lines, line_colors, thicknesses, True))

    def generate_lines(self, line_info, lineVAOs=None):
        """
        Render lines with GL
        """
        if line_info is not None:
            lines, line_colors, thicknesses, create_new = line_info
            for idx, (line, line_color, thickness) in enumerate(
                zip(lines, line_colors, thicknesses)
            ):
                GL.glLineWidth(thickness)
                if create_new:
                    VAO = GL.glGenVertexArrays(1)
                    GL.glBindVertexArray(VAO)
                    vertexData = []
                    linea, lineb = line[0].T, line[1].T
                    line_num = len(lineb)
                    if type(line_color) is not np.ndarray:
                        line_color = (
                            np.tile(line_color, line_num).reshape(-1, 3) / 255.0
                        )  # [np.array(line_color[0])  / 255.] * line_num
                    else:
                        line_color = line_color / 255.0

                    for i in np.arange(line_num):
                        vertexData.append(
                            [
                                linea[i][0],
                                linea[i][1],
                                linea[i][2],
                                0,
                                0,
                                0,
                                line_color[i][0],
                                line_color[i][1],
                                line_color[i][2],
                            ]
                        )
                        vertexData.append(
                            [
                                lineb[i][0],
                                lineb[i][1],
                                lineb[i][2],
                                0,
                                0,
                                0,
                                line_color[i][0],
                                line_color[i][1],
                                line_color[i][2],
                            ]
                        )

                    vertexData = np.array(vertexData).astype(np.float32)
                    # Need VBO for triangle vertices and texture UV coordinates
                    VBO = GL.glGenBuffers(1)
                    GL.glBindBuffer(GL.GL_ARRAY_BUFFER, VBO)
                    GL.glBufferData(
                        GL.GL_ARRAY_BUFFER,
                        vertexData.nbytes,
                        vertexData,
                        GL.GL_STATIC_DRAW,
                    )

                    # enable array and set up data
                    positionAttrib = GL.glGetAttribLocation(
                        self.shaderProgram_simple_color, "position"
                    )
                    colorAttrib = GL.glGetAttribLocation(
                        self.shaderProgram_simple_color, "texCoords"
                    )
                    GL.glEnableVertexAttribArray(0)
                    GL.glVertexAttribPointer(
                        positionAttrib, 3, GL.GL_FLOAT, GL.GL_FALSE, 36, None
                    )
                    GL.glEnableVertexAttribArray(2)
                    GL.glVertexAttribPointer(
                        colorAttrib,
                        3,
                        GL.GL_FLOAT,
                        GL.GL_FALSE,
                        36,
                        ctypes.c_void_p(24),
                    )

                    GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)
                    GL.glBindVertexArray(0)
                    line_num = line[0].shape[1] * 2
                    if lineVAOs is not None:
                        lineVAOs.append((VAO, line_num))
                    GL.glUseProgram(self.shaderProgram_simple_color)
                    GL.glBindVertexArray(VAO)
                    GL.glUniformMatrix4fv(
                        GL.glGetUniformLocation(self.shaderProgram_simple_color, "V"),
                        1,
                        GL.GL_TRUE,
                        self.V,
                    )
                    GL.glUniformMatrix4fv(
                        GL.glGetUniformLocation(self.shaderProgram_simple_color, "P"),
                        1,
                        GL.GL_FALSE,
                        self.P,
                    )
                    GL.glDrawElements(
                        GL.GL_LINES,
                        line_num,
                        GL.GL_UNSIGNED_INT,
                        np.arange(line_num, dtype=np.int),
                    )
                    GL.glBindVertexArray(0)
                    GL.glUseProgram(0)

        if lineVAOs is not None:
            for idx, (VAO, line_num) in enumerate(lineVAOs):
                GL.glUseProgram(self.shaderProgram_simple_color)
                GL.glBindVertexArray(VAO)
                GL.glUniformMatrix4fv(
                    GL.glGetUniformLocation(self.shaderProgram_simple_color, "V"),
                    1,
                    GL.GL_TRUE,
                    self.V,
                )
                GL.glUniformMatrix4fv(
                    GL.glGetUniformLocation(self.shaderProgram_simple_color, "P"),
                    1,
                    GL.GL_FALSE,
                    self.P,
                )
                GL.glDrawElements(
                    GL.GL_LINES,
                    line_num,
                    GL.GL_UNSIGNED_INT,
                    np.arange(line_num, dtype=np.int),
                )
                GL.glBindVertexArray(0)
                GL.glUseProgram(0)
        GL.glLineWidth(1)
        return lineVAOs

    def generate_points(self, point_info, pointVAOs=None):
        """
        Render points with GL
        """
        if point_info is not None:
            points, points_colors, thicknesses, create_new = point_info

            for idx, (point, point_color, thickness) in enumerate(
                zip(points, points_colors, thicknesses)
            ):
                GL.glPointSize(thickness)
                if create_new:
                    
                    VAO = GL.glGenVertexArrays(1)
                    GL.glBindVertexArray(VAO)
                    vertexData = []
                    point = point.T
                    point_num = len(point)
                    if type(point_color) is not np.ndarray:
                        point_color = (
                            np.tile(point_color, point_num).reshape(-1, 3) / 255.0
                        )  # [np.array(line_color[0])  / 255.] * line_num
                    else:
                        point_color = point_color / 255.0

                    for i in np.arange(point_num):
                        vertexData.append(
                            [
                                point[i][0],
                                point[i][1],
                                point[i][2],
                                0,
                                0,
                                0,
                                point_color[i][0],
                                point_color[i][1],
                                point_color[i][2],
                            ]
                        )

                    vertexData = np.array(vertexData).astype(np.float32)
                    VBO = GL.glGenBuffers(1)
                    GL.glBindBuffer(GL.GL_ARRAY_BUFFER, VBO)
                    GL.glBufferData(
                        GL.GL_ARRAY_BUFFER,
                        vertexData.nbytes,
                        vertexData,
                        GL.GL_STATIC_DRAW,
                    )

                    # enable array and set up data
                    positionAttrib = GL.glGetAttribLocation(
                        self.shaderProgram_simple_color, "position"
                    )
                    colorAttrib = GL.glGetAttribLocation(
                        self.shaderProgram_simple_color, "texCoords"
                    )
                    GL.glEnableVertexAttribArray(0)
                    GL.glVertexAttribPointer(
                        positionAttrib, 3, GL.GL_FLOAT, GL.GL_FALSE, 36, None
                    )
                    GL.glEnableVertexAttribArray(2)
                    GL.glVertexAttribPointer(
                        colorAttrib,
                        3,
                        GL.GL_FLOAT,
                        GL.GL_FALSE,
                        36,
                        ctypes.c_void_p(24),
                    )

                    GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)
                    GL.glBindVertexArray(0)
                    point_num = point.shape[0]
                    if pointVAOs is not None:
                        pointVAOs.append((VAO, point_num))
                    GL.glUseProgram(self.shaderProgram_simple_color)
                    GL.glBindVertexArray(VAO)

                    GL.glUniformMatrix4fv(
                        GL.glGetUniformLocation(self.shaderProgram_simple_color, "V"),
                        1,
                        GL.GL_TRUE,
                        self.V,
                    )
                    GL.glUniformMatrix4fv(
                        GL.glGetUniformLocation(self.shaderProgram_simple_color, "P"),
                        1,
                        GL.GL_FALSE,
                        self.P,
                    )
                    GL.glDrawElements(
                        GL.GL_POINTS,
                        point_num,
                        GL.GL_UNSIGNED_INT,
                        np.arange(point_num, dtype=np.int),
                    )
                    GL.glBindVertexArray(0)
                    GL.glUseProgram(0)
        if pointVAOs is not None:
            for idx, (VAO, point_num) in enumerate(pointVAOs):
                GL.glUseProgram(self.shaderProgram_simple_color)
                GL.glBindVertexArray(VAO)

                GL.glUniformMatrix4fv(
                    GL.glGetUniformLocation(self.shaderProgram_simple_color, "V"),
                    1,
                    GL.GL_TRUE,
                    self.V,
                )
                GL.glUniformMatrix4fv(
                    GL.glGetUniformLocation(self.shaderProgram_simple_color, "P"),
                    1,
                    GL.GL_FALSE,
                    self.P,
                )
                GL.glDrawElements(
                    GL.GL_POINTS,
                    point_num,
                    GL.GL_UNSIGNED_INT,
                    np.arange(point_num, dtype=np.int),
                )
                GL.glBindVertexArray(0)
                GL.glUseProgram(0)

        GL.glPointSize(1)
        return pointVAOs

    def bind_texture_buffer(self):
        """
        bind texture buffer with GL
        """
        self.fbo = GL.glGenFramebuffers(1)
        self.color_tex = GL.glGenTextures(1)
        self.color_tex_2 = GL.glGenTextures(1)
        self.color_tex_3 = GL.glGenTextures(1)
        self.color_tex_4 = GL.glGenTextures(1)
        self.color_tex_5 = GL.glGenTextures(1)
        self.depth_tex = GL.glGenTextures(1)

        GL.glBindTexture(GL.GL_TEXTURE_2D, self.color_tex)
        GL.glTexImage2D(
            GL.GL_TEXTURE_2D,
            0,
            GL.GL_RGBA32F,
            self.width,
            self.height,
            0,
            GL.GL_RGBA,
            GL.GL_UNSIGNED_BYTE,
            None,
        )

        GL.glBindTexture(GL.GL_TEXTURE_2D, self.color_tex_2)
        GL.glTexImage2D(
            GL.GL_TEXTURE_2D,
            0,
            GL.GL_RGBA32F,
            self.width,
            self.height,
            0,
            GL.GL_RGBA,
            GL.GL_UNSIGNED_BYTE,
            None,
        )

        GL.glBindTexture(GL.GL_TEXTURE_2D, self.color_tex_3)
        GL.glTexImage2D(
            GL.GL_TEXTURE_2D,
            0,
            GL.GL_RGBA32F,
            self.width,
            self.height,
            0,
            GL.GL_RGBA,
            GL.GL_UNSIGNED_BYTE,
            None,
        )

        GL.glBindTexture(GL.GL_TEXTURE_2D, self.color_tex_4)
        GL.glTexImage2D(
            GL.GL_TEXTURE_2D,
            0,
            GL.GL_RGBA32F,
            self.width,
            self.height,
            0,
            GL.GL_RGBA,
            GL.GL_FLOAT,
            None,
        )

        GL.glBindTexture(GL.GL_TEXTURE_2D, self.color_tex_5)
        GL.glTexImage2D(
            GL.GL_TEXTURE_2D,
            0,
            GL.GL_RGBA32F,
            self.width,
            self.height,
            0,
            GL.GL_RGBA,
            GL.GL_FLOAT,
            None,
        )

        GL.glBindTexture(GL.GL_TEXTURE_2D, self.depth_tex)

        GL.glTexImage2D.wrappedOperation(
            GL.GL_TEXTURE_2D,
            0,
            GL.GL_DEPTH24_STENCIL8,
            self.width,
            self.height,
            0,
            GL.GL_DEPTH_STENCIL,
            GL.GL_UNSIGNED_INT_24_8,
            None,
        )

        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.fbo)
        GL.glFramebufferTexture2D(
            GL.GL_FRAMEBUFFER,
            GL.GL_COLOR_ATTACHMENT0,
            GL.GL_TEXTURE_2D,
            self.color_tex,
            0,
        )
        GL.glFramebufferTexture2D(
            GL.GL_FRAMEBUFFER,
            GL.GL_COLOR_ATTACHMENT1,
            GL.GL_TEXTURE_2D,
            self.color_tex_2,
            0,
        )
        GL.glFramebufferTexture2D(
            GL.GL_FRAMEBUFFER,
            GL.GL_COLOR_ATTACHMENT2,
            GL.GL_TEXTURE_2D,
            self.color_tex_3,
            0,
        )
        GL.glFramebufferTexture2D(
            GL.GL_FRAMEBUFFER,
            GL.GL_COLOR_ATTACHMENT3,
            GL.GL_TEXTURE_2D,
            self.color_tex_4,
            0,
        )
        GL.glFramebufferTexture2D(
            GL.GL_FRAMEBUFFER,
            GL.GL_COLOR_ATTACHMENT4,
            GL.GL_TEXTURE_2D,
            self.color_tex_5,
            0,
        )
        GL.glFramebufferTexture2D(
            GL.GL_FRAMEBUFFER,
            GL.GL_DEPTH_STENCIL_ATTACHMENT,
            GL.GL_TEXTURE_2D,
            self.depth_tex,
            0,
        )
        GL.glViewport(0, 0, self.width, self.height)
        GL.glDrawBuffers(
            5,
            [
                GL.GL_COLOR_ATTACHMENT0,
                GL.GL_COLOR_ATTACHMENT1,
                GL.GL_COLOR_ATTACHMENT2,
                GL.GL_COLOR_ATTACHMENT3,
                GL.GL_COLOR_ATTACHMENT4,
            ],
        )

        assert (
            GL.glCheckFramebufferStatus(GL.GL_FRAMEBUFFER) == GL.GL_FRAMEBUFFER_COMPLETE
        )

    def load_object(
        self, obj_path, texture_path, scene, textures_i, scale=1, data=None
    ):
        """
        load a single object and bind to buffer
        """
        is_texture = []
        is_materialed = True
        textures = []
        start_time = time.time()
        (
            vertices,
            faces,
            materials,
            texture_paths,
        ) = scene  # self.load_mesh(obj_path, scale, scene)

        self.materials.append(materials)
        is_textured = []
        is_colored = []

        if not self.parallel_textures:
            for texture_path in texture_paths:
                is_texture = False
                is_color = False
                if texture_path == "":
                    textures.append(texture_path)
                elif texture_path == "color":
                    is_color = True
                    textures.append(texture_path)
                else:
                    texture_path = os.path.join(
                        "/".join(obj_path.split("/")[:-1]), texture_path
                    )
                    texture = loadTexture(texture_path)
                    textures.append(texture)
                    is_texture = True
                is_textured.append(is_texture)
                is_colored.append(is_color)
        else:
            textures, is_colored, is_textured = textures_i

        self.textures.append(textures)  # textures
        self.is_textured.append(is_textured)  # is_textured
        self.is_materialed.append(is_materialed)

        if is_materialed:
            for idx in range(len(vertices)):
                vertexData = vertices[idx].astype(np.float32)
                face = faces[idx]
                VAO = GL.glGenVertexArrays(1)
                GL.glBindVertexArray(VAO)

                # Need VBO for triangle vertices and texture UV coordinates
                VBO = GL.glGenBuffers(1)
                GL.glBindBuffer(GL.GL_ARRAY_BUFFER, VBO)
                GL.glBufferData(
                    GL.GL_ARRAY_BUFFER, vertexData.nbytes, vertexData, GL.GL_STATIC_DRAW
                )
                if is_textured[idx]:
                    positionAttrib = GL.glGetAttribLocation(
                        self.shaderProgram_textureMat, "position"
                    )
                    normalAttrib = GL.glGetAttribLocation(
                        self.shaderProgram_textureMat, "normal"
                    )
                    coordsAttrib = GL.glGetAttribLocation(
                        self.shaderProgram_textureMat, "texCoords"
                    )
                elif is_colored[idx]:
                    positionAttrib = GL.glGetAttribLocation(
                        self.shaderProgram_textureless, "position"
                    )
                    normalAttrib = GL.glGetAttribLocation(
                        self.shaderProgram_textureless, "normal"
                    )
                    colorAttrib = GL.glGetAttribLocation(
                        self.shaderProgram_textureless, "color"
                    )
                else:
                    positionAttrib = GL.glGetAttribLocation(
                        self.shaderProgram_material, "position"
                    )
                    normalAttrib = GL.glGetAttribLocation(
                        self.shaderProgram_material, "normal"
                    )

                GL.glEnableVertexAttribArray(0)
                GL.glEnableVertexAttribArray(1)
                # the last parameter is a pointer

                if is_textured[idx]:
                    GL.glEnableVertexAttribArray(2)
                    GL.glVertexAttribPointer(
                        positionAttrib, 3, GL.GL_FLOAT, GL.GL_FALSE, 32, None
                    )
                    GL.glVertexAttribPointer(
                        normalAttrib,
                        3,
                        GL.GL_FLOAT,
                        GL.GL_FALSE,
                        32,
                        ctypes.c_void_p(12),
                    )
                    GL.glVertexAttribPointer(
                        coordsAttrib,
                        2,
                        GL.GL_FLOAT,
                        GL.GL_TRUE,
                        32,
                        ctypes.c_void_p(24),
                    )
                elif is_colored[idx]:
                    GL.glEnableVertexAttribArray(2)
                    GL.glVertexAttribPointer(
                        positionAttrib, 3, GL.GL_FLOAT, GL.GL_FALSE, 36, None
                    )
                    GL.glVertexAttribPointer(
                        normalAttrib,
                        3,
                        GL.GL_FLOAT,
                        GL.GL_FALSE,
                        36,
                        ctypes.c_void_p(12),
                    )
                    GL.glVertexAttribPointer(
                        colorAttrib,
                        3,
                        GL.GL_FLOAT,
                        GL.GL_FALSE,
                        36,
                        ctypes.c_void_p(24),
                    )
                else:
                    GL.glVertexAttribPointer(
                        positionAttrib, 3, GL.GL_FLOAT, GL.GL_FALSE, 24, None
                    )
                    GL.glVertexAttribPointer(
                        normalAttrib,
                        3,
                        GL.GL_FLOAT,
                        GL.GL_FALSE,
                        24,
                        ctypes.c_void_p(12),
                    )

                GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)
                GL.glBindVertexArray(0)
                self.VAOs.append(VAO)
                self.VBOs.append(VBO)
                self.faces.append(face)
            self.objects.append(obj_path)
            self.poses_rot.append(np.eye(4))
            self.poses_trans.append(np.eye(4))

    def load_offset(self):
        """
        Load offsets, mainly for robots
        """
        try:
            cur_path = os.path.abspath(os.path.dirname(__file__))
            offset_file = os.path.join(cur_path, "../data/robots/", "center_offset.txt")
            model_file = os.path.join(cur_path, "../data/robots/", "models.txt")
            with open(model_file, "r+") as file:
                content = file.readlines()
                model_paths = [path.strip().split("/")[-1] for path in content]
            offset = np.loadtxt(offset_file).astype(np.float32)
            offset_map = {}
            for i in range(offset.shape[0]):
                offset_map[model_paths[i]] = offset[i, :]
                colors = ["red", "green", "yellow"]
                for c in colors:
                    color_name_ = model_paths[i].replace(".DAE", "_{}.DAE".format(c))
                    offset_map[color_name_] = offset[i, :]
            return offset_map
        except:
            print("offsets are not used")
            return {}

    def parallel_assimp_load(self, mesh_files, scales):
        """
        use multiprocessing to load objects
        """

        if len(mesh_files) == 0:
            return None, None
        if not self.parallel_load_mesh:
            scenes = [load_mesh_single([mesh_files[i], scales[i], self._offset_map])
                    for i in range(len(mesh_files))]
        else:
            p = multiprocessing.Pool(processes=4) 
            scenes = p.map_async(
                load_mesh_single,
                [
                    [mesh_files[i], scales[i], self._offset_map]
                    for i in range(len(mesh_files))
                ],
            ).get()
            p.terminate()

        textures = [0 for _ in scenes]

        if self.parallel_textures:
            p =  multiprocessing.pool.ThreadPool(
                processes=4
            ) 
            textures = p.map_async(
                load_texture_single,
                [[mesh_files[i], scenes[i][-1]] for i in range(len(scenes))],
            ).get()
            p.terminate()
            texture_img = [t[0] for t in textures]
            texture_id = bindTexture(texture_img)
            for i, id in enumerate(texture_id):
                textures[i][0] = id

        return scenes, textures

    def load_objects(
        self,
        obj_paths,
        texture_paths=None,
        colors=None,
        scale=None,
        data=None,
        add=False,
    ):
        if scale is None:
            scale = [1] * len(obj_paths)
        if texture_paths is None:
            texture_paths = [""] * len(obj_paths)
        if colors is None:
            colors = get_mask_colors(len(obj_paths))

        self.colors.extend(colors)
        start_time = time.time()
        scenes, textures = self.parallel_assimp_load(obj_paths, scale)
        # print("assimp time:", time.time() - start_time)

        for i in range(len(obj_paths)):
            if len(self.instances) == 0:
                self.instances.append(0)
            else:
                self.instances.append(self.instances[-1] + len(self.materials[-1]))
            self.load_object(
                obj_paths[i], texture_paths[i], scenes[i], textures[i], scale[i], data
            )

    def set_camera(self, camera, target, up):
        self.camera = camera
        self.target = target
        self.up = up
        V = lookat(self.camera, self.target, up=self.up)
        self.V = np.ascontiguousarray(V, np.float32)

    def set_camera_default(self):
        self.V = np.eye(4)

    def set_fov(self, fov):
        self.fov = fov
        # this is vertical fov
        P = perspective(self.fov, float(self.width) / float(self.height), 0.01, 100)
        self.P = np.ascontiguousarray(P, np.float32)

    def set_projection_matrix(self, w, h, fu, fv, u0, v0, znear, zfar):
        L = -(u0) * znear / fu
        R = +(w - u0) * znear / fu
        T = -(v0) * znear / fv
        B = +(h - v0) * znear / fv

        P = np.zeros((4, 4), dtype=np.float32)
        P[0, 0] = 2 * znear / (R - L)
        P[1, 1] = 2 * znear / (T - B)
        P[2, 0] = (R + L) / (L - R)
        P[2, 1] = (T + B) / (B - T)
        P[2, 2] = (zfar + znear) / (zfar - znear)
        P[2, 3] = 1.0
        P[3, 2] = (2 * zfar * znear) / (znear - zfar)
        self.P = P

        # set intrinsics
        self.intrinsic_matrix = np.eye(3)
        self.intrinsic_matrix[0, 0] = fu
        self.intrinsic_matrix[1, 1] = fv
        self.intrinsic_matrix[0, 2] = u0
        self.intrinsic_matrix[1, 2] = v0

    def set_light_color(self, color):
        self.lightcolor = color

    def set_light_pos(self, light):
        self.lightpos = light

    def render_grid(self):
        GL.glUseProgram(self.shaderProgram_simple)
        GL.glBindVertexArray(self.grid)
        GL.glUniformMatrix4fv(
            GL.glGetUniformLocation(self.shaderProgram_simple, "V"),
            1,
            GL.GL_TRUE,
            self.V,
        )
        GL.glUniformMatrix4fv(
            GL.glGetUniformLocation(self.shaderProgram_simple, "P"),
            1,
            GL.GL_FALSE,
            self.P,
        )
        GL.glDrawElements(
            GL.GL_LINES, 160, GL.GL_UNSIGNED_INT, np.arange(160, dtype=np.int)
        )
        GL.glBindVertexArray(0)
        GL.glUseProgram(0)

    def render(
        self,
        cls_indexes,
        image_tensor,
        seg_tensor,
        point_info=None,
        color_idx=None,
        color_list=None,
        normal_tensor=None,
        line_info=None,
        pc1_tensor=None,
        pc2_tensor=None,
        cpu=False,
        only_rgb=False,
        white_bg=True,
        draw_grid=False,
        polygon_mode=0,
        coordinate_axis=0,
        draw_normal=False,
        point_capture_toggle=False,
    ):
        frame = 0
        if white_bg:
            GL.glClearColor(1.0, 1.0, 1.0, 1.0)
        else:
            GL.glClearColor(0, 0, 0, 1)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        GL.glEnable(GL.GL_DEPTH_TEST)

        if draw_grid:
            if not hasattr(self, "grid"):
                self.grid = self.generate_grid()
            self.render_grid()

        if line_info is not None or len(self.lineVAOs) > 0:
            self.lineVAOs = self.generate_lines(line_info, lineVAOs=self.lineVAOs)
        if point_info is not None or len(self.pointVAOs) > 0:
            self.pointVAOs = self.generate_points(point_info, pointVAOs=self.pointVAOs)
        if coordinate_axis > 0:
            self.coordVAOs = self.generate_coordinate_axis(coordVAOs=self.coordVAOs)

        if polygon_mode == 0:
            GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_FILL)
        if polygon_mode == 1:
            GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_LINE)
        if polygon_mode == 2:
            GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_POINT)

        size = 0
        color_cnt = 0

        for render_cnt in range(2):
            if point_capture_toggle and point_info is not None:
                break
            for i in range(len(cls_indexes)):
                index = cls_indexes[i]
                is_materialed = self.is_materialed[index]
                if is_materialed:
                    num = len(self.materials[index])
                    for idx in range(num):
                        is_texture = self.is_textured[index][idx]  # index
                        if is_texture:
                            shader = (
                                self.shaderProgram_textureMat
                            )  # self.shaderProgram_textureMat
                        elif self.textures[index][idx] == "color":
                            shader = self.shaderProgram_textureless
                        else:
                            shader = self.shaderProgram_material
                        GL.glUseProgram(shader)

                        if draw_normal and render_cnt == 1:
                            shader = self.shaderProgram_textureMatNormal
                            GL.glUseProgram(shader)
                            GL.glUniform3f(
                                GL.glGetUniformLocation(
                                    self.shaderProgram_textureMatNormal, "normal_color"
                                ),
                                *[0.0, 1.0, 0.0]
                            )
                            GL.glUniform1f(
                                GL.glGetUniformLocation(
                                    self.shaderProgram_textureMatNormal,
                                    "normal_magnitude",
                                ),
                                0.03,
                            )
                            GL.glUniform1f(
                                GL.glGetUniformLocation(
                                    self.shaderProgram_textureMatNormal, "face_normal"
                                ),
                                1.0,
                            )

                        GL.glUniformMatrix4fv(
                            GL.glGetUniformLocation(shader, "V"), 1, GL.GL_TRUE, self.V
                        )
                        GL.glUniformMatrix4fv(
                            GL.glGetUniformLocation(shader, "P"), 1, GL.GL_FALSE, self.P
                        )
                        GL.glUniformMatrix4fv(
                            GL.glGetUniformLocation(shader, "pose_trans"),
                            1,
                            GL.GL_FALSE,
                            self.poses_trans[i],
                        )
                        GL.glUniformMatrix4fv(
                            GL.glGetUniformLocation(shader, "pose_rot"),
                            1,
                            GL.GL_TRUE,
                            self.poses_rot[i],
                        )
                        GL.glUniform3f(
                            GL.glGetUniformLocation(shader, "light_position"),
                            *self.lightpos
                        )
                        GL.glUniform3f(
                            GL.glGetUniformLocation(shader, "instance_color"),
                            *self.colors[index]
                        )
                        GL.glUniform3f(
                            GL.glGetUniformLocation(shader, "light_color"),
                            *self.lightcolor
                        )
                        if color_idx is None or i not in color_idx:
                            GL.glUniform3f(
                                GL.glGetUniformLocation(shader, "mat_diffuse"),
                                *self.materials[index][idx][:3]
                            )
                            GL.glUniform3f(
                                GL.glGetUniformLocation(shader, "mat_specular"),
                                *self.materials[index][idx][3:6]
                            )
                            GL.glUniform3f(
                                GL.glGetUniformLocation(shader, "mat_ambient"),
                                *self.materials[index][idx][6:9]
                            )
                            GL.glUniform1f(
                                GL.glGetUniformLocation(shader, "mat_shininess"),
                                self.materials[index][idx][-1],
                            )
                        else:
                            GL.glUniform3f(
                                GL.glGetUniformLocation(shader, "mat_diffuse"),
                                *color_list[color_cnt]
                            )
                            GL.glUniform3f(
                                GL.glGetUniformLocation(shader, "mat_specular"),
                                *color_list[color_cnt]
                            )
                            GL.glUniform3f(
                                GL.glGetUniformLocation(shader, "mat_ambient"),
                                *color_list[color_cnt]
                            )
                            GL.glUniform1f(
                                GL.glGetUniformLocation(shader, "mat_shininess"), 100
                            )
                            color_cnt += 1
                        try:
                            if is_texture:
                                GL.glActiveTexture(GL.GL_TEXTURE0)
                                GL.glBindTexture(
                                    GL.GL_TEXTURE_2D, self.textures[index][idx]
                                )  # self.instances[index]
                                # GL.glUniform1i(self.texUnitUniform_textureMat, 0)
                            GL.glBindVertexArray(
                                self.VAOs[self.instances[index] + idx]
                            )  #
                            GL.glDrawElements(
                                GL.GL_TRIANGLES,
                                self.faces[self.instances[index] + idx].size,
                                GL.GL_UNSIGNED_INT,
                                self.faces[self.instances[index] + idx],
                            )
                        finally:
                            GL.glBindVertexArray(0)
                            GL.glUseProgram(0)
            if not draw_normal:
                break
        GL.glDisable(GL.GL_DEPTH_TEST)

        # mapping
        if not cpu:
            self.r.map_tensor(
                int(self.color_tex),
                int(self.width),
                int(self.height),
                image_tensor.data_ptr(),
            )
            self.r.map_tensor(
                int(self.color_tex_3),
                int(self.width),
                int(self.height),
                seg_tensor.data_ptr(),
            )
            if normal_tensor is not None:
                self.r.map_tensor(
                    int(self.color_tex_2),
                    int(self.width),
                    int(self.height),
                    normal_tensor.data_ptr(),
                )
            if pc1_tensor is not None:
                self.r.map_tensor(
                    int(self.color_tex_4),
                    int(self.width),
                    int(self.height),
                    pc1_tensor.data_ptr(),
                )
            if pc2_tensor is not None:
                self.r.map_tensor(
                    int(self.color_tex_5),
                    int(self.width),
                    int(self.height),
                    pc2_tensor.data_ptr(),
                )

        else:
            GL.glReadBuffer(GL.GL_COLOR_ATTACHMENT0)
            frame = GL.glReadPixels(
                0, 0, self.width, self.height, GL.GL_BGRA, GL.GL_FLOAT
            )
            frame = frame.reshape(self.height, self.width, 4)[::-1, :]
            if only_rgb:
                return [frame]

            GL.glReadBuffer(GL.GL_COLOR_ATTACHMENT1)
            normal = GL.glReadPixels(
                0, 0, self.width, self.height, GL.GL_BGRA, GL.GL_FLOAT
            )
            normal = normal.reshape(self.height, self.width, 4)[::-1, :]

            GL.glReadBuffer(GL.GL_COLOR_ATTACHMENT2)
            seg = GL.glReadPixels(
                0, 0, self.width, self.height, GL.GL_BGRA, GL.GL_FLOAT
            )
            seg = seg.reshape(self.height, self.width, 4)[::-1, :]

            # points in camera coordinate
            GL.glReadBuffer(GL.GL_COLOR_ATTACHMENT4)
            pc3 = GL.glReadPixels(
                0, 0, self.width, self.height, GL.GL_RGBA, GL.GL_FLOAT
            )
            pc3 = pc3.reshape(self.height, self.width, 4)[::-1, :]
            pc3 = pc3[:, :, :4]  # 3

            return [frame, seg, normal, pc3]

    def get_num_objects(self):
        return len(self.objects)

    def set_poses(self, poses):
        self.poses_rot = [np.ascontiguousarray(quat2rotmat(item[3:])) for item in poses]
        self.poses_trans = [np.ascontiguousarray(xyz2mat(item[:3])) for item in poses]

    def set_allocentric_poses(self, poses):
        self.poses_rot = []
        self.poses_trans = []
        for pose in poses:
            x, y, z = pose[:3]
            quat_input = pose[3:]
            dx = np.arctan2(x, -z)
            dy = np.arctan2(y, -z)
            quat = euler2quat(-dy, -dx, 0, axes="sxyz")
            quat = qmult(quat, quat_input)
            self.poses_rot.append(np.ascontiguousarray(quat2rotmat(quat)))
            self.poses_trans.append(np.ascontiguousarray(xyz2mat(pose[:3])))

    def transform_vector(self, vec):
        vec = np.array(vec)
        zeros = np.zeros_like(vec)

        vec_t = self.transform_point(vec)
        zero_t = self.transform_point(zeros)

        v = vec_t - zero_t
        return v

    def transform_point(self, vec):
        vec = np.array(vec)
        if vec.shape[0] == 3:
            v = self.V.dot(np.concatenate([vec, np.array([1])]))
            return v[:3] / v[-1]
        elif vec.shape[0] == 4:
            v = self.V.dot(vec)
            return v / v[-1]
        else:
            return None

    def transform_pose(self, pose):
        pose_rot = quat2rotmat(pose[3:])
        pose_trans = xyz2mat(pose[:3])
        pose_cam = self.V.dot(pose_trans.T).dot(pose_rot).T
        return np.concatenate([mat2xyz(pose_cam), safemat2quat(pose_cam[:3, :3].T)])

    def get_poses(self):
        mat = [
            self.V.dot(self.poses_trans[i].T).dot(self.poses_rot[i]).T
            for i in range(len(self.poses_rot))
        ]

        poses = [
            np.concatenate([mat2xyz(item), safemat2quat(item[:3, :3].T)])
            for item in mat
        ]
        return poses

    def get_world_poses(self):
        mat = [
            self.poses_trans[i].T.dot(self.poses_rot[i].T)
            for i in range(len(self.poses_rot))
        ]
        return mat

    def get_egocentric_poses(self):
        return self.get_poses()

    def get_allocentric_poses(self):
        poses = self.get_poses()
        poses_allocentric = []
        for pose in poses:
            dx = np.arctan2(pose[0], -pose[2])
            dy = np.arctan2(pose[1], -pose[2])
            quat = euler2quat(-dy, -dx, 0, axes="sxyz")
            quat = qmult(qinverse(quat), pose[3:])
            poses_allocentric.append(np.concatenate([pose[:3], quat]))
        return poses_allocentric

    def release(self):
        print(self.glstring)
        self.clean()
        self.r.release()

    def clean(self):
        GL.glDeleteTextures(
            [
                self.color_tex,
                self.color_tex_2,
                self.color_tex_3,
                self.color_tex_4,
                self.depth_tex,
            ]
        )
        self.color_tex = None
        self.color_tex_2 = None
        self.color_tex_3 = None
        self.color_tex_4 = None
        self.depth_tex = None

        GL.glDeleteFramebuffers(1, [self.fbo])
        self.fbo = None
        GL.glDeleteBuffers(len(self.VAOs), self.VAOs)
        self.VAOs = []
        GL.glDeleteBuffers(len(self.VBOs), self.VBOs)
        self.VBOs = []
        self.clean_line_point()

        def flatten(container):
            for i in container:
                if isinstance(i, (list,tuple)):
                    for j in flatten(i):
                        yield j
                else:
                    yield i

        textures_ = list(flatten(self.textures))
        if len(textures_) > 0:
            GL.glDeleteTextures(textures_)
        self.textures = []
        self.objects = []   
        self.faces = []   
        self.poses_trans = []   
        self.poses_rot = []   
        self.colors = []

    def clean_line_point(self):
        if self.lineVAOs is not None and len(self.lineVAOs) > 0:
            GL.glDeleteBuffers(len(self.lineVAOs), self.lineVAOs)
            self.lineVAOs = [] 
        if self.pointVAOs is not None and len(self.pointVAOs) > 0:
            GL.glDeleteBuffers(len(self.pointVAOs), self.pointVAOs)
            self.pointVAOs = []   

    def get_num_instances(self):
        return len(self.instances)

    def capture_point(self, frames):
        point_mask = frames[1]
        bg_mask = (point_mask[..., :3].sum(-1) != 3) * (
            point_mask[..., :3].sum(-1) != 0
        )
        point_pos = frames[3][..., :3].reshape(-1, 3)[bg_mask.reshape(-1)].T
        point_pos = self.V[:3, :3].T.dot(point_pos - self.V[:3, [3]])
        point_color = frames[0][..., :3].reshape(-1, 3)[bg_mask.reshape(-1)] * 255

        point_size = [1]
        point_info = [[point_pos], [point_color[:, [2, 1, 0]]], point_size, True]

        return point_info

    def vis(
        self,
        poses,
        cls_indexes,
        color_idx=None,
        color_list=None,
        cam_pos=[0, 0, 2.0],
        V=None,
        distance=2.0,
        shifted_pose=None,
        interact=0,
        visualize_context={},
        window_name="test",
    ):
        """
        a complicated visualization function
        """
        theta = 0
        cam_x, cam_y, cam_z = cam_pos
        sample = []
        new_poses = []

        # center view
        if len(poses) > 0:
            origin = np.linalg.inv(unpack_pose(poses[0]))
            if shifted_pose is not None:
                origin = np.linalg.inv(shifted_pose)
            for pose in poses:
                pose = unpack_pose(pose)
                pose = origin.dot(pose)
                new_poses.append(pack_pose(pose))
            poses = new_poses
            self.set_poses(poses)

        cam_pos = np.array([cam_x, cam_y, cam_z])
        self.set_camera(cam_pos, cam_pos * 2, [0, 1, 0])
        if V is not None:
            self.V = V[...]
            cam_pos = V[:3, 3]
        self.set_light_pos(cam_pos)

        mouse_events = {
            "view_dir": -self.V[:3, 3],
            "view_origin": np.array([0, 0, 0.0]),  # anchor
            "_mouse_ix": -1,
            "_mouse_iy": -1,
            "down": False,
            "shift": False,
            "trackball": Trackball(self.width, self.height, cam_pos=cam_pos),
        }

        def update_dir():
            view_dir = mouse_events["view_origin"] - self.V[:3, 3]
            self.set_camera(
                self.V[:3, 3], self.V[:3, 3] - view_dir, [0, 1, 0]
            )  # would shift along the sphere
            self.V[...] = self.V[...].dot(mouse_events["trackball"].property["model"].T)
            if V is not None:
                V[...] = self.V

        def change_dir(
            event, x, y, flags, param
        ):  # fix later to be a finalized version
             
            if event == cv2.EVENT_LBUTTONDOWN:
                mouse_events["_mouse_ix"], mouse_events["_mouse_iy"] = x, y
                mouse_events["down"] = True
               
            if event == cv2.EVENT_MBUTTONDOWN:
                mouse_events["_mouse_ix"], mouse_events["_mouse_iy"] = x, y
                mouse_events["shift"] = True
             
            if event == cv2.EVENT_MOUSEMOVE  and (flags >= 8):
                if mouse_events["down"] and flags < 15:
                    dx = (x - mouse_events["_mouse_ix"]) / -7.0
                    dy = (y - mouse_events["_mouse_iy"]) / -7.0
                    mouse_events["trackball"].on_mouse_drag(x, y, dx, dy)
                    update_dir()
                
                if mouse_events["down"] and flags > 15:
                    dx = (x - mouse_events["_mouse_ix"]) / (-4000.0 / self.V[2, 3])
                    dy = (y - mouse_events["_mouse_iy"]) / (-4000.0 / self.V[2, 3])
                    self.V[:3, 3] += 0.5 * np.array([0, 0, dx + dy])
                    mouse_events["view_origin"] += 0.5 * np.array(
                        [0, 0, dx + dy]
                    )  # change                    
                    update_dir()

                if mouse_events["shift"]:
                    dx = (x - mouse_events["_mouse_ix"]) / (-4000.0 / self.V[2, 3])
                    dy = (y - mouse_events["_mouse_iy"]) / (-4000.0 / self.V[2, 3])
                    self.V[:3, 3] += 0.5 * np.array([dx, dy, 0])
                    mouse_events["view_origin"] += 0.5 * np.array(
                        [-dx, dy, 0]
                    )  # change
                    update_dir()

            if event == cv2.EVENT_LBUTTONUP:
                mouse_events["down"] = False
            if event == cv2.EVENT_MBUTTONUP:
                mouse_events["shift"] = False
          
        if interact > 0:
            cv2.namedWindow(window_name)
            cv2.setMouseCallback(window_name, change_dir)

        # update_dir()
        img = np.zeros([self.height, self.width, 3])
        img_toggle = 0
        polygon_toggle = 0
        coordinate_axis_toggle = 0
        normal_toggle = False
        mouse_toggle  = False
        white_bg_toggle = "white_bg" in visualize_context
        grid_toggle = "grid" in visualize_context
        write_video_toggle = "write_video" in visualize_context
        point_capture_toggle = False

        video_writer = None
        line_info, point_info = None, None

        if "reset_line_point" in visualize_context:
            self.clean_line_point()

        if "line" in visualize_context:
            if "line_color" not in visualize_context:
                visualize_context["line_color"] = [
                    [255, 0, 0] for _ in range(len(visualize_context["line"]))
                ]
            if "thickness" not in visualize_context:
                visualize_context["thickness"] = [
                    2 for _ in range(len(visualize_context["line"]))
                ]
            line_info = [
                visualize_context["line"],
                visualize_context["line_color"],
                visualize_context["thickness"],
                True,
            ]
            self.lineVAOs = self.generate_lines(line_info, lineVAOs=self.lineVAOs)
            line_info[-1] = False

        if "project_point" in visualize_context:
            if "project_color" not in visualize_context:
                visualize_context["project_color"] = [
                    [255, 0, 0] for _ in range(len(visualize_context["project_point"]))
                ]
            if "point_size" not in visualize_context:
                visualize_context["point_size"] = [
                    2 for _ in range(len(visualize_context["project_point"]))
                ]

            point_info = [
                visualize_context["project_point"],
                visualize_context["project_color"],
                visualize_context["point_size"],
                True,
            ]
            self.pointVAOs = self.generate_points(point_info, pointVAOs=self.pointVAOs)
            point_info[-1] = False

        while True:
            new_cam_pos = -self.V[:3, :3].T.dot(self.V[:3, 3])
            q = cv2.waitKey(1)

            if interact > 0:
                if q == 9:
                    img_toggle = (img_toggle + 1) % 4
                elif q == 96:
                    polygon_toggle = (polygon_toggle + 1) % 3
                elif q == 32:
                    coordinate_axis_toggle = (coordinate_axis_toggle + 1) % 2
                elif q == ord("s"):
                    interact = 2
                elif q == ord("u"):
                    interact = 1
                elif q == ord("q"):
                    mouse_events["trackball"].theta_delta(5)
                    update_dir()
                elif q == ord("w"):
                    white_bg_toggle = not white_bg_toggle
                elif q == ord("v"):
                    write_video_toggle = not write_video_toggle
                elif q == ord("g"):
                    grid_toggle = not grid_toggle
                elif q == ord("1"):
                    normal_toggle = not normal_toggle

                elif q == ord("2"):
                    point_capture_toggle = not point_capture_toggle
                    self.pointVAOs = []
                    point_info = None

                elif q == ord("3"):
                    point_capture_toggle = not point_capture_toggle
                    point_info = None

                elif q == ord("e"):
                    mouse_events["trackball"].theta_delta(-5)
                    update_dir()

                elif q == ord("a"):
                    mouse_events["trackball"].phi_delta(5)
                    update_dir()
                elif q == ord("d"):
                    mouse_events["trackball"].phi_delta(-5)
                    update_dir()
                elif q == ord("z"):
                    self.V[:3, 3] += 0.02 * (
                        self.V[:3, 3] - mouse_events["view_origin"]
                    )
                    update_dir()
                elif q == ord("c"):  # move closer
                    self.V[:3, 3] -= 0.02 * (
                        self.V[:3, 3] - mouse_events["view_origin"]
                    )
                    update_dir()
                elif q == ord("x"):  # reset
                    self.set_camera(cam_pos, cam_pos * 2, [0, 1, 0])
                    mouse_events = {
                        "view_dir": -self.V[:3, 3],
                        "view_origin": np.array([0, 0, 0.0]),  # anchor
                        "_mouse_ix": -1,
                        "_mouse_iy": -1,
                        "down": False,
                        "shift": False,
                        "trackball": Trackball(
                            self.width, self.height, cam_pos=cam_pos
                        ),
                    }

                elif q == ord("i"):
                    for pose in poses:
                        pose[1] += 0.02
                elif q == ord("k"):
                    for pose in poses:
                        pose[1] -= 0.02
                elif q == ord("j"):
                    for pose in poses:
                        pose[0] -= 0.02
                elif q == ord("l"):
                    for pose in poses:
                        pose[0] += 0.02
                elif q == ord("m"):
                    for pose in poses:
                        pose[2] -= 0.02
                elif q == ord(","):
                    for pose in poses:
                        pose[2] += 0.02
                elif q == ord("n"):
                    print("camera V", self.V)
                elif q == ord("p"):
                    cur_dir = os.path.dirname(os.path.abspath(__file__))
                    Image.fromarray(
                        (np.clip(frame[0][:, :, [2, 1, 0]] * 255, 0, 255)).astype(
                            np.uint8
                        )
                    ).save(cur_dir + "/test.png")
                elif q == 27:  #
                    if V is not None:
                        V[:4, :4] = self.V
                    break
                elif q == ord("r"):  # rotate
                    for pose in poses:
                        pose[3:] = qmult(
                            axangle2quat([0, 0, 1], 5 / 180.0 * np.pi), pose[3:]
                        )
                

            self.set_poses(poses)
            self.set_light_pos(new_cam_pos)

            start_time = time.time()
            s = time.time()
            frames = self.render(
                cls_indexes,
                None,
                None,
                line_info=line_info,
                point_info=point_info,
                color_idx=color_idx,
                color_list=color_list,
                cpu=True,
                white_bg=white_bg_toggle,
                polygon_mode=polygon_toggle,
                point_capture_toggle=point_capture_toggle,
                coordinate_axis=coordinate_axis_toggle,
                draw_normal=normal_toggle,
                draw_grid=grid_toggle,
            )
            frame = [frames[img_toggle]]

            if point_info is not None:
                point_info[-1] = False
            if point_capture_toggle and (point_info is None):
                point_info = self.capture_point(frames)

            if img_toggle == 3:
                frame[0] = (frame[0] - frame[0].min()) / (
                    frame[0].max() - frame[0].min() + 1e-4
                )

            if "text" in visualize_context:  # annotate at top left
                text = visualize_context["text"]
                frame_ = frame[0].copy()
                for i, t in enumerate(text):
                    cv2.putText(
                        frame_,
                        t,
                        (0, 25 + i * 25),
                        cv2.FONT_HERSHEY_DUPLEX,
                        0.7,
                        [255, 255, 255],
                    )  # 0.7
                frame[0] = frame_

            img = cv2.cvtColor(np.concatenate(frame, axis=1), cv2.COLOR_RGB2BGR)
            if interact > 0:
                cv2.imshow(window_name, img[:, :, ::-1])
            if interact < 2:
                break
            if write_video_toggle:
                if not video_writer:
                    print("writing video to test.avi")
                    video_writer = cv2.VideoWriter(
                        "test.avi",
                        cv2.VideoWriter_fourcc(*"MJPG"),
                        10.0,
                        (self.width, self.height),
                    )
                video_writer.write(
                    np.clip(255 * img, 0, 255).astype(np.uint8)[..., [2, 1, 0]]
                )

        return np.clip(255 * img, 0, 255).astype(np.uint8)

def get_collision_points():
    """
    load collision points with the order of the link list and end effector
    """
    collision_pts = []
    links = [
        "link1",
        "link2",
        "link3",
        "link4",
        "link5",
        "link6",
        "link7",
        "hand",
        "finger",
        "finger",
    ]
    for i in range(len(links)):
        file = "../data/robots/{}.xyz".format(links[i])
        pts = np.loadtxt(file)[:, :3]
        sample_pts = pts[random.sample(range(pts.shape[0]), 50)]
        collision_pts.append(sample_pts)
    return np.array(collision_pts)


if __name__ == "__main__":
    
    model_path = "panda_arm" if len(sys.argv) < 2 else sys.argv[1]
    joints = np.array([[0.0258, -1.3479, -0.0357, -2.3427,  0.0027, 1.5498,  0.6864, 0, 0.04, 0.04]]) * 180 / 3.14

    if model_path == "ycb":
        models = ["003_cracker_box", "002_master_chef_can", "011_banana"]
        colors = [[0.9, 0, 0], [0.6, 0, 0], [0.3, 0, 0]]
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        obj_paths = [
            "{}/../data/objects/{}/model_normalized.obj".format(cur_dir, item)
            for item in models
        ]
        texture_paths = ["" for _ in range(len(obj_paths))]
        colors = [[0.9, 0, 0], [0, 0.9, 0], [0, 0, 0.9]]
        texture_paths = ["" for _ in range(len(obj_paths))]
        pose = np.array([-0, 0, 0.05, 1, 0, 0, 0])
        pose2 = np.array(
            [
                -1,
                0.05060109,
                -0.028915625,
                0.6582951,
                0.03479896,
                -0.036391996,
                -1.75107396,
            ]
        )
        pose3 = np.array(
            [
                1,
                0.019853603,
                0.12159989,
                -0.40458265,
                -0.036644224,
                -0.8464779,
                1.64578354,
            ]
        )
        poses = [pose, pose2, pose3]

    elif model_path == "shapenet" or model_path == "shapenet2":
        models = ["mug"] if model_path == "shapenet" else ["mug_2"]
        colors = [[0.9, 0, 0]]
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        obj_paths = [
            "{}/../data/objects/{}/models/model_normalized.obj".format(cur_dir, item)
            for item in models
        ]
        texture_paths = ["" for _ in range(len(obj_paths))]
        colors = [[0.9, 0, 0]]
        texture_paths = ["" for _ in range(len(obj_paths))]
        pose = np.array([-0, 0, 0.05, 0, 1, 0, 0])
        poses = [pose]

    elif model_path == "panda_arm":
        from robotPose.robot_pykdl import *
        base_link = "panda_link0"
        robot = robot_kinematics(model_path)
        models = [
            "link1",
            "link2",
            "link3",
            "link4",
            "link5",
            "link6",
            "link7",
            "hand",
            "finger",
            "finger",
        ]
        obj_paths = ["../data/robots/{}.DAE".format(item) for item in models]
        colors = [[0, 0.1 * (idx + 1), 0] for idx in range(len(models))]
        texture_paths = ["" for item in models]

        poses = []
        pose = robot.forward_kinematics_parallel((joints))[0]

        for i in range(len(pose)):
            pose_i = pose[i]  # camera_extrinsics.dot()
            quat = mat2quat(pose_i[:3, :3])
            trans = pose_i[:3, 3]
            poses.append(np.hstack((trans, quat)))

    elif model_path == "points":
        robot = robot_kinematics(model_path)
        poses = []
        obj_paths = []

    else:
        if "," in model_path:
            model_path = model_path.split(",")
        else:
            model_path = [model_path]
        obj_paths = [
            os.path.join(p, "model_normalized.obj") if os.path.isdir(p) else p
            for p in model_path
        ]
        poses = [
            np.array([0.5 * i - 0.8, 0, 0, 1, 0, 0, 0]) for i in range(len(obj_paths))
        ]

    width = 640  # 800
    height = 480  # 600
    start_time = time.time()
    renderer = YCBRenderer(
        width=width,
        height=height,
        gpu_id=0,
        render_marker=False,
        robot=model_path,
        offset=True,
    )
    print("setup time:{:.3f}".format(time.time() - start_time))
    start_time = time.time()
    print("obj_paths:", obj_paths)
    if len(obj_paths) > 0:
        renderer.load_objects(obj_paths)
    print("load time:{:.3f}".format(time.time() - start_time))
    theta = 0
    z = 1
    fix_pos = [np.sin(theta), z, np.cos(theta)]
    renderer.set_camera(fix_pos, [0, 0, 0], [0, 1, 0])
    fix_pos = np.zeros(3)

    renderer.set_light_pos([0, 0, 1])
    renderer.set_camera_default()
    renderer.set_projection_matrix(640, 480, 525, 525, 319.5, 239.5, 0.1, 6)
    if model_path == "points":
        points = get_collision_points()
        pose = robot.forward_kinematics_parallel(joints)[0]
        r = pose[..., :3, :3]
        t = pose[..., :3, [3]]
        x = np.matmul(r, points.swapaxes(-1, -2)).swapaxes(-1, -2) + t.swapaxes(-1, -2)
        x = x.reshape([-1, 3])
        x = np.concatenate([x] * 100, axis=0)
        renderer.vis(
            poses,
            range(len(poses)),
            shifted_pose=np.eye(4),
            interact=2,
            visualize_context={
                "white_bg": True,
                "project_point": [x.T],
                "static_buffer": True,
            },
        )
    else:
        renderer.vis(
            poses,
            range(len(poses)),
            shifted_pose=np.eye(4),
            interact=2,
            visualize_context={
                "white_bg": True,
            },
        )
