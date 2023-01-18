import random
import glfw
import glfw.GLFW as GLFW_CONSTANTS
from OpenGL.GL import *
import numpy as np
from OpenGL.GL.shaders import compileProgram, compileShader
import pyrr
from PIL import Image
import math
import time

SCREEN_WIDTH = 1980
SCREEN_HEIGHT = 1080
RETURN_ACTION_CONTINUE = 0
RETURN_ACTION_END = 1
NUMBER_OF_PARTICLES = 750


def initialize_glfw():
    glfw.init()
    glfw.window_hint(GLFW_CONSTANTS.GLFW_CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(GLFW_CONSTANTS.GLFW_CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(
        GLFW_CONSTANTS.GLFW_OPENGL_PROFILE,
        GLFW_CONSTANTS.GLFW_OPENGL_CORE_PROFILE
    )
    glfw.window_hint(
        GLFW_CONSTANTS.GLFW_OPENGL_FORWARD_COMPAT,
        GLFW_CONSTANTS.GLFW_TRUE
    )
    glfw.window_hint(GLFW_CONSTANTS.GLFW_DOUBLEBUFFER, GL_FALSE)

    window = glfw.create_window(SCREEN_WIDTH, SCREEN_HEIGHT, "My Game", None, None)
    glfw.make_context_current(window)
    glfw.set_input_mode(
        window,
        GLFW_CONSTANTS.GLFW_CURSOR,
        GLFW_CONSTANTS.GLFW_CURSOR_HIDDEN
    )
    print('Renderer:', glGetString(GL_RENDERER).decode("utf-8"))

    return window


class Particle:
    def __init__(self, position, eulers):
        self.position = np.array(position, dtype=np.float32)
        self.eulers = np.array(eulers, dtype=np.float32)
        self.lifespan = 20
        self.creation_time = time.time()
        self.color = np.array([1.0, 1.0, 1.0], dtype=np.float32)

    def update_lifespan(self):
        self.time_alive = time.time() - self.creation_time
        if self.time_alive < self.lifespan:

            self.color[0] = (self.lifespan - self.time_alive) / self.lifespan
            self.color[1] = (self.lifespan - self.time_alive) / self.lifespan
            self.color[2] = (self.lifespan - self.time_alive) / self.lifespan


            self.position[2] -= 0.1
            if self.position[2] < 0.0:
                self.position[2] = random.randint(50, 100)/10


class Player:
    def __init__(self, position):
        self.position = np.array(position, dtype=np.float32)
        self.theta = 0
        self.phi = 0
        self.update_vectors()

    def update_vectors(self):
        self.forwards = np.array(
            [
                np.cos(np.deg2rad(self.theta)) * np.cos(np.deg2rad(self.phi)),
                np.sin(np.deg2rad(self.theta)) * np.cos(np.deg2rad(self.phi)),
                np.sin(np.deg2rad(self.phi))
            ]
        )

        globalUp = np.array([0, 0, 1], dtype=np.float32)

        self.right = np.cross(self.forwards, globalUp)

        self.up = np.cross(self.right, self.forwards)


class Scene:
    def __init__(self):
        self.snowflakes = []
        for i in range(NUMBER_OF_PARTICLES):
            self.snowflakes.append(Particle(
                position=[random.randint(-100, 100)/10, random.randint(-100, 100)/10, random.randint(50, 100)/10],
                eulers=[0, 0, 0]
            ))

        self.player = Player(position=[0, 0, 2])

    def update(self, rate):
        for snowflake in self.snowflakes:
            snowflake.update_lifespan()

    def move_player(self, dPos):
        dPos = np.array(dPos, dtype=np.float32)
        self.player.position += dPos

    def spin_player(self, dTheta, dPhi):
        self.player.theta += dTheta
        if self.player.theta > 360:
            self.player.theta -= 360
        elif self.player.theta < 0:
            self.player.theta += 360

        self.player.phi = min(
            89, max(-89, self.player.phi + dPhi)
        )

        self.player.update_vectors()


class GraphicEngine:
    def __init__(self):
        self.snow_texture = Material("textureFiles/blue_snowflake.png")
        self.quad = Quad()

        glClearColor(1, 1, 1, 1)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        self.shader = self.createShader("shaders/vertex.txt", "shaders/fragment.txt")
        glUseProgram(self.shader)
        glUniform1i(glGetUniformLocation(self.shader, "imageTexture"), 0)
        glEnable(GL_DEPTH_TEST)

        projection_transform = pyrr.matrix44.create_perspective_projection(
            fovy=45,
            aspect=640 / 480,
            near=0.1,
            far=30,
            dtype=np.float32
        )

        glUniformMatrix4fv(
            glGetUniformLocation(self.shader, "projection"),
            1, GL_FALSE, projection_transform
        )

        self.colorLocation = glGetUniformLocation(self.shader, "color")
        self.modelMatrixLocation = glGetUniformLocation(self.shader, "model")
        self.viewMatrixLocation = glGetUniformLocation(self.shader, "view")

    def createShader(self, vertexFilepath, fragmentFilepath):
        with open(vertexFilepath, 'r') as f:
            vertex_src = f.readlines()

        with open(fragmentFilepath, 'r') as f:
            fragment_src = f.readlines()

        shader = compileProgram(
            compileShader(vertex_src, GL_VERTEX_SHADER),
            compileShader(fragment_src, GL_FRAGMENT_SHADER)
        )
        return shader

    def render(self, scene):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        glUseProgram(self.shader)
        self.snow_texture.use()
        glBindVertexArray(self.quad.vao)

        view_transform = pyrr.matrix44.create_look_at(
            eye=scene.player.position,
            target=scene.player.position + scene.player.forwards,
            up=scene.player.up,
            dtype=np.float32)
        glUniformMatrix4fv(self.viewMatrixLocation, 1, GL_FALSE, view_transform)

        for snowflake in scene.snowflakes:
            dist_vector = scene.player.position - snowflake.position
            angle1 = np.arctan2(-dist_vector[1], dist_vector[0])
            dist2d = math.sqrt(dist_vector[0]**2 + dist_vector[1]**2)
            angle2 = np.arctan2(dist_vector[2], dist2d) + math.radians(70)

            model_transform = pyrr.matrix44.create_identity(dtype=np.float32)
            model_transform = pyrr.matrix44.multiply(
                model_transform,
                pyrr.matrix44.create_from_scale(scale=[0.2, 0.2, 0.2], dtype=np.float32)
            )
            model_transform = pyrr.matrix44.multiply(
                model_transform,
                pyrr.matrix44.create_from_y_rotation(theta=angle2, dtype=np.float32)
            )
            model_transform = pyrr.matrix44.multiply(
                model_transform,
                pyrr.matrix44.create_from_z_rotation(theta=angle1, dtype=np.float32)
            )
            model_transform = pyrr.matrix44.multiply(
                model_transform,
                pyrr.matrix44.create_from_translation(vec=snowflake.position, dtype=np.float32)
            )

            color = pyrr.vector3.create(snowflake.color[0], snowflake.color[1], snowflake.color[2], dtype=np.float32)
            glUniform3fv(self.colorLocation, 1, color)
            glUniformMatrix4fv(self.modelMatrixLocation, 1, GL_FALSE, model_transform)
            glDrawArrays(GL_TRIANGLE_FAN, 0, self.quad.vertex_count)

        glFlush()

    def quit(self):
        self.quad.destroy()
        self.snow_texture.destroy()
        glDeleteProgram(self.shader)


class App:
    def __init__(self, window):
        self.window = window
        self.renderer = GraphicEngine()
        self.scene = Scene()

        self.lastTime = glfw.get_time()
        self.currentTime = 0
        self.numFrames = 0
        self.frameTime = 0

        self.walk_offset_lookup = {
            1: 0,
            2: 90,
            3: 45,
            4: 180,
            6: 135,
            7: 90,
            8: 270,
            9: 315,
            11: 0,
            12: 225,
            13: 270,
            14: 180
        }

        self.mainLoop()

    def mainLoop(self):
        running = True
        while running:
            # check events

            if glfw.window_should_close(self.window) \
                    or glfw.get_key(self.window, GLFW_CONSTANTS.GLFW_KEY_ESCAPE) == GLFW_CONSTANTS.GLFW_PRESS:
                running = False

            self.handleKeys()
            self.handleMouse()

            glfw.poll_events()

            self.scene.update(self.frameTime / 16.7)

            self.renderer.render(self.scene)

            # timing
            self.calculateFramerate()
        self.quit()

    def handleKeys(self):

        """
        w: 1 -> 0 degrees
        a: 2 -> 90 degrees
        w & a: 3 -> 45 degrees
        s: 4 -> 180 degrees
        w & s: 5 -> x
        a & s: 6 -> 135 degrees
        w & a & s: 7 -> 90 degrees
        d: 8 -> 270 degrees
        w & d: 9 -> 315 degrees
        a & d: 10 -> x
        w & a & d: 11 -> 0 degrees
        s & d: 12 -> 225 degrees
        w & s & d: 13 -> 270 degrees
        a & s & d: 14 -> 180 degrees
        w & a & s & d: 15 -> x
        """
        combo = 0
        directionModifier = 0

        if glfw.get_key(self.window, GLFW_CONSTANTS.GLFW_KEY_W) == GLFW_CONSTANTS.GLFW_PRESS:
            combo += 1
        if glfw.get_key(self.window, GLFW_CONSTANTS.GLFW_KEY_A) == GLFW_CONSTANTS.GLFW_PRESS:
            combo += 2
        if glfw.get_key(self.window, GLFW_CONSTANTS.GLFW_KEY_S) == GLFW_CONSTANTS.GLFW_PRESS:
            combo += 4
        if glfw.get_key(self.window, GLFW_CONSTANTS.GLFW_KEY_D) == GLFW_CONSTANTS.GLFW_PRESS:
            combo += 8

        if combo in self.walk_offset_lookup:
            directionModifier = self.walk_offset_lookup[combo]
            dPos = [
                0.1 * self.frameTime / 16.7 * np.cos(np.deg2rad(self.scene.player.theta + directionModifier)),
                0.1 * self.frameTime / 16.7 * np.sin(np.deg2rad(self.scene.player.theta + directionModifier)),
                0
            ]
            self.scene.move_player(dPos)

    def handleMouse(self):

        (x, y) = glfw.get_cursor_pos(self.window)
        rate = self.frameTime / 50
        theta_increment = rate * ((SCREEN_WIDTH / 2) - x)
        phi_increment = rate * ((SCREEN_HEIGHT / 2) - y)
        self.scene.spin_player(theta_increment, phi_increment)
        glfw.set_cursor_pos(self.window, SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2)

    def calculateFramerate(self):

        self.currentTime = glfw.get_time()
        delta = self.currentTime - self.lastTime
        if delta >= 1:
            framerate = max(1, int(self.numFrames / delta))
            glfw.set_window_title(self.window, f"Running at {framerate} fps.")
            self.lastTime = self.currentTime
            self.numFrames = -1
            self.frameTime = float(1000.0 / max(1, framerate))
        self.numFrames += 1

    def quit(self):
        self.renderer.quit()


class Quad:
    def __init__(self):
        self.vertices = (
            -0.5, -0.5, 0, 0, 1, 1, 0, 1,
            -0.5, 0.5, 0, 0, 0, 1, 0, 1,
            0.5, 0.5, 0, 1, 0, 1, 0, 1,
            0.5, -0.5, 0, 1, 1, 1, 0, 1
        )

        self.vertices = np.array(self.vertices, dtype=np.float32)
        self.vertex_count = len(self.vertices) // 8

        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)
        self.vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, self.vertices.nbytes, self.vertices, GL_STATIC_DRAW)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 32, ctypes.c_void_p(0))
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 32, ctypes.c_void_p(12))
        glEnableVertexAttribArray(2)
        glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 32, ctypes.c_void_p(20))

    def destroy(self):
        glDeleteVertexArrays(1, (self.vao,))
        glDeleteBuffers(1, (self.vbo,))


class Mesh:
    def __init__(self, filepath):
        self.vertices = self.loadMesh(filepath)

        self.vertices = np.array(self.vertices, dtype=np.float32)
        self.vertex_count = len(self.vertices) // 5

        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)
        self.vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, self.vertices.nbytes, self.vertices, GL_STATIC_DRAW)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 20, ctypes.c_void_p(0))
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 20, ctypes.c_void_p(12))

    def loadMesh(self, filename):

        # raw, unassembled data
        v = []
        vt = []
        vn = []

        # final, assembled and packed result
        vertices = []

        # open the obj file and read the data
        with open(filename, 'r') as f:
            line = f.readline()
            while line:
                firstSpace = line.find(" ")
                flag = line[0:firstSpace]
                if flag == "v":
                    # vertex
                    line = line.replace("v ", "")
                    line = line.split(" ")
                    l = [float(x) for x in line]
                    v.append(l)
                elif flag == "vt":
                    # texture coordinate
                    line = line.replace("vt ", "")
                    line = line.split(" ")
                    l = [float(x) for x in line]
                    vt.append(l)
                elif flag == "vn":
                    # normal
                    line = line.replace("vn ", "")
                    line = line.split(" ")
                    l = [float(x) for x in line]
                    vn.append(l)
                elif flag == "f":
                    # face, three or more vertices in v/vt/vn form
                    line = line.replace("f ", "")
                    line = line.replace("\n", "")
                    # get the individual vertices for each line
                    line = line.split(" ")
                    faceVertices = []
                    faceTextures = []
                    faceNormals = []
                    for vertex in line:
                        # break out into [v,vt,vn],
                        # correct for 0 based indexing.
                        l = vertex.split("/")
                        position = int(l[0]) - 1
                        faceVertices.append(v[position])
                        texture = int(l[1]) - 1
                        faceTextures.append(vt[texture])
                        normal = int(l[2]) - 1
                        faceNormals.append(vn[normal])
                    # obj file uses triangle fan format for each face individually.
                    # unpack each face
                    triangles_in_face = len(line) - 2

                    vertex_order = []
                    """
                        eg. 0,1,2,3 unpacks to vertices: [0,1,2,0,2,3]
                    """
                    for i in range(triangles_in_face):
                        vertex_order.append(0)
                        vertex_order.append(i + 1)
                        vertex_order.append(i + 2)
                    for i in vertex_order:
                        for x in faceVertices[i]:
                            vertices.append(x)
                        for x in faceTextures[i]:
                            vertices.append(x)
                        for x in faceNormals[i]:
                            vertices.append(x)
                line = f.readline()
        return vertices

    def destroy(self):
        glDeleteVertexArrays(1, (self.vao,))
        glDeleteBuffers(1, (self.vbo,))


class Material:
    def __init__(self, filepath):
        self.texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.texture)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST_MIPMAP_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

        with Image.open(filepath, mode='r') as image:
            image_width, image_height = image.size
            image = image.convert("RGBA")
            image_data = bytes(image.tobytes())
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, image_width, image_height, 0, GL_RGBA, GL_UNSIGNED_BYTE, image_data)
        glGenerateMipmap(GL_TEXTURE_2D)

    def use(self):
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self.texture)

    def destroy(self):
        glDeleteTextures(1, (self.texture,))


if __name__ == "__main__":
    window = initialize_glfw()
    myApp = App(window)
