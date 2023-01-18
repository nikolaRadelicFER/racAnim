from __future__ import division
from OpenGL.GL import *
import numpy as np
from OpenGL.GL import shaders
import glfw
import glfw.GLFW as GLFW_CONSTANTS
from glfw.GLFW import *
from numpy import double

SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 720

zoomOut = 1.2
zoomIn = 0.8
movementSpeed = 0.002

high_precision = True

x = 0.0
y = 0.0
z = double(1.0)


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


class Main:
    def __init__(self, window):
        self.window = window

        if high_precision == True:
            self.shader = self.createShader("shaders/vertex.txt", "shaders/fragments_double.txt")
        else:
            self.shader = self.createShader("shaders/vertex.txt", "shaders/fragment.txt")

        glUseProgram(self.shader)

        self.resolution = SCREEN_WIDTH, SCREEN_HEIGHT
        glUniform2f(glGetUniformLocation(self.shader, "resolution"), *self.resolution)

        self.vertices = np.array([-1.0, -1.0,
                                  1.0, -1.0,
                                  1.0, 1.0,
                                  -1.0, 1.0
                                  ], dtype='float32')

        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)

        self.vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, self.vertices, GL_STATIC_DRAW)

        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, None)

        self.CenterX = glGetUniformLocation(self.shader, 'CenterX')
        self.CenterY = glGetUniformLocation(self.shader, 'CenterY')
        self.ZoomScale = glGetUniformLocation(self.shader, 'ZoomScale')

    def createShader(self, vertexFilepath, fragmentFilepath):  # funkcija za učitavanje shadera
        with open(vertexFilepath, 'r') as f:
            vertex_src = f.readlines()

        with open(fragmentFilepath, 'r') as f:
            fragment_src = f.readlines()

        shader = shaders.compileProgram(
            shaders.compileShader(vertex_src, GL_VERTEX_SHADER),
            shaders.compileShader(fragment_src, GL_FRAGMENT_SHADER)
        )
        return shader

    def handleScroll(self, window, xoffset, yoffset):  # funkcija za obradu zooma
        global z
        if yoffset == 1:
            z = z * zoomIn
            if z > 1.0:
                z = 1.0
        elif yoffset == -1:
            z = z * zoomOut
            if z > 1.0:
                z = 1.0

    def handlePosition(self):
        global x, y, z

        mouseX, mouseY = glfw.get_cursor_pos(self.window)  # dohvati položaj miša
        if glfwGetMouseButton(self.window, GLFW_MOUSE_BUTTON_LEFT):  # mijenjaj položaj samo ako je lijevi gumb stisnut
            x = x + (SCREEN_WIDTH / 2 - mouseX) * z * 0.001
            y = y - (SCREEN_HEIGHT / 2 - mouseY) * z * 0.001
        glfw.set_cursor_pos(self.window, SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2)  # postavi miš na centar ekrana

    def mainLoop(self):
        global x, y, z
        glfw.set_scroll_callback(self.window, self.handleScroll)

        while not glfw.window_should_close(self.window) and not glfw.get_key(self.window, GLFW_CONSTANTS.GLFW_KEY_ESCAPE) == GLFW_CONSTANTS.GLFW_PRESS:
            glClearColor(0.0, 0.0, 0.0, 1.0)
            glClear(GL_COLOR_BUFFER_BIT)

            # dohvati sve događaje
            glfw.poll_events()

            # obradi položaj miša
            self.handlePosition()

            # pošalji varijable shaderu
            if high_precision == True:
                glUniform1d(self.CenterX, x)
                glUniform1d(self.CenterY, y)
                glUniform1d(self.ZoomScale, z)
            else:
                glUniform1f(self.CenterX, x)
                glUniform1f(self.CenterY, y)
                glUniform1f(self.ZoomScale, z)

            # crtaj
            glDrawArrays(GL_TRIANGLE_FAN, 0, 4)

            glFlush()

        glfw.terminate()


if __name__ == '__main__':
    window = initialize_glfw()
    Main(window).mainLoop()
