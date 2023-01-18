import numpy as np

import pygame
from pygame.locals import *

from OpenGL.GL import *
from OpenGL.GLU import *

verticesObj = []
facesObj = []

verticesBezier = []
vertices_b_spline = []
vertices_b_spline_norm = []

tangents_b_spline = []
normals_b_spline = []


def read_bezier(obj_file):
    objFile = open(obj_file, 'r')
    objLines = objFile.readlines()

    for line in objLines:
        if line.startswith("v"):
            x = float(line.split(" ")[1].strip())
            y = float(line.split(" ")[2].strip())
            z = float(line.split(" ")[3].strip())
            element = [x, y, z]
            verticesBezier.append(element)


def read_obj(obj_file):
    objFile = open(obj_file, 'r')
    objLines = objFile.readlines()

    for line in objLines:
        if line.startswith("v"):
            x = float(line.split(" ")[1].strip())
            y = float(line.split(" ")[2].strip())
            z = float(line.split(" ")[3].strip())
            element = [x, y, z]
            verticesObj.append(element)
        if line.startswith("f"):
            x = int(line.split(" ")[1].strip())
            y = int(line.split(" ")[2].strip())
            z = int(line.split(" ")[3].strip())

            a = ((verticesObj[y - 1][1] - verticesObj[x - 1][1] * verticesObj[z - 1][2] - verticesObj[x - 1][2]) -
                 (verticesObj[z - 1][1] - verticesObj[x - 1][1] * verticesObj[y - 1][2] - verticesObj[x - 1][2]))

            b = ((verticesObj[z - 1][0] - verticesObj[x - 1][0] * verticesObj[y - 1][2] - verticesObj[x - 1][2]) -
                 (verticesObj[y - 1][0] - verticesObj[x - 1][0] * verticesObj[z - 1][2] - verticesObj[x - 1][2]))

            c = ((verticesObj[y - 1][0] - verticesObj[x - 1][0] * verticesObj[z - 1][1] - verticesObj[x - 1][1]) -
                 (verticesObj[z - 1][0] - verticesObj[x - 1][0] * verticesObj[y - 1][1] - verticesObj[x - 1][1]))

            element = [x, y, z, a, b, c]
            facesObj.append(element)


def b_spline_calculate():
    global vertices_b_spline_norm
    read_bezier("bezier.obj")
    bezier_matrix = np.matrix([[-1, 3, -3, 1],
                               [3, -6, 3, 0],
                               [-3, 0, 3, 0],
                               [1, 4, 1, 0]])

    bezier_matrix_tan = np.matrix([[-1, 3, -3, 1],
                                   [2, -4, 2, 0],
                                   [-1, 0, 1, 0]])

    bezier_matrix_nor = np.matrix([[-1, 3, -3, 1],
                                   [1, -2, 1, 0]])

    for i in range(len(verticesBezier) - 3):
        for t in range(0, 1000, 1):
            control_vector = np.matrix([verticesBezier[i],
                                        verticesBezier[i + 1],
                                        verticesBezier[i + 2],
                                        verticesBezier[i + 3]])

            t_vector = np.matrix([pow(t / 1000, 3), pow(t / 1000, 2), t / 1000, 1])
            temp_matrix = np.dot(t_vector, bezier_matrix)
            point = np.dot(temp_matrix, control_vector) * (1 / 6)
            vertices_b_spline.append([point[0, 0], point[0, 1], point[0, 2]])

            t_vector_tan = np.matrix([pow(t / 1000, 2), t / 1000, 1])
            temp_matrix_tan = np.dot(t_vector_tan, bezier_matrix_tan)
            point_tan = np.dot(temp_matrix_tan, control_vector) * (1 / 2)
            tangents_b_spline.append([point_tan[0, 0], point_tan[0, 1], point_tan[0, 2]])

            t_vector_nor = np.matrix([t / 1000, 1])
            temp_matrix_nor = np.dot(t_vector_nor, bezier_matrix_nor)
            point_nor = np.dot(temp_matrix_nor, control_vector)
            normals_b_spline.append([point_nor[0, 0], point_nor[0, 1], point_nor[0, 2]])

    temp_matrix = np.matrix(vertices_b_spline)
    norms = np.linalg.norm(temp_matrix)
    tempMatrix = temp_matrix / (norms / 70)
    for element in tempMatrix:
        vertices_b_spline_norm.append([element[0, 0], element[0, 1], element[0, 2]])


def b_spline_draw():
    glColor3f(0.4, 0.3, 0.2)
    glBegin(GL_LINE_STRIP)
    for element in vertices_b_spline_norm:
        glVertex3f(element[0], element[1], element[2])
    glEnd()


def obj_file_draw(dcm_inv, tangent, translate, u, v):
    glColor3f(0.2, 0.3, 0.4)
    translate_matrix = np.matrix([translate, translate, translate])

    for face in facesObj:
        vertexA = np.array((verticesObj[face[0] - 1][0], verticesObj[face[0] - 1][1], verticesObj[face[0] - 1][2]))
        vertexB = np.array((verticesObj[face[1] - 1][0], verticesObj[face[1] - 1][1], verticesObj[face[1] - 1][2]))
        vertexC = np.array((verticesObj[face[2] - 1][0], verticesObj[face[2] - 1][1], verticesObj[face[2] - 1][2]))

        vertices_matrix = np.matrix([vertexA, vertexB, vertexC])
        vertices_matrix_scaled = vertices_matrix * 3
        vertices_matrix_rotated = np.dot(vertices_matrix_scaled, dcm_inv)

        vertices_matrix_rotated = np.dot(vertices_matrix_rotated, np.matrix([[0, -1, 0],
                                                                             [1, 0, 0],
                                                                             [0, 0, 1]]))

        vertices_matrix_translated = vertices_matrix_rotated + translate_matrix

        glBegin(GL_TRIANGLES)
        glVertex3f(vertices_matrix_translated[0, 0], vertices_matrix_translated[0, 1], vertices_matrix_translated[0, 2])
        glVertex3f(vertices_matrix_translated[1, 0], vertices_matrix_translated[1, 1], vertices_matrix_translated[1, 2])
        glVertex3f(vertices_matrix_translated[2, 0], vertices_matrix_translated[2, 1], vertices_matrix_translated[2, 2])
        glEnd()

    tangent = tangent / 1000
    u = u / 1000
    v = v / 1000
    print(tangent)
    print(u)
    print(v)

    glBegin(GL_LINES)
    glColor3f(0.6, 0.6, 0.6)
    glVertex3f(translate[0], translate[1], translate[2])
    glVertex3f(tangent[0] + translate[0], tangent[1] + translate[1], tangent[2] + translate[2])
    glEnd()

    glBegin(GL_LINES)
    glColor3f(0.4, 0.6, 0.6)
    glVertex3f(translate[0], translate[1], translate[2])
    glVertex3f(u[0] + translate[0], u[1] + translate[1], u[2] + translate[2])
    glEnd()

    glBegin(GL_LINES)
    glColor3f(0.6, 0.6, 0.4)
    glVertex3f(translate[0], translate[1], translate[2])
    glVertex3f(v[0] + translate[0], v[1] + translate[1], v[2] + translate[2])
    glEnd()


def main():
    pygame.init()
    display = (1300, 900)
    pygame.display.set_mode(display, DOUBLEBUF | OPENGL)

    gluPerspective(45, (display[0] / display[1]), 0.1, 50.0)
    gluLookAt(0, 0, 3, 0, 0, 0, 0, 1, 0)
    glTranslatef(0.0, 0.0, 2)
    glRotatef(20, 10, 30, 0)

    read_obj("f16.obj")
    b_spline_calculate()

    i = len(vertices_b_spline_norm)//2
    wireframe = 0

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    if wireframe:
                        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
                        wireframe = 0
                    else:
                        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
                        wireframe = 1

            keys = pygame.key.get_pressed()
            if keys[pygame.K_w]:
                glRotatef(-2, 1, 0, 0)
            if keys[pygame.K_s]:
                glRotatef(2, 1, 0, 0)
            if keys[pygame.K_a]:
                glRotatef(-2, 0, 1, 0)
            if keys[pygame.K_d]:
                glRotatef(2, 0, 1, 0)
            if keys[pygame.K_m]:
                i = i + 20
                if i >= len(vertices_b_spline_norm):
                    i = 0
            if keys[pygame.K_n]:
                i = i - 20
                if i <= 0:
                    i = len(vertices_b_spline_norm) - 1

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        glPushMatrix()

        w_vector = tangents_b_spline[i]
        norms = np.linalg.norm(w_vector)
        w_vector = w_vector / (norms / 70)

        u_vector = np.cross(w_vector, normals_b_spline[i])
        norms = np.linalg.norm(u_vector)
        u_vector = u_vector / (norms / 70)

        v_vector = np.cross(w_vector, u_vector)
        norms = np.linalg.norm(v_vector)
        v_vector = v_vector / (norms / 70)

        dcm_matrix = np.transpose(np.matrix([w_vector, u_vector, v_vector]))
        obj_file_draw(np.linalg.inv(dcm_matrix), w_vector, vertices_b_spline_norm[i], u_vector, v_vector)

        glPopMatrix()

        b_spline_draw()
        pygame.display.flip()


main()
