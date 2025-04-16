import pygame
import math
import numpy as np

# Parámetros físicos del sistema
PIXELES_POR_METRO = 100
RADIO_RUEDA = 0.1
LONGITUD_EJE = 0.5
KV_MOTOR = 150          # Constante de velocidad del motor (RPM/Voltio)
VOLTAJE_MAXIMO = 12     # Voltaje máximo aplicable a los motores

# Paleta de colores
BLANCO = (255, 255, 255)
GRIS = (230, 230, 230)
NEGRO = (0, 0, 0)
ROJO = (255, 50, 50)
AZUL = (50, 100, 255)
VERDE = (0, 150, 0)
MORADO = (150, 0, 150)

# Dimensiones de la zona de simulación
ANCHO_SIM = 700
ALTO_SIM = 600
AREA_LIMITE = pygame.Rect(50, 50, ANCHO_SIM - 100, ALTO_SIM - 100)

# Configuración de gráficas
ANCHO_GRAFICO = 180
ALTO_GRAFICO = 60
ESPACIADO_GRAFICOS = 10
NUM_GRAFICAS = 5
ANCHO_AREA_GRAFICOS = ANCHO_GRAFICO + 40
ANCHO_PANTALLA = ANCHO_SIM + ANCHO_AREA_GRAFICOS
ALTO_PANTALLA = ALTO_SIM

class RobotDiferencial:
    def __init__(self, x, y, orientacion):
        self.x = x          # Posición en píxeles (eje X)
        self.y = y          # Posición en píxeles (eje Y)
        self.orientacion = orientacion  # Orientación en radianes
        self.voltaje_izquierdo = 0
        self.voltaje_derecho = 0
        self.factor_rpm_a_rads = 2 * math.pi / 60
        self.velocidad_lineal = 0
        self.trayectoria = []
        self.objetivo_alcanzado = False
        self.objetivo = None
        self.historial = {
            'velocidad': [], 'velocidad_angular': [],
            'orientacion': [], 'voltaje_izquierdo': [], 'voltaje_derecho': []
        }

    def voltaje_a_velocidad(self, voltaje):
        """Convierte voltaje a velocidad angular en radianes/segundo"""
        rpm = KV_MOTOR * (voltaje / VOLTAJE_MAXIMO) * VOLTAJE_MAXIMO
        return rpm * self.factor_rpm_a_rads

    def actualizar(self, dt):
        """Actualiza el estado del robot basado en la física"""
        # Calcular velocidades de las ruedas
        vel_izq = self.voltaje_a_velocidad(self.voltaje_izquierdo) * RADIO_RUEDA
        vel_der = self.voltaje_a_velocidad(self.voltaje_derecho) * RADIO_RUEDA

        # Cinemática diferencial
        velocidad = (vel_der + vel_izq) / 2
        velocidad_angular = (vel_der - vel_izq) / LONGITUD_EJE

        # Actualizar posición y orientación
        dx = velocidad * math.cos(self.orientacion) * dt
        dy = velocidad * math.sin(self.orientacion) * dt
        self.x += dx * PIXELES_POR_METRO
        self.y += dy * PIXELES_POR_METRO
        self.orientacion += velocidad_angular * dt

        self.velocidad_lineal = velocidad
        if not self.objetivo_alcanzado:
            self.trayectoria.append((self.x, self.y))

        # Registrar datos históricos
        self.historial['velocidad'].append(velocidad)
        self.historial['velocidad_angular'].append(velocidad_angular)
        self.historial['orientacion'].append(self.orientacion)
        self.historial['voltaje_izquierdo'].append(self.voltaje_izquierdo)
        self.historial['voltaje_derecho'].append(self.voltaje_derecho)

        # Mantener un historial limitado
        for clave in self.historial:
            self.historial[clave] = self.historial[clave][-200:]

    def establecer_voltajes(self, izquierdo, derecho):
        """Establece los voltajes de los motores con límites"""
        self.voltaje_izquierdo = np.clip(izquierdo, -VOLTAJE_MAXIMO, VOLTAJE_MAXIMO)
        self.voltaje_derecho = np.clip(derecho, -VOLTAJE_MAXIMO, VOLTAJE_MAXIMO)

    def establecer_objetivo(self, objetivo):
        """Define un nuevo objetivo de navegación"""
        tx, ty = objetivo
        tx = np.clip(tx, AREA_LIMITE.left, AREA_LIMITE.right)
        ty = np.clip(ty, AREA_LIMITE.top, AREA_LIMITE.bottom)
        self.trayectoria = []
        self.objetivo_alcanzado = False
        self.objetivo = (tx, ty)

    def algoritmo_control(self, objetivo):
        """Algoritmo de control para navegación autónoma"""
        UMBRAL_DETENCION = 0.05  # 5 cm
        tx = objetivo[0] / PIXELES_POR_METRO
        ty = objetivo[1] / PIXELES_POR_METRO
        x = self.x / PIXELES_POR_METRO
        y = self.y / PIXELES_POR_METRO

        dx = tx - x
        dy = ty - y
        distancia = math.hypot(dx, dy)

        if distancia < UMBRAL_DETENCION:
            self.establecer_voltajes(0, 0)
            self.objetivo_alcanzado = True
            return

        # Cálculo de error angular
        orientacion_deseada = math.atan2(dy, dx)
        error_angular = orientacion_deseada - self.orientacion
        error_angular = (error_angular + math.pi) % (2 * math.pi) - math.pi

        # Parámetros del controlador
        KP_DISTANCIA = 4.0      # Ganancia proporcional para distancia
        KP_ANGULO = 6.0         # Ganancia proporcional para ángulo
        VELOCIDAD_MAXIMA = 3.0  # Velocidad máxima en m/s

        # Cálculo de velocidades
        velocidad_base = np.clip(KP_DISTANCIA * distancia, -VELOCIDAD_MAXIMA, VELOCIDAD_MAXIMA)
        correccion_angular = KP_ANGULO * error_angular

        voltaje_izq = velocidad_base - correccion_angular
        voltaje_der = velocidad_base + correccion_angular
        self.establecer_voltajes(voltaje_izq, voltaje_der)

class Simulacion:
    def __init__(self):
        pygame.init()
        self.pantalla = pygame.display.set_mode((ANCHO_PANTALLA, ALTO_PANTALLA))
        pygame.display.set_caption("Simulación de Robot Diferencial + Recorrido por Puntos")
        self.reloj = pygame.time.Clock()
        self.fuente = pygame.font.SysFont("Arial", 14)

        self.imagen_robot = pygame.image.load("carrito.png").convert_alpha()
        self.imagen_robot = pygame.transform.scale(self.imagen_robot, (50, 30))

        self.robot = RobotDiferencial(AREA_LIMITE.centerx, AREA_LIMITE.centery, 0)
        self.ejecutando = True
        self.objetivo = None
        self.modo = 'manual'

        self.lista_objetivos = []
        self.indice_objetivo = 0

        self.x_input = 0
        self.y_input = 0
        self.theta_input = 0

        self.input_rect_x = pygame.Rect(ANCHO_SIM + 20, 20, 100, 30)
        self.input_rect_y = pygame.Rect(ANCHO_SIM + 20, 60, 100, 30)
        self.input_rect_theta = pygame.Rect(ANCHO_SIM + 20, 100, 100, 30)
        self.botón_dirigir = pygame.Rect(ANCHO_SIM + 20, 140, 100, 30)

    def dibujar_rejilla(self, espaciado=25):
        for x in range(AREA_LIMITE.left, AREA_LIMITE.right, espaciado):
            pygame.draw.line(self.pantalla, (210, 210, 210), (x, AREA_LIMITE.top), (x, AREA_LIMITE.bottom))
        for y in range(AREA_LIMITE.top, AREA_LIMITE.bottom, espaciado):
            pygame.draw.line(self.pantalla, (210, 210, 210), (AREA_LIMITE.left, y), (AREA_LIMITE.right, y))

    def dibujar_limites(self):
        pygame.draw.rect(self.pantalla, GRIS, AREA_LIMITE)
        self.dibujar_rejilla()
        pygame.draw.rect(self.pantalla, NEGRO, AREA_LIMITE, 2)

    def dibujar_robot(self):
        angulo_visual = -math.degrees(self.robot.orientacion)
        imagen_rotada = pygame.transform.rotate(self.imagen_robot, angulo_visual)
        rect = imagen_rotada.get_rect(center=(int(self.robot.x), int(self.robot.y)))
        self.pantalla.blit(imagen_rotada, rect)

    def dibujar_trayectoria(self):
        if len(self.robot.trayectoria) > 1:
            puntos = [(int(x), int(y)) for x, y in self.robot.trayectoria]
            pygame.draw.aalines(self.pantalla, AZUL, False, puntos, 2)

        # Dibujar los puntos objetivo
        for i, punto in enumerate(self.lista_objetivos):
            pygame.draw.circle(self.pantalla, ROJO, punto, 5)
            texto = self.fuente.render(str(i + 1), True, NEGRO)
            self.pantalla.blit(texto, (punto[0] + 6, punto[1] - 6))

    def dibujar_info(self):
        texto_pos = self.fuente.render(
            f"Posición: ({self.robot.x/PIXELES_POR_METRO:.2f}, {self.robot.y/PIXELES_POR_METRO:.2f}) m", True, NEGRO)
        texto_vel = self.fuente.render(
            f"Velocidad: {self.robot.velocidad_lineal:.2f} m/s", True, NEGRO)
        self.pantalla.blit(texto_pos, (10, 10))
        self.pantalla.blit(texto_vel, (10, 30))

    def dibujar_graficas(self):
        espacio_extra = 10
        altura_total = NUM_GRAFICAS * (ALTO_GRAFICO + ESPACIADO_GRAFICOS + espacio_extra) - ESPACIADO_GRAFICOS
        y_inicial = (ALTO_SIM - altura_total) // 2
        x_inicial = ANCHO_SIM + 20

        config_graficas = [
            ("Velocidad (m/s)", ROJO, 'velocidad'),
            ("Vel. Angular (rad/s)", AZUL, 'velocidad_angular'),
            ("Orientación (rad)", NEGRO, 'orientacion'),
            ("Voltaje Izq (V)", VERDE, 'voltaje_izquierdo'),
            ("Voltaje Der (V)", MORADO, 'voltaje_derecho')
        ]

        for i, (etiqueta, color, clave) in enumerate(config_graficas):
            y = y_inicial + i * (ALTO_GRAFICO + ESPACIADO_GRAFICOS + espacio_extra)
            self.dibujar_grafica_individual(
                x_inicial, y, ANCHO_GRAFICO, ALTO_GRAFICO,
                self.robot.historial[clave], color, etiqueta)

    def dibujar_grafica_individual(self, x, y, ancho, alto, datos, color, etiqueta):
        sombra_color = (200, 200, 200)
        pygame.draw.rect(self.pantalla, sombra_color, (x + 2, y + 2, ancho, alto))
        pygame.draw.rect(self.pantalla, BLANCO, (x, y, ancho, alto))
        pygame.draw.rect(self.pantalla, NEGRO, (x, y, ancho, alto), 1)

        if len(datos) > 1:
            max_valor = max(abs(d) for d in datos) or 1
            escala_y = alto / (2 * max_valor)
            escala_x = ancho / len(datos)
            puntos = [(x + i * escala_x, y + alto / 2 - d * escala_y) for i, d in enumerate(datos)]
            pygame.draw.aalines(self.pantalla, color, False, puntos, 2)

        texto = self.fuente.render(etiqueta, True, NEGRO)
        self.pantalla.blit(texto, (x + 5, y - 18))

    def manejar_entradas(self):
        teclas = pygame.key.get_pressed()
        izq, der = 0, 0
        if teclas[pygame.K_UP]: izq = der = 6
        if teclas[pygame.K_DOWN]: izq = der = -6
        if teclas[pygame.K_LEFT]: izq -= 4; der += 4
        if teclas[pygame.K_RIGHT]: izq += 4; der -= 4

        if self.modo == 'manual':
            self.robot.establecer_voltajes(izq, der)

    def manejar_eventos(self, evento):
        if evento.type == pygame.MOUSEBUTTONDOWN:
            if evento.pos[0] < ANCHO_SIM:
                self.lista_objetivos.append(evento.pos)
                if not self.objetivo:
                    self.objetivo = self.lista_objetivos[0]
                    self.robot.establecer_objetivo(self.objetivo)
                    self.indice_objetivo = 0
                    self.modo = 'recorrido'

            if self.botón_dirigir.collidepoint(evento.pos):
                try:
                    x = float(self.x_input)
                    y = float(self.y_input)
                    theta = float(self.theta_input) * math.pi / 180
                    punto = (x * PIXELES_POR_METRO, y * PIXELES_POR_METRO)
                    self.lista_objetivos = [punto]
                    self.indice_objetivo = 0
                    self.objetivo = punto
                    self.robot.orientacion = theta
                    self.robot.establecer_objetivo(punto)
                    self.modo = 'recorrido'
                except ValueError:
                    print("Error en los inputs.")

    def ejecutar(self):
        while self.ejecutando:
            dt = self.reloj.tick(60) / 1000.0
            self.pantalla.fill(BLANCO)

            for evento in pygame.event.get():
                if evento.type == pygame.QUIT:
                    self.ejecutando = False
                self.manejar_eventos(evento)

            if self.modo == 'recorrido' and self.objetivo:
                self.robot.algoritmo_control(self.objetivo)
                if self.robot.objetivo_alcanzado:
                    self.indice_objetivo += 1
                    if self.indice_objetivo < len(self.lista_objetivos):
                        self.objetivo = self.lista_objetivos[self.indice_objetivo]
                        self.robot.establecer_objetivo(self.objetivo)
                    else:
                        self.objetivo = None
                        self.lista_objetivos = []
                        self.modo = 'manual'

            self.manejar_entradas()
            self.robot.actualizar(dt)
            self.robot.x = np.clip(self.robot.x, AREA_LIMITE.left, AREA_LIMITE.right)
            self.robot.y = np.clip(self.robot.y, AREA_LIMITE.top, AREA_LIMITE.bottom)

            self.dibujar_limites()
            self.dibujar_trayectoria()
            self.dibujar_robot()
            self.dibujar_info()
            self.dibujar_graficas()

            if self.objetivo and AREA_LIMITE.collidepoint(self.objetivo):
                pygame.draw.circle(self.pantalla, AZUL, self.objetivo, 5)

            pygame.display.flip()

        pygame.quit()

if __name__ == "__main__":
    simulador = Simulacion()
    simulador.ejecutar()
