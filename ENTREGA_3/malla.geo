SetFactory("Built-in");

// === PARÁMETRO DE DISTANCIA PARA PUNTOS INTERMEDIOS ===
d = 16.0;

// === INTERIOR CENTRAL ===
Point(101) = {-27.5, 3.25, 0, 1.0};
Point(102) = {27.5, 3.25, 0, 1.0};
Point(103) = {27.5, -3.25, 0, 1.0};
Point(104) = {-27.5, -3.25, 0, 1.0};
Point(105) = {-27.5, 0, 0 ,1.0};
Point(106) = {27.5, 0, 0 ,1.0};

Circle(101) = {101, 105, 104};      // arco izquierdo: arriba → abajo
Line(102) = {104, 103};             // abajo recto
Circle(103) = {103, 106, 102};      // arco derecho: abajo → arriba
Line(104) = {102, 101};             // arriba recto

Line Loop(1000) = {101, 102, 103, 104};
Plane Surface(1) = {1000};

// === EXTERIOR ===
Point(201) = {-20.18, 5.5, 0, 1.0};
Point(202) = {-20.18, -5.5, 0, 1.0};
Point(203) = {-46.362, 13.468, 0, 1.0};
Point(204) = {-46.362, -13.468, 0, 1.0};
Point(205) = {30.638, 5.5, 0, 1.0};
Point(206) = {30.638, -5.5, 0, 1.0};
Point(207) = {-20.18, 52.5, 0, 1.0};
Point(208) = {-20.18, -52.5, 0, 1.0};
Point(209) = {-55, 0, 0, 1.0};
Point(210) = {55, 0, 0, 1.0};
Point(211) = {30.638, 28.5, 0, 1.0};
Point(212) = {30.638, -28.5, 0, 1.0};
Point(214) = {45.586, 11.02, 0, 1.0};
Point(215) = {45.586, -11.02, 0, 1.0};

Circle(201) = {203, 207, 201};
Circle(202) = {202, 208, 204};
Line(203) = {201, 205};
Line(204) = {206, 202};
Circle(205) = {204, 209, 203};
Circle(206) = {205, 211, 214};
Circle(207) = {206, 212, 215};
Circle(208) = {214, 210, 215};

Line Loop(2000) = {
  201,
  203,
  206,
  208,
  -207,
  204,
  202,
  205
};

Plane Surface(3) = {2000, 1000};

// === LLAVE 17 ===
Point(300) = {-61, -2, 0, 1.0};
Point(301) = {-54.347, 9.174, 0, 1.0};
Point(302) = {-48.981, -6.953, 0, 1.0};
Point(303) = {-70.548, 3.777, 0, 1.0};
Point(304) = {-65.172, -12.350, 0, 1.0};

// --- Punto intermedio entre 303 y 301 (origen: 303) ---
dx = -54.347 - (-70.548);
dy = 9.174 - 3.777;
L = Sqrt(dx^2 + dy^2);
x_target = -70.548 + d * dx / L;
y_target = 3.777 + d * dy / L;
Point(305) = {x_target, y_target, 0, 1.0};

// --- Punto intermedio entre 304 y 302 (origen: 304) ---
dx2 = -48.981 - (-65.172);
dy2 = -6.953 - (-12.350);
L2 = Sqrt(dx2^2 + dy2^2);
x_target2 = -65.172 + d * dx2 / L2;
y_target2 = -12.350 + d * dy2 / L2;
Point(306) = {x_target2, y_target2, 0, 1.0};

// --- Geometría de la llave 17 ---
Line(301) = {301, 305};
Line(306) = {305, 303};
Line(302) = {302, 306};
Line(307) = {306, 304};
Circle(303) = {302, 300, 301};
Circle(304) = {303, 209, 203};
Circle(305) = {304, 209, 204};

Line Loop(3000) = {301, 306, 304, -205, -305, -302, -307, 303};
Plane Surface(4) = {3000};

// === LLAVE 13 ===
Point(400) = {61, 2, 0, 1.0};
Point(401) = {57.151, -6.135, 0, 1.0};
Point(402) = {53.049, 6.198, 0, 1.0};
Point(403) = {69.352, -2.068, 0, 1.0};
Point(404) = {65.251, 10.265, 0, 1.0};

Line(402) = {402, 404};
Circle(404) = {404, 210, 214};
// Reutiliza Circle(208): 214 → 215
Circle(405) = {403, 210, 215};
Line(401) = {401, 403};
Circle(403) = {402, 400, 401};

Line Loop(4000) = {402, 404, 208, -405, -401, -403};
Plane Surface(5) = {4000};

// === GRUPOS FÍSICOS ===
Physical Surface(1) = {1};  // Superficie central interna
Physical Surface(2) = {3};  // Superficie exterior con hueco interior
Physical Surface(3) = {4};  // Superficie de la llave izquierda
Physical Surface(4) = {5};  // Superficie de la llave derecha

Physical Line(10) = {306, 307};  // Borde fijo de la llave izquierda

Physical Line(11) = {404};  // Arco con carga distribuida

// Se eligio un contacto de 16 mm ya que la tuerca usada es de 17 mm de espesor, de esta manera se deja un margen de 1 mm
