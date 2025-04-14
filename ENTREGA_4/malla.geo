SetFactory("Built-in");

// === PARÁMETRO DE DISTANCIA PARA PUNTOS INTERMEDIOS ===
d = 16.0;

Point(101) = {-27.5, 3.25, 0, 1.0};     // arriba izquierda
Point(102) = {27.5, 3.25, 0, 1.0};      // arriba derecha
Point(103) = {27.5, -3.25, 0, 1.0};     // abajo derecha
Point(104) = {-27.5, -3.25, 0, 1.0};    // abajo izquierda
Point(105) = {-27.5, 0, 0 ,1.0};        // centro izquierda
Point(106) = {27.5, 0, 0 ,1.0};         // centro derecha

Circle(101) = {101, 105, 104};          // arco izquierdo: arriba → abajo
Line(102) = {104, 103};                 // abajo recto
Circle(103) = {103, 106, 102};          // arco derecho: abajo → arriba
Line(104) = {102, 101};                 // arriba recto

Line Loop(1000) = {101, 102, 103, 104}; // ordenado correctamente
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

Circle(201) = {203, 207, 201}; // arco arriba izquierda
Circle(202) = {202, 208, 204}; // arco abajo izquierda
Line(203) = {201, 205};
Line(204) = {206, 202};
Circle(205) = {204, 209, 203}; // semicirculo izquierda
Circle(206) = {205, 211, 214}; // arco arriba derecha
Circle(207) = {206, 212, 215}; // arco abajo derecha
Circle(208) = {214, 210, 215}; // semicirculo derecha

Line Loop(2000) = {
  201,        // 203 → 201
  203,        // 201 → 205
  206,        // 205 → 214
  208,        // 214 → 215
  -207,       // 215 → 206
  204,        // 206 → 202
  202,        // 202 → 204
  205         // 204 → 203
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

// Desde 402 → 404 → 214 → 215 → 403 → 401 → 402
Line(402) = {402, 404};             // 402 → 404
Circle(404) = {404, 210, 214};      // 404 → 214
// Circle(208) = {214, 210, 215};      // 214 → 215
Circle(405) = {403, 210, 215};      // 403 → 215, necesitamos invertir → -405
Line(401) = {401, 403};             // 401 → 403 → necesitamos invertir → -401
Circle(403) = {402, 400, 401};      // 402 → 401 → necesitamos invertir → -403



Line Loop(4000) = {402, 404, 208, -405, -401, -403};
Plane Surface(5) = {4000};

// === Extrusión simétrica de Superficie 1 ===
vol1[] = Extrude {0, 0, -0.8} { Surface{1}; };
vol2[] = Extrude {0, 0,  0.8} { Surface{1}; };
Physical Volume("Extrusion1") = {vol1[1]};
Physical Volume("Extrusion2") = {vol2[1]};

// === Extrusión simétrica de Superficie 3 ===
vol3[] = Extrude {0, 0, -1.5} { Surface{3}; };
vol4[] = Extrude {0, 0,  1.5} { Surface{3}; };
Physical Volume("Extrusion3") = {vol3[1]};
Physical Volume("Extrusion4") = {vol4[1]};

// === Extrusión simétrica de Superficie 4 ===
vol5[] = Extrude {0, 0, -2.5} { Surface{4}; };
vol6[] = Extrude {0, 0,  2.5} { Surface{4}; };
Physical Volume("Extrusion5") = {vol5[1]};
Physical Volume("Extrusion6") = {vol6[1]};

// === Extrusión simétrica de Superficie 5 ===
vol7[] = Extrude {0, 0, -2.5} { Surface{5}; };
vol8[] = Extrude {0, 0,  2.5} { Surface{5}; };
Physical Volume("Extrusion7") = {vol7[1]};
Physical Volume("Extrusion8") = {vol8[1]};


Line(1000) = {569, 573};
Line(1001) = {573, 540};
Line(1002) = {540, 536};
Line(1003) = {536, 569};

Line Loop(6000) = {1000, 1001, 1002, 1003};
Plane Surface(6) = {6000};

Line(2000) = {588, 592};
Line(2001) = {592, 559};
Line(2002) = {559, 555};
Line(2003) = {555, 588};

Line Loop(7000) = {2000, 2001, 2002, 2003};
Plane Surface(7) = {7000};

Circle(3000) = {627, 631, 632};
Line(3001) = {632, 607};
Circle(3002) = {607,606, 602};
Line(3003) = {602, 627};

Line Loop(8000) = {3000, 3001, 3002, 3003};
Plane Surface(8) = {8000};

Physical Surface("RestriccionFija") = {6};
Physical Surface("RestriccionFija2") = {7};

Physical Surface("Fuerza") = {8};

// El peso se estima en 8.61 g a partir del archivo STL




