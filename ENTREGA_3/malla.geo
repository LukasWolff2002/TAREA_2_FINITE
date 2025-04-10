SetFactory("Built-in");

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

Plane Surface(2) = {2000};  // sin agujeros internos
Plane Surface(3) = {2000, 1000};

// === LLAVE 17 ===

Point(300) = {-65, -5, 0, 1.0};
Point(301) = {-59.735, 7.55, 0, 1.0};
Point(302) = {-51.936, -7.526, 0, 1.0};
Point(303) = {-70.898, 1.806, 0, 1.0};
Point(304) = {-63.542, -13.529, 0, 1.0};

Line(301) = {301, 303};
Line(302) = {302, 304};
Circle(303) = {302,300,301};
Circle(304) = {303, 209, 203};
Circle(305) = {304, 209, 204};

Line Loop(3000) = {301, 304, -205, -305, -302, 303};  // ¡Corregido!
Plane Surface(4) = {3000};

// === LLAVE 13 ===

Point(400) = {61.753, 4.113, 0, 1.0};
Point(401) = {60.44, -4.792, 0, 1.0};
Point(402) = {52.912, 5.799, 0, 1.0};
Point(403) = {69.422, 1.593, 0, 1.0};
Point(404) = {62.341, 12.51, 0, 1.0};

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

// === Agrupación final de todos los volúmenes en un solo grupo físico ===
Physical Volume("CuerpoCompleto") = {
  vol1[1], vol2[1],
  vol3[1], vol4[1],
  vol5[1], vol6[1],
  vol7[1], vol8[1]
};

SetFactory("Built-in");


