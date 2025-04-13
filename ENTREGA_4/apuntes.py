
Line(500) = {560, 535};
Line(501) = {535, 536};
Line(502) = {536, 561};
Line(503) = {561, 560};

Line Loop(500) = {500, 501, 502, 503};

Plane Surface(500) = {500};

Physical Surface("RestriccionFija") = {500};

Line(600) = {626, 630};
Line(601) = {630, 605};
Line(602) = {605, 601};
Line(603) = {601, 626};

Line Loop(600) = {600, 601, 602, 603};

Plane Surface(600) = {600};

Physical Surface("RestriccionFija") = {600};















