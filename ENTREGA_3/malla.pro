Group {
  Vol = Region[1, 2, 3, 4];        // Todas las superficies físicas (2D)
  Fijado = Region[10];            // Línea con restricciones
  Carga = Region[11];             // Línea con fuerza
}

Function {
  E = 200e9;       // Módulo de Young (Pa)
  nu = 0.3;        // Coeficiente de Poisson
  fx = 0;
  fy = -1e6;       // Fuerza en Y negativa (N/m)
}

Constraint {
  { Name DirichletX; Type Assign; Case {
      { Region Fijado; Value 0; }
  }; }
  { Name DirichletY; Type Assign; Case {
      { Region Fijado; Value 0; }
  }; }
}

Formulation {
  { Name Elasticity; Type FemEquation;
    Quantity {
      { Name u; Type Vector; NameOfSpace V; }
    }
    Equation {
      Galerkin { [ sigma[E, nu] : Dof{u} , {u} ]; In Vol; Integration Int; }
      Galerkin { [ fx , {u_x} ]; In Carga; Integration Int; }
      Galerkin { [ fy , {u_y} ]; In Carga; Integration Int; }
    }
  }
}

Integration {
  { Name Int; Type Gauss; Order 2; }
}

FunctionSpace {
  { Name V; Type Vector;
    BasisFunction {
      { Name sn; NameOfCoef vn; Function BF_Node; }
    }
    Constraint {
      { NameOfCoef vn; EntityType Edges; NameOfConstraint DirichletX; }
      { NameOfCoef vn; EntityType Edges; NameOfConstraint DirichletY; }
    }
  }
}

Resolution {
  { Name Res; System {
      { Name S; NameOfFormulation Elasticity; }
    }
    Operation {
      Generate[S]; Solve[S]; SaveSolution[S];
    }
  }
}

PostProcessing {
  { Name Desplazamientos; NameOfFormulation Elasticity;
    Quantity {
      { Name u; Value { Local { [ {u} ]; In Vol; } } }
    }
  }
}

PostOperation {
  { Name Desplazamientos; NameOfPostProcessing Desplazamientos;
    Operation {
      Print[u, OnElementsOf Vol, File "resultados.pos"];
    }
  }
}
