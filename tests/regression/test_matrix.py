from firedrake import *
from firedrake import matrix
import pytest


@pytest.fixture
def V():
    mesh = UnitIntervalMesh(2)
    V = FunctionSpace(mesh, "CG", 1)
    return V


@pytest.fixture
def a(V):
    u = TrialFunction(V)
    v = TestFunction(V)

    return u*v*dx


@pytest.fixture(params=["nest", "aij", "matfree"])
def mat_type(request):
    return request.param


def test_assemble_returns_matrix(a):
    A = assemble(a)

    assert isinstance(A, matrix.Matrix)


def test_assemble_with_bcs_then_not(a, V):
    bc1 = DirichletBC(V, 0, 1)
    A = assemble(a, bcs=[bc1])
    Abcs = A.M.values

    A = assemble(a)
    assert not A.has_bcs
    Anobcs = A.M.values

    assert (Anobcs != Abcs).any()

    A = assemble(a, bcs=[bc1])
    Abcs = A.M.values
    assemble(a, tensor=A)
    Anobcs = A.M.values
    assert not A.has_bcs
    assert (Anobcs != Abcs).any()
