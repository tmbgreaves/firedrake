import abc
from pyop2 import op2
from pyop2.utils import as_tuple
from firedrake.petsc import PETSc


class MatrixBase(object, metaclass=abc.ABCMeta):
    """A representation of the linear operator associated with a
    bilinear form and bcs.  Explicitly assembled matrices and matrix-free
    matrix classes will derive from this

    :arg a: the bilinear form this :class:`MatrixBase` represents.

    :arg bcs: an iterable of boundary conditions to apply to this
        :class:`MatrixBase`.  May be `None` if there are no boundary
        conditions to apply.
    :arg mat_type: matrix type of assembled matrix, or 'matfree' for matrix-free
    """
    def __init__(self, a, bcs, mat_type):
        self.a = a
        # Iteration over bcs must be in a parallel consistent order
        # (so we can't use a set, since the iteration order may differ
        # on different processes)
        self.bcs = bcs
        test, trial = a.arguments()
        self.comm = test.function_space().comm
        self.block_shape = (len(test.function_space()),
                            len(trial.function_space()))
        self.mat_type = mat_type
        """Matrix type.

        Matrix type used in the assembly of the PETSc matrix: 'aij', 'baij', or 'nest',
        or 'matfree' for matrix-free."""

    @property
    def bcs(self):
        return self._bcs

    @bcs.setter
    def bcs(self, bcs):
        self._bcs = tuple(bc for bc in bcs) if bcs is not None else ()
        self.has_bcs = self._bcs != ()

    @abc.abstractmethod
    def assemble(self):
        """Ensures that the matrix is assembled and populated with
        values, ready for sending to PETSc solvers."""
        pass

    def __repr__(self):
        return "%s(a=%r, bcs=%r)" % (type(self).__name__,
                                     self.a,
                                     self.bcs)

    def __str__(self):
        return "%s(a=%s, bcs=%s)" % (type(self).__name__,
                                     self.a, self.bcs)


class Matrix(MatrixBase):
    """A representation of an assembled bilinear form.

    :arg a: the bilinear form this :class:`Matrix` represents.

    :arg bcs: an iterable of boundary conditions to apply to this
        :class:`Matrix`.  May be `None` if there are no boundary
        conditions to apply.

    :arg mat_type: matrix type of assembled matrix.

    A :class:`pyop2.Mat` will be built from the remaining
    arguments, for valid values, see :class:`pyop2.Mat`.

    .. note::

        This object acts to the right on an assembled :class:`.Function`
        and to the left on an assembled cofunction (currently represented
        by a :class:`.Function`).

    """

    def __init__(self, a, bcs, mat_type, *args, **kwargs):
        # sets self._a, self._bcs, and self._mat_type
        super().__init__(a, bcs, mat_type)
        options_prefix = kwargs.pop("options_prefix")
        self.M = op2.Mat(*args, **kwargs)
        self.petscmat = self.M.handle
        self.petscmat.setOptionsPrefix(options_prefix)
        self.mat_type = mat_type

    def assemble(self):
        "Ensures that the matrix is fully assembled."
        self.M.assemble()


class ImplicitMatrix(MatrixBase):
    """A representation of the action of bilinear form operating
    without explicitly assembling the associated matrix.  This class
    wraps the relevant information for Python PETSc matrix.

    :arg a: the bilinear form this :class:`Matrix` represents.

    :arg bcs: an iterable of boundary conditions to apply to this
        :class:`Matrix`.  May be `None` if there are no boundary
        conditions to apply.


    .. note::

        This object acts to the right on an assembled :class:`.Function`
        and to the left on an assembled cofunction (currently represented
        by a :class:`.Function`).

    """
    def __init__(self, a, bcs, *args, **kwargs):
        super().__init__(a, bcs, "matfree")

        options_prefix = kwargs.pop("options_prefix")
        appctx = kwargs.get("appctx", {})

        from firedrake.matrix_free.operators import ImplicitMatrixContext
        ctx = ImplicitMatrixContext(a,
                                    row_bcs=self.bcs,
                                    col_bcs=self.bcs,
                                    fc_params=kwargs["fc_params"],
                                    appctx=appctx)
        self.petscmat = PETSc.Mat().create(comm=self.comm)
        self.petscmat.setType("python")
        self.petscmat.setSizes((ctx.row_sizes, ctx.col_sizes),
                               bsize=ctx.block_size)
        self.petscmat.setPythonContext(ctx)
        self.petscmat.setOptionsPrefix(options_prefix)
        self.petscmat.setUp()

    def update_bcs(self, bcs):
        bcs = as_tuple(bcs)
        if set(self.bcs) == set(bcs):
            return
        self.bcs = bcs
        ctx = self.petscmat.getPythonContext()
        ctx.update_bcs(bcs, bcs)

    def assemble(self):
        self.petscmat.assemble()
