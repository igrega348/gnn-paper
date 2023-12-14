import numpy as np

from . import elasticity_func

def test_Mandel_Voigt():
    # Come up with PSD stiffness matrix in Mandel notation
    C = np.random.rand(6,6)
    C = C + C.T # symmetric
    C = C @ C # PSD
    assert np.all(np.linalg.eigvalsh(C) > 0)
    # also test compliance
    S = np.linalg.inv(C)

    # Convert to Voigt notation
    C_voigt = elasticity_func.stiffness_Mandel_to_Voigt(C)
    S_voigt = np.linalg.inv(C_voigt)
    # ensure backward conversion gives the same result
    assert np.allclose(C, elasticity_func.stiffness_Voigt_to_Mandel(C_voigt))
    assert np.allclose(S, elasticity_func.compliance_Voigt_to_Mandel(S_voigt))

    # Ensure the conversion of Voigt and Mandel to 4th order tensors are the same
    C4_v = elasticity_func.stiffness_Voigt_to_4th_order(C_voigt)
    C4_m = elasticity_func.numpy_Mandel_to_cart_4(C)
    assert np.allclose(C4_v, C4_m)
    S4_v = elasticity_func.compliance_Voigt_to_4th_order(S_voigt)
    S4_m = elasticity_func.numpy_Mandel_to_cart_4(S)
    assert np.allclose(S4_v, S4_m)

    # Ensure the conversion back to Voigt and Mandel match
    assert np.allclose(C_voigt, elasticity_func.stiffness_4th_order_to_Voigt(C4_v))
    assert np.allclose(C, elasticity_func.numpy_cart_4_to_Mandel(C4_m))
    assert np.allclose(S_voigt, elasticity_func.compliance_4th_order_to_Voigt(S4_v))
    assert np.allclose(S, elasticity_func.numpy_cart_4_to_Mandel(S4_m))

def test_rotation_rules():
    # TODO
    pass