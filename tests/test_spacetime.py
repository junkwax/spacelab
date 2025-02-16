def test_schwarzschild_metric():
    spacetime = SpacetimeGeometry(mass=10)
    r = 1e7
    g_tt, g_rr, g_theta_theta, g_phi_phi = spacetime.schwarzschild_metric(r)
    
    # Compute the correct expected values
    rs = 2 * spacetime.G * (spacetime.mass * 1.98847e30) / (spacetime.c**2)
    expected_g_tt = -(1 - rs / r)
    expected_g_rr = 1 / (1 - rs / r)

    assert np.isclose(g_tt, expected_g_tt, atol=1e-6)  # Use computed expected value
    assert np.isclose(g_rr, expected_g_rr, atol=1e-6)
    assert np.isclose(g_theta_theta, r**2)
    assert np.isclose(g_phi_phi, r**2)
