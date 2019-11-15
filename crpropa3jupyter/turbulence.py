corr_lenght_finite = (
    lambda n, Lbo, Lmax, Lmin: (2 * np.pi) ** n
    * (n - 1)
    / (2 * n * (n + 2))
    * (
        Lmax ** n
        / Lbo ** (n - 1)
        * (2 * np.pi ** 2 * (n + 2) + Lmax ** 2 / Lbo ** 2)
        / (4 * np.pi ** 2 + Lmax ** 2 / Lbo ** 2) ** (n / 2 + 1)
        - Lmin ** n
        / Lbo ** (n - 1)
        * (2 * np.pi ** 2 * (n + 2) + Lmin ** 2 / Lbo ** 2)
        / (4 * np.pi ** 2 + Lmin ** 2 / Lbo ** 2) ** (n / 2 + 1)
    )
    / (
        (Lmax / Lbo) ** (n - 1)
        * scipy.special.hyp2f1(
            (n - 1) / 2.0,
            (n + 4) / 2.0,
            (n + 1) / 2.0,
            -Lmax ** 2 / (4 * np.pi ** 2 * Lbo ** 2),
        )
        - (Lmin / Lbo) ** (n - 1)
        * scipy.special.hyp2f1(
            (n - 1) / 2.0,
            (n + 4) / 2.0,
            (n + 1) / 2.0,
            -Lmin ** 2 / (4 * np.pi ** 2 * Lbo ** 2),
        )
    )
)

corr_lenght_finite_with_grid_size = lambda n, Lbo, Lmax, N: corr_lenght_finite(n, Lbo, Lmax, 2*Lmax/N)

corr_length_exact = (
    lambda n, Lbo: 8
    * np.sqrt(np.pi)
    * Lbo
    / (3 * (n ** 2 + 2 * n))
    * scipy.special.gamma(n / 2.0 + 2)
    / scipy.special.gamma(n / 2.0 - 1 / 2)
)
