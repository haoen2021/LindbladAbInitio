import numpy as np
def spinorb_from_spatial(one_body_integrals, two_body_integrals, EQ_TOLERANCE = 1e-15, abab = True):
    n_qubits = 2 * one_body_integrals.shape[0]
    # Initialize Hamiltonian coefficients.
    one_body_coefficients = np.zeros((n_qubits, n_qubits))
    two_body_coefficients = np.zeros((n_qubits, n_qubits, n_qubits, n_qubits))
    # Loop through integrals.
    if abab:
        for p in range(n_qubits // 2):
            for q in range(n_qubits // 2):
                # Populate 1-body coefficients. Require p and q have same spin.
                one_body_coefficients[2 * p, 2 * q] = one_body_integrals[p, q]
                one_body_coefficients[2 * p + 1, 2 * q + 1] = one_body_integrals[p, q]
                # Continue looping to prepare 2-body coefficients.
                for r in range(n_qubits // 2):
                    for s in range(n_qubits // 2):
                        # Mixed spin
                        two_body_coefficients[2 * p, 2 * q + 1, 2 * r + 1, 2 * s] = two_body_integrals[
                            p, q, r, s
                        ]
                        two_body_coefficients[2 * p + 1, 2 * q, 2 * r, 2 * s + 1] = two_body_integrals[
                            p, q, r, s
                        ]

                        # Same spin
                        two_body_coefficients[2 * p, 2 * q, 2 * r, 2 * s] = two_body_integrals[
                            p, q, r, s
                        ]
                        two_body_coefficients[
                            2 * p + 1, 2 * q + 1, 2 * r + 1, 2 * s + 1
                        ] = two_body_integrals[p, q, r, s]
    else:
        for p in range(n_qubits // 2):
            for q in range(n_qubits // 2):
                # Populate 1-body coefficients. Require p and q have same spin.
                one_body_coefficients[p, q] = one_body_integrals[p, q]
                one_body_coefficients[p + (n_qubits//2), q + (n_qubits//2)] = one_body_integrals[p, q]
                # Continue looping to prepare 2-body coefficients.
                for r in range(n_qubits // 2):
                    for s in range(n_qubits // 2):
                        # Mixed spin
                        two_body_coefficients[p, q + (n_qubits//2), r + (n_qubits//2), s] = two_body_integrals[
                            p, q, r, s
                        ]
                        two_body_coefficients[p + (n_qubits//2), q, r, s + (n_qubits//2)] = two_body_integrals[
                            p, q, r, s
                        ]

                        # Same spin
                        two_body_coefficients[p, q, r, s] = two_body_integrals[
                            p, q, r, s
                        ]
                        two_body_coefficients[
                            p + (n_qubits//2), q + (n_qubits//2), r  + (n_qubits//2), s +  (n_qubits//2)
                        ] = two_body_integrals[p, q, r, s]

    # Truncate.
    one_body_coefficients[np.absolute(one_body_coefficients) < EQ_TOLERANCE] = 0.0
    two_body_coefficients[np.absolute(two_body_coefficients) < EQ_TOLERANCE] = 0.0

    return one_body_coefficients, two_body_coefficients

def spinorb_from_spatial_ovlp(one_body_integrals, o_integral, two_body_integrals, EQ_TOLERANCE = 1e-15, abab = True):
    n_qubits = 2 * one_body_integrals.shape[0]
    # Initialize Hamiltonian coefficients.
    one_body_coefficients = np.zeros((n_qubits, n_qubits))
    two_body_coefficients = np.zeros((n_qubits, n_qubits, n_qubits, n_qubits))
    S_matrix = np.zeros((n_qubits, n_qubits))
    # Loop through integrals.
    if abab:
        for p in range(n_qubits // 2):
            for q in range(n_qubits // 2):
                # Populate 1-body coefficients. Require p and q have same spin.
                one_body_coefficients[2 * p, 2 * q] = one_body_integrals[p, q]
                one_body_coefficients[2 * p + 1, 2 * q + 1] = one_body_integrals[p, q]
                S_matrix[2 * p, 2 * q] = o_integral[p, q]
                S_matrix[2 * p + 1, 2 * q + 1] = o_integral[p, q]
                # Continue looping to prepare 2-body coefficients.
                for r in range(n_qubits // 2):
                    for s in range(n_qubits // 2):
                        # Mixed spin
                        two_body_coefficients[2 * p + 1, 2 * q, 2 * r + 1, 2 * s] = two_body_integrals[
                            p, q, r, s
                        ]
                        two_body_coefficients[2 * p, 2 * q + 1, 2 * r, 2 * s + 1] = two_body_integrals[
                            p, q, r, s
                        ]

                        # Same spin
                        two_body_coefficients[2 * p, 2 * q, 2 * r, 2 * s] = two_body_integrals[
                            p, q, r, s
                        ]
                        two_body_coefficients[
                            2 * p + 1, 2 * q + 1, 2 * r + 1, 2 * s + 1
                        ] = two_body_integrals[p, q, r, s]
    else:
        for p in range(n_qubits // 2):
            for q in range(n_qubits // 2):
                # Populate 1-body coefficients. Require p and q have same spin.
                one_body_coefficients[p, q] = one_body_integrals[p, q]
                one_body_coefficients[p + (n_qubits//2), q + (n_qubits//2)] = one_body_integrals[p, q]
                S_matrix[p, q] = o_integral[p, q]
                S_matrix[p + (n_qubits//2), q + (n_qubits//2)] = o_integral[p, q]
                # Continue looping to prepare 2-body coefficients.
                for r in range(n_qubits // 2):
                    for s in range(n_qubits // 2):
                        # Mixed spin
                        two_body_coefficients[p + (n_qubits//2), q , r + (n_qubits//2), s] = two_body_integrals[
                            p, q, r, s
                        ]
                        two_body_coefficients[p, q+ (n_qubits//2), r, s + (n_qubits//2)] = two_body_integrals[
                            p, q, r, s
                        ]

                        # Same spin
                        two_body_coefficients[p, q, r, s] = two_body_integrals[
                            p, q, r, s
                        ]
                        two_body_coefficients[
                            p + (n_qubits//2), q + (n_qubits//2), r  + (n_qubits//2), s +  (n_qubits//2)
                        ] = two_body_integrals[p, q, r, s]

    # Truncate.
    one_body_coefficients[np.absolute(one_body_coefficients) < EQ_TOLERANCE] = 0.0
    two_body_coefficients[np.absolute(two_body_coefficients) < EQ_TOLERANCE] = 0.0
    S_matrix[np.absolute(S_matrix) < EQ_TOLERANCE] = 0.0
    return one_body_coefficients, S_matrix, two_body_coefficients
