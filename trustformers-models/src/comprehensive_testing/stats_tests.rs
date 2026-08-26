//! Tests for the statistical primitives, checked against published values.

#[cfg(test)]
mod tests {
    use crate::comprehensive_testing::stats::*;

    #[test]
    fn test_ln_gamma_known_values() {
        // Γ(1) = 1, Γ(5) = 24, Γ(0.5) = sqrt(pi)
        assert!(ln_gamma(1.0).abs() < 1e-12);
        assert!((ln_gamma(5.0) - 24.0f64.ln()).abs() < 1e-10);
        assert!((ln_gamma(0.5) - std::f64::consts::PI.sqrt().ln()).abs() < 1e-10);
    }

    #[test]
    fn test_chi_square_critical_values() {
        // Textbook 5% critical values.
        let p = chi_square_p_value(3.841, 1).expect("p-value");
        assert!((p - 0.05).abs() < 1e-3, "chi2=3.841, df=1 -> p={p}");

        let p = chi_square_p_value(5.991, 2).expect("p-value");
        assert!((p - 0.05).abs() < 1e-3, "chi2=5.991, df=2 -> p={p}");

        let p = chi_square_p_value(9.488, 4).expect("p-value");
        assert!((p - 0.05).abs() < 1e-3, "chi2=9.488, df=4 -> p={p}");

        // 1% critical value with 4 degrees of freedom.
        let p = chi_square_p_value(13.277, 4).expect("p-value");
        assert!((p - 0.01).abs() < 1e-3, "chi2=13.277, df=4 -> p={p}");
    }

    #[test]
    fn test_chi_square_p_value_monotone_and_bounded() {
        let mut previous = 1.0;
        for statistic in [0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0] {
            let p = chi_square_p_value(statistic, 3).expect("p-value");
            assert!((0.0..=1.0).contains(&p), "p={p} out of range");
            assert!(
                p <= previous + 1e-12,
                "p-value must decrease: {p} > {previous}"
            );
            previous = p;
        }
        assert_eq!(chi_square_p_value(0.0, 1).expect("p-value"), 1.0);
        assert!(chi_square_p_value(1.0, 0).is_err());
        assert!(chi_square_p_value(-1.0, 1).is_err());
    }

    #[test]
    fn test_chi_square_test_known_contingency_table() {
        // 2x2 table with equal margins: every expected count is 25.
        // X^2 = 4 * (5^2 / 25) = 4.0, df = 1, p ~= 0.0455.
        let table = vec![vec![20.0, 30.0], vec![30.0, 20.0]];
        let result = chi_square_test(&table).expect("chi-square");

        assert_eq!(result.degrees_of_freedom, 1);
        assert!((result.statistic - 4.0).abs() < 1e-9, "{:?}", result);
        assert!((result.p_value - 0.0455).abs() < 1e-3, "{:?}", result);
        assert!((result.min_expected_count - 25.0).abs() < 1e-9);
    }

    #[test]
    fn test_chi_square_test_independent_table_is_not_significant() {
        // Perfectly proportional table: statistic is exactly zero.
        let table = vec![vec![10.0, 20.0], vec![20.0, 40.0]];
        let result = chi_square_test(&table).expect("chi-square");
        assert!(result.statistic.abs() < 1e-9, "{:?}", result);
        assert!((result.p_value - 1.0).abs() < 1e-9, "{:?}", result);
    }

    #[test]
    fn test_chi_square_test_three_by_two_table() {
        // Independence-test example with three groups.
        // Expected counts: rows 40/40/40, columns 60/60 -> all cells 20.
        let table = vec![vec![30.0, 10.0], vec![20.0, 20.0], vec![10.0, 30.0]];
        let result = chi_square_test(&table).expect("chi-square");
        assert_eq!(result.degrees_of_freedom, 2);
        // (10^2 + 10^2 + 0 + 0 + 10^2 + 10^2)/20 = 20.0
        assert!((result.statistic - 20.0).abs() < 1e-9, "{:?}", result);
        assert!(result.p_value < 1e-4, "{:?}", result);
    }

    #[test]
    fn test_chi_square_test_rejects_degenerate_tables() {
        assert!(chi_square_test(&[vec![1.0, 2.0]]).is_err(), "single row");
        assert!(
            chi_square_test(&[vec![1.0], vec![2.0]]).is_err(),
            "single column"
        );
        assert!(
            chi_square_test(&[vec![1.0, 2.0], vec![3.0]]).is_err(),
            "ragged table"
        );
        assert!(
            chi_square_test(&[vec![0.0, 0.0], vec![0.0, 0.0]]).is_err(),
            "empty table"
        );
        assert!(
            chi_square_test(&[vec![1.0, 1.0], vec![0.0, 0.0]]).is_err(),
            "empty row"
        );
    }

    #[test]
    fn test_normal_cdf_known_values() {
        assert!((normal_cdf(0.0) - 0.5).abs() < 1e-9);
        assert!((normal_cdf(1.0) - 0.841_344_746).abs() < 1e-6);
        assert!((normal_cdf(-1.96) - 0.025).abs() < 1e-4);
        assert!((normal_cdf(1.96) - 0.975).abs() < 1e-4);
        assert!((normal_cdf(2.575_829) - 0.995).abs() < 1e-5);
    }

    #[test]
    fn test_normal_quantile_inverts_the_cdf() {
        for p in [0.001, 0.025, 0.1, 0.5, 0.9, 0.975, 0.999] {
            let z = normal_quantile(p);
            assert!(
                (normal_cdf(z) - p).abs() < 1e-9,
                "quantile({p}) = {z} does not invert the CDF"
            );
        }
        assert!((normal_quantile(0.975) - 1.959_963_985).abs() < 1e-6);
    }

    #[test]
    fn test_two_proportion_z_test_known_example() {
        // 30/100 vs 20/100: pooled p = 0.25, SE = sqrt(0.25*0.75*0.02) = 0.061237,
        // z = 0.10 / 0.061237 = 1.63299, two-sided p = 0.10247.
        let test = two_proportion_z_test(30.0, 100.0, 20.0, 100.0, 0.95).expect("z-test");
        assert!((test.difference - 0.1).abs() < 1e-12);
        assert!((test.z - 1.632_993).abs() < 1e-5, "{:?}", test);
        assert!((test.p_value - 0.102_470).abs() < 1e-4, "{:?}", test);

        // The interval brackets the observed difference and covers zero here.
        assert!(test.confidence_interval.0 < test.difference);
        assert!(test.confidence_interval.1 > test.difference);
        assert!(test.confidence_interval.0 < 0.0 && test.confidence_interval.1 > 0.0);
    }

    #[test]
    fn test_two_proportion_z_test_detects_a_large_gap() {
        // 90/100 vs 10/100 is overwhelmingly significant.
        let test = two_proportion_z_test(90.0, 100.0, 10.0, 100.0, 0.95).expect("z-test");
        assert!((test.difference - 0.8).abs() < 1e-12);
        assert!(test.p_value < 1e-10, "{:?}", test);
        assert!(test.confidence_interval.0 > 0.0, "{:?}", test);
    }

    #[test]
    fn test_two_proportion_z_test_identical_groups() {
        let test = two_proportion_z_test(25.0, 50.0, 25.0, 50.0, 0.95).expect("z-test");
        assert!(test.difference.abs() < 1e-12);
        assert!((test.p_value - 1.0).abs() < 1e-9, "{:?}", test);
    }

    #[test]
    fn test_two_proportion_z_test_rejects_invalid_input() {
        assert!(two_proportion_z_test(1.0, 0.0, 1.0, 10.0, 0.95).is_err());
        assert!(two_proportion_z_test(11.0, 10.0, 1.0, 10.0, 0.95).is_err());
        assert!(two_proportion_z_test(1.0, 10.0, 1.0, 10.0, 1.5).is_err());
    }

    #[test]
    fn test_expected_calibration_error() {
        // Perfectly calibrated: confidence 0 for negatives, 1 for positives.
        let predictions = vec![0.0, 0.0, 1.0, 1.0];
        let labels = vec![0, 0, 1, 1];
        let ece = expected_calibration_error(&predictions, &labels, 10).expect("ece");
        assert!(
            ece.abs() < 1e-12,
            "perfect calibration must give 0, got {ece}"
        );

        // Maximally miscalibrated: confident and always wrong.
        let predictions = vec![1.0, 1.0, 0.0, 0.0];
        let labels = vec![0, 0, 1, 1];
        let ece = expected_calibration_error(&predictions, &labels, 10).expect("ece");
        assert!((ece - 1.0).abs() < 1e-12, "got {ece}");

        // Half the confident predictions are wrong -> ECE 0.5.
        let predictions = vec![1.0, 1.0, 1.0, 1.0];
        let labels = vec![1, 1, 0, 0];
        let ece = expected_calibration_error(&predictions, &labels, 10).expect("ece");
        assert!((ece - 0.5).abs() < 1e-12, "got {ece}");

        assert!(expected_calibration_error(&[0.5], &[], 10).is_err());
        assert!(expected_calibration_error(&[], &[], 10).is_err());
        assert!(expected_calibration_error(&[0.5], &[1], 0).is_err());
    }

    #[test]
    fn test_regularized_gamma_identities() {
        for (a, x) in [(1.0, 0.5), (2.5, 3.0), (5.0, 1.0), (0.5, 4.0)] {
            let p = regularized_gamma_p(a, x).expect("p");
            let q = regularized_gamma_q(a, x).expect("q");
            assert!(
                (p + q - 1.0).abs() < 1e-12,
                "P+Q must be 1 for a={a}, x={x}"
            );
            assert!((0.0..=1.0).contains(&p));
        }
        // P(1, x) = 1 - exp(-x) for the exponential distribution.
        let p = regularized_gamma_p(1.0, 2.0).expect("p");
        assert!((p - (1.0 - (-2.0f64).exp())).abs() < 1e-12);

        assert!(regularized_gamma_p(0.0, 1.0).is_err());
        assert!(regularized_gamma_p(1.0, -1.0).is_err());
    }
}
