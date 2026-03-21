//! Tests for complex tensor functionality
//!
//! This module contains tests for the complex number support in the tensor system.

use crate::errors::Result;
use crate::tensor::Tensor;

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_complex_tensor_creation() -> Result<()> {
        let real = vec![1.0, 2.0, 3.0, 4.0];
        let imag = vec![0.5, 1.5, 2.5, 3.5];
        let tensor = Tensor::complex(real, imag, &[2, 2])?;

        assert_eq!(tensor.shape(), vec![2, 2]);
        assert_eq!(tensor.dtype(), crate::tensor::DType::C32);
        Ok(())
    }

    #[test]
    fn test_complex_zeros() -> Result<()> {
        let tensor = Tensor::zeros_c32(&[3, 3])?;
        assert_eq!(tensor.shape(), vec![3, 3]);
        assert_eq!(tensor.dtype(), crate::tensor::DType::C32);
        Ok(())
    }

    #[test]
    fn test_real_and_imag_extraction() -> Result<()> {
        let real = vec![1.0, 2.0, 3.0, 4.0];
        let imag = vec![0.5, 1.5, 2.5, 3.5];
        let tensor = Tensor::complex(real.clone(), imag.clone(), &[2, 2])?;

        let real_part = tensor.real()?;
        let imag_part = tensor.imag()?;

        assert_eq!(real_part.data()?, real);
        assert_eq!(imag_part.data()?, imag);
        Ok(())
    }

    #[test]
    fn test_magnitude() -> Result<()> {
        let real = vec![3.0, 4.0];
        let imag = vec![4.0, 3.0];
        let tensor = Tensor::complex(real, imag, &[2])?;

        let magnitude = tensor.magnitude()?;
        let expected = [5.0, 5.0]; // sqrt(3^2 + 4^2) = sqrt(4^2 + 3^2) = 5

        for (actual, expected) in magnitude.data()?.iter().zip(expected.iter()) {
            assert_relative_eq!(actual, expected, epsilon = 1e-5);
        }
        Ok(())
    }

    #[test]
    fn test_phase() -> Result<()> {
        let real = vec![1.0, 0.0, -1.0, 0.0];
        let imag = vec![0.0, 1.0, 0.0, -1.0];
        let tensor = Tensor::complex(real, imag, &[4])?;

        let phase = tensor.phase()?;
        let expected = [
            0.0,
            std::f32::consts::FRAC_PI_2,
            std::f32::consts::PI,
            -std::f32::consts::FRAC_PI_2,
        ];

        for (actual, expected) in phase.data()?.iter().zip(expected.iter()) {
            assert_relative_eq!(actual, expected, epsilon = 1e-5);
        }
        Ok(())
    }

    #[test]
    fn test_complex_conjugate() -> Result<()> {
        let real = vec![1.0, 2.0];
        let imag = vec![3.0, 4.0];
        let tensor = Tensor::complex(real.clone(), imag.clone(), &[2])?;

        let conj = tensor.conj()?;
        let conj_real = conj.real()?;
        let conj_imag = conj.imag()?;

        assert_eq!(conj_real.data()?, real);
        assert_eq!(conj_imag.data()?, vec![-3.0, -4.0]);
        Ok(())
    }

    #[test]
    fn test_real_to_complex_conversion() -> Result<()> {
        let real_tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2])?;
        let complex_tensor = real_tensor.to_complex()?;

        assert_eq!(complex_tensor.dtype(), crate::tensor::DType::C32);
        assert_eq!(complex_tensor.real()?.data()?, vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(complex_tensor.imag()?.data()?, vec![0.0, 0.0, 0.0, 0.0]);
        Ok(())
    }

    #[test]
    fn test_complex_f64() -> Result<()> {
        let real: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0];
        let imag: Vec<f64> = vec![0.5, 1.5, 2.5, 3.5];
        let tensor = Tensor::complex_f64(real.clone(), imag.clone(), &[2, 2])?;

        assert_eq!(tensor.shape(), vec![2, 2]);
        assert_eq!(tensor.dtype(), crate::tensor::DType::C64);

        let real_part = tensor.real()?;
        let imag_part = tensor.imag()?;

        // Note: conversion to f32 might have small differences
        for (actual, expected) in real_part.data()?.iter().zip(real.iter()) {
            assert_relative_eq!(*actual, *expected as f32, epsilon = 1e-5);
        }
        for (actual, expected) in imag_part.data()?.iter().zip(imag.iter()) {
            assert_relative_eq!(*actual, *expected as f32, epsilon = 1e-5);
        }
        Ok(())
    }

    #[test]
    fn test_real_tensor_magnitude() -> Result<()> {
        let tensor = Tensor::from_vec(vec![-3.0, 4.0, -5.0], &[3])?;
        let magnitude = tensor.magnitude()?;

        assert_eq!(magnitude.data()?, vec![3.0, 4.0, 5.0]);
        Ok(())
    }

    #[test]
    fn test_real_tensor_phase() -> Result<()> {
        let tensor = Tensor::from_vec(vec![1.0, -2.0, 3.0], &[3])?;
        let phase = tensor.phase()?;

        let expected = [0.0, std::f32::consts::PI, 0.0];
        let phase_data = phase.data()?;
        for (actual, expected) in phase_data.iter().zip(expected.iter()) {
            assert_relative_eq!(actual, expected, epsilon = 1e-5);
        }
        Ok(())
    }

    #[test]
    fn test_real_tensor_conjugate() -> Result<()> {
        let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3])?;
        let conj = tensor.conj()?;

        // For real tensors, conjugate should be the same
        assert_eq!(conj.data()?, tensor.data()?);
        Ok(())
    }
}
