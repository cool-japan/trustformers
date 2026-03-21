//! 2D Convolutional layer implementation.
//!
//! This module provides a Conv2d layer that performs 2D convolution using the
//! im2col (image to column) approach: input patches are unrolled into a matrix
//! and the convolution is computed as a single matrix multiplication.
//!
//! Supports configurable kernel size, stride, padding, dilation, groups, and
//! optional bias.

use crate::errors::{Result, TrustformersError};
use crate::tensor::Tensor;
use crate::traits::Layer;
use serde::{Deserialize, Serialize};

/// 2D Convolutional layer
///
/// Applies a 2D convolution over an input tensor with shape `[N, C_in, H, W]`.
/// The weight tensor has shape `[C_out, C_in/groups, kH, kW]`.
///
/// The forward pass uses im2col + matmul for efficient computation:
/// 1. Extract overlapping patches from the input and arrange them as columns.
/// 2. Reshape the weight into a 2D matrix.
/// 3. Compute the output as a matrix product of weights and columns.
/// 4. Reshape the result back to `[N, C_out, H_out, W_out]`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Conv2d {
    pub in_channels: usize,
    pub out_channels: usize,
    pub kernel_size: (usize, usize),
    pub stride: (usize, usize),
    pub padding: (usize, usize),
    pub dilation: (usize, usize),
    pub groups: usize,
    pub bias: bool,
    #[serde(skip)]
    pub weight: Option<Tensor>,
    #[serde(skip)]
    pub bias_term: Option<Tensor>,
}

impl Conv2d {
    /// Create a new Conv2d layer
    ///
    /// # Arguments
    ///
    /// * `in_channels` - Number of input channels
    /// * `out_channels` - Number of output channels
    /// * `kernel_size` - Size of the convolutional kernel
    /// * `stride` - Stride of the convolution
    /// * `padding` - Padding applied to the input
    /// * `bias` - Whether to include a bias term
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
        bias: bool,
    ) -> Result<Self> {
        Ok(Self {
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation: (1, 1),
            groups: 1,
            bias,
            weight: None,
            bias_term: None,
        })
    }

    /// Create a Conv2d layer with full configuration including dilation and groups
    ///
    /// # Arguments
    ///
    /// * `in_channels` - Number of input channels
    /// * `out_channels` - Number of output channels
    /// * `kernel_size` - Size of the convolutional kernel
    /// * `stride` - Stride of the convolution
    /// * `padding` - Padding applied to the input
    /// * `dilation` - Spacing between kernel elements
    /// * `groups` - Number of blocked connections from input to output channels
    /// * `bias` - Whether to include a bias term
    pub fn new_full(
        in_channels: usize,
        out_channels: usize,
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
        dilation: (usize, usize),
        groups: usize,
        bias: bool,
    ) -> Result<Self> {
        if groups == 0 {
            return Err(TrustformersError::tensor_op_error(
                "groups must be > 0",
                "Conv2d::new_full",
            ));
        }
        if in_channels % groups != 0 {
            return Err(TrustformersError::tensor_op_error(
                &format!(
                    "in_channels ({}) must be divisible by groups ({})",
                    in_channels, groups
                ),
                "Conv2d::new_full",
            ));
        }
        if out_channels % groups != 0 {
            return Err(TrustformersError::tensor_op_error(
                &format!(
                    "out_channels ({}) must be divisible by groups ({})",
                    out_channels, groups
                ),
                "Conv2d::new_full",
            ));
        }
        Ok(Self {
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            weight: None,
            bias_term: None,
        })
    }

    /// Create a Conv2d layer with simple kernel size (same for width and height)
    pub fn new_simple(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        bias: bool,
    ) -> Result<Self> {
        Self::new(
            in_channels,
            out_channels,
            (kernel_size, kernel_size),
            (1, 1),
            (0, 0),
            bias,
        )
    }

    /// Initialize weights for the layer
    pub fn init_weights(&mut self, weight: Tensor, bias: Option<Tensor>) -> Result<()> {
        self.weight = Some(weight);
        self.bias_term = bias;
        Ok(())
    }

    /// Compute output spatial dimensions
    fn compute_output_size(&self, h_in: usize, w_in: usize) -> Result<(usize, usize)> {
        let (k_h, k_w) = self.kernel_size;
        let (s_h, s_w) = self.stride;
        let (p_h, p_w) = self.padding;
        let (d_h, d_w) = self.dilation;

        // Effective kernel size with dilation
        let eff_k_h = d_h * (k_h - 1) + 1;
        let eff_k_w = d_w * (k_w - 1) + 1;

        let padded_h = h_in + 2 * p_h;
        let padded_w = w_in + 2 * p_w;

        if padded_h < eff_k_h || padded_w < eff_k_w {
            return Err(TrustformersError::tensor_op_error(
                &format!(
                    "Padded input size ({}, {}) is smaller than effective kernel size ({}, {})",
                    padded_h, padded_w, eff_k_h, eff_k_w
                ),
                "Conv2d::compute_output_size",
            ));
        }

        let h_out = (padded_h - eff_k_h) / s_h + 1;
        let w_out = (padded_w - eff_k_w) / s_w + 1;

        Ok((h_out, w_out))
    }

    /// Perform im2col: extract input patches into a column matrix.
    ///
    /// For a single sample with `c_in_per_group` input channels, produces a matrix
    /// of shape `[c_in_per_group * k_h * k_w, h_out * w_out]`.
    ///
    /// Each column corresponds to one output spatial location and contains all
    /// the input values that participate in computing that output element.
    fn im2col(
        &self,
        input_data: &[f32],
        c_in_per_group: usize,
        h_in: usize,
        w_in: usize,
        h_out: usize,
        w_out: usize,
        c_offset: usize,
    ) -> Vec<f32> {
        let (k_h, k_w) = self.kernel_size;
        let (s_h, s_w) = self.stride;
        let (p_h, p_w) = self.padding;
        let (d_h, d_w) = self.dilation;

        let col_rows = c_in_per_group * k_h * k_w;
        let col_cols = h_out * w_out;
        let mut col = vec![0.0f32; col_rows * col_cols];

        for c in 0..c_in_per_group {
            let c_abs = c + c_offset;
            for kh in 0..k_h {
                for kw in 0..k_w {
                    let row_idx = c * k_h * k_w + kh * k_w + kw;
                    for oh in 0..h_out {
                        for ow in 0..w_out {
                            let ih = oh * s_h + kh * d_h;
                            let iw = ow * s_w + kw * d_w;
                            // ih and iw are relative to the padded input
                            let ih_orig = ih as isize - p_h as isize;
                            let iw_orig = iw as isize - p_w as isize;

                            let val = if ih_orig >= 0
                                && ih_orig < h_in as isize
                                && iw_orig >= 0
                                && iw_orig < w_in as isize
                            {
                                let ih_u = ih_orig as usize;
                                let iw_u = iw_orig as usize;
                                // input layout: [C_in, H, W] (for a single sample)
                                input_data[c_abs * h_in * w_in + ih_u * w_in + iw_u]
                            } else {
                                0.0 // zero-padding
                            };

                            col[row_idx * col_cols + oh * w_out + ow] = val;
                        }
                    }
                }
            }
        }

        col
    }
}

impl Layer for Conv2d {
    type Input = Tensor;
    type Output = Tensor;

    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        let weight = self.weight.as_ref().ok_or_else(|| {
            TrustformersError::tensor_op_error(
                "Conv2d weights not initialized. Call init_weights() first.",
                "Conv2d::forward",
            )
        })?;

        // Validate input shape: [N, C_in, H, W]
        let input_shape = input.shape();
        if input_shape.len() != 4 {
            return Err(TrustformersError::tensor_op_error(
                &format!(
                    "Conv2d expects 4D input [N, C_in, H, W], got {}D with shape {:?}",
                    input_shape.len(),
                    input_shape
                ),
                "Conv2d::forward",
            ));
        }

        let batch_size = input_shape[0];
        let c_in = input_shape[1];
        let h_in = input_shape[2];
        let w_in = input_shape[3];

        if c_in != self.in_channels {
            return Err(TrustformersError::tensor_op_error(
                &format!(
                    "Input channels ({}) do not match expected ({})",
                    c_in, self.in_channels
                ),
                "Conv2d::forward",
            ));
        }

        // Validate weight shape: [C_out, C_in/groups, kH, kW]
        let weight_shape = weight.shape();
        if weight_shape.len() != 4 {
            return Err(TrustformersError::tensor_op_error(
                &format!(
                    "Conv2d weight must be 4D [C_out, C_in/groups, kH, kW], got {}D with shape {:?}",
                    weight_shape.len(),
                    weight_shape
                ),
                "Conv2d::forward",
            ));
        }

        let c_in_per_group = self.in_channels / self.groups;
        let c_out_per_group = self.out_channels / self.groups;

        if weight_shape[0] != self.out_channels
            || weight_shape[1] != c_in_per_group
            || weight_shape[2] != self.kernel_size.0
            || weight_shape[3] != self.kernel_size.1
        {
            return Err(TrustformersError::tensor_op_error(
                &format!(
                    "Weight shape {:?} does not match expected [{}, {}, {}, {}]",
                    weight_shape,
                    self.out_channels,
                    c_in_per_group,
                    self.kernel_size.0,
                    self.kernel_size.1
                ),
                "Conv2d::forward",
            ));
        }

        // Compute output spatial dimensions
        let (h_out, w_out) = self.compute_output_size(h_in, w_in)?;

        // Get contiguous data
        let input_contig = input.contiguous()?;
        let weight_contig = weight.contiguous()?;
        let input_data = input_contig.data()?;
        let weight_data = weight_contig.data()?;

        // Bias data (if present)
        let bias_data = if self.bias {
            let bias_tensor = self.bias_term.as_ref().ok_or_else(|| {
                TrustformersError::tensor_op_error(
                    "Conv2d bias is enabled but bias_term is not set",
                    "Conv2d::forward",
                )
            })?;
            let bias_shape = bias_tensor.shape();
            if bias_shape.len() != 1 || bias_shape[0] != self.out_channels {
                return Err(TrustformersError::tensor_op_error(
                    &format!(
                        "Bias shape {:?} does not match out_channels {}",
                        bias_shape, self.out_channels
                    ),
                    "Conv2d::forward",
                ));
            }
            Some(bias_tensor.data()?)
        } else {
            None
        };

        // Allocate output: [N, C_out, H_out, W_out]
        let output_spatial = h_out * w_out;
        let col_rows = c_in_per_group * self.kernel_size.0 * self.kernel_size.1;
        let mut output_data = vec![0.0f32; batch_size * self.out_channels * output_spatial];

        let sample_input_size = c_in * h_in * w_in;
        let sample_output_size = self.out_channels * output_spatial;

        for n in 0..batch_size {
            let input_sample = &input_data[n * sample_input_size..(n + 1) * sample_input_size];

            for g in 0..self.groups {
                let c_in_start = g * c_in_per_group;
                let c_out_start = g * c_out_per_group;

                // im2col: extract patches for this group
                // Result shape: [col_rows, output_spatial]
                let col = self.im2col(
                    input_sample,
                    c_in_per_group,
                    h_in,
                    w_in,
                    h_out,
                    w_out,
                    c_in_start,
                );

                // Weight for this group: [c_out_per_group, c_in_per_group * kH * kW]
                // weight layout: [C_out, C_in/groups, kH, kW]
                // For group g, we want filters c_out_start..c_out_start+c_out_per_group
                // Matmul: weight_group [c_out_per_group, col_rows] x col [col_rows, output_spatial]
                //       = result [c_out_per_group, output_spatial]
                for oc in 0..c_out_per_group {
                    let w_row_start = (c_out_start + oc) * col_rows;
                    let out_offset = n * sample_output_size + (c_out_start + oc) * output_spatial;

                    for os in 0..output_spatial {
                        let mut sum = 0.0f32;
                        for k in 0..col_rows {
                            sum += weight_data[w_row_start + k] * col[k * output_spatial + os];
                        }

                        // Add bias if present
                        if let Some(ref bd) = bias_data {
                            sum += bd[c_out_start + oc];
                        }

                        output_data[out_offset + os] = sum;
                    }
                }
            }
        }

        Tensor::from_vec(output_data, &[batch_size, self.out_channels, h_out, w_out])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper to create a Conv2d with initialized weights for testing
    fn make_conv2d(
        in_channels: usize,
        out_channels: usize,
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
        dilation: (usize, usize),
        groups: usize,
        use_bias: bool,
        weight_data: Vec<f32>,
        bias_data: Option<Vec<f32>>,
    ) -> Result<Conv2d> {
        let mut conv = Conv2d::new_full(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            use_bias,
        )?;
        let c_in_per_group = in_channels / groups;
        let weight = Tensor::from_vec(
            weight_data,
            &[out_channels, c_in_per_group, kernel_size.0, kernel_size.1],
        )?;
        let bias = if use_bias {
            Some(Tensor::from_vec(
                bias_data.ok_or_else(|| {
                    TrustformersError::tensor_op_error("bias_data required", "test")
                })?,
                &[out_channels],
            )?)
        } else {
            None
        };
        conv.init_weights(weight, bias)?;
        Ok(conv)
    }

    #[test]
    fn test_conv2d_basic_3x3() -> Result<()> {
        // 1 input channel, 1 output channel, 3x3 kernel, no padding, stride=1
        // Input: 1x1x4x4 (batch=1, c=1, h=4, w=4)
        // Weight: 1x1x3x3 (all ones)
        // Output should be 1x1x2x2

        let input_data: Vec<f32> = (1..=16).map(|x| x as f32).collect();
        let weight_data = vec![1.0f32; 9]; // 3x3 all ones

        let conv = make_conv2d(
            1,
            1,
            (3, 3),
            (1, 1),
            (0, 0),
            (1, 1),
            1,
            false,
            weight_data,
            None,
        )?;

        let input = Tensor::from_vec(input_data, &[1, 1, 4, 4])?;
        let output = conv.forward(input)?;

        assert_eq!(output.shape(), vec![1, 1, 2, 2]);

        let out_data = output.data()?;
        // Manual computation for a 4x4 input with 3x3 all-ones kernel:
        // pos (0,0): 1+2+3+5+6+7+9+10+11 = 54
        // pos (0,1): 2+3+4+6+7+8+10+11+12 = 63
        // pos (1,0): 5+6+7+9+10+11+13+14+15 = 90
        // pos (1,1): 6+7+8+10+11+12+14+15+16 = 99
        assert!((out_data[0] - 54.0).abs() < 1e-5);
        assert!((out_data[1] - 63.0).abs() < 1e-5);
        assert!((out_data[2] - 90.0).abs() < 1e-5);
        assert!((out_data[3] - 99.0).abs() < 1e-5);

        Ok(())
    }

    #[test]
    fn test_conv2d_with_bias() -> Result<()> {
        // Same as basic but with bias = 10.0
        let input_data: Vec<f32> = (1..=16).map(|x| x as f32).collect();
        let weight_data = vec![1.0f32; 9];
        let bias_data = vec![10.0f32];

        let conv = make_conv2d(
            1,
            1,
            (3, 3),
            (1, 1),
            (0, 0),
            (1, 1),
            1,
            true,
            weight_data,
            Some(bias_data),
        )?;

        let input = Tensor::from_vec(input_data, &[1, 1, 4, 4])?;
        let output = conv.forward(input)?;

        assert_eq!(output.shape(), vec![1, 1, 2, 2]);

        let out_data = output.data()?;
        assert!((out_data[0] - 64.0).abs() < 1e-5); // 54 + 10
        assert!((out_data[1] - 73.0).abs() < 1e-5); // 63 + 10
        assert!((out_data[2] - 100.0).abs() < 1e-5); // 90 + 10
        assert!((out_data[3] - 109.0).abs() < 1e-5); // 99 + 10

        Ok(())
    }

    #[test]
    fn test_conv2d_stride_2() -> Result<()> {
        // 1x1x5x5 input, 1x1x3x3 kernel, stride=2, no padding
        // Output should be 1x1x2x2 (floor((5-3)/2)+1 = 2)
        let input_data: Vec<f32> = (1..=25).map(|x| x as f32).collect();
        let weight_data = vec![1.0f32; 9]; // 3x3 all ones

        let conv = make_conv2d(
            1,
            1,
            (3, 3),
            (2, 2),
            (0, 0),
            (1, 1),
            1,
            false,
            weight_data,
            None,
        )?;

        let input = Tensor::from_vec(input_data, &[1, 1, 5, 5])?;
        let output = conv.forward(input)?;

        assert_eq!(output.shape(), vec![1, 1, 2, 2]);

        let out_data = output.data()?;
        // pos (0,0): sum of rows 0-2, cols 0-2
        // 1+2+3+6+7+8+11+12+13 = 63
        assert!((out_data[0] - 63.0).abs() < 1e-5);
        // pos (0,1): rows 0-2, cols 2-4
        // 3+4+5+8+9+10+13+14+15 = 81
        assert!((out_data[1] - 81.0).abs() < 1e-5);
        // pos (1,0): rows 2-4, cols 0-2
        // 11+12+13+16+17+18+21+22+23 = 153
        assert!((out_data[2] - 153.0).abs() < 1e-5);
        // pos (1,1): rows 2-4, cols 2-4
        // 13+14+15+18+19+20+23+24+25 = 171
        assert!((out_data[3] - 171.0).abs() < 1e-5);

        Ok(())
    }

    #[test]
    fn test_conv2d_padding() -> Result<()> {
        // 1x1x3x3 input, 1x1x3x3 kernel, stride=1, padding=1
        // Output should be 1x1x3x3 (same size due to padding)
        let input_data: Vec<f32> = (1..=9).map(|x| x as f32).collect();
        let weight_data = vec![1.0f32; 9]; // identity-like with all ones

        let conv = make_conv2d(
            1,
            1,
            (3, 3),
            (1, 1),
            (1, 1),
            (1, 1),
            1,
            false,
            weight_data,
            None,
        )?;

        let input = Tensor::from_vec(input_data, &[1, 1, 3, 3])?;
        let output = conv.forward(input)?;

        assert_eq!(output.shape(), vec![1, 1, 3, 3]);

        let out_data = output.data()?;
        // Center element (1,1): sum of entire 3x3 = 45
        assert!((out_data[4] - 45.0).abs() < 1e-5);
        // Corner (0,0): only input[0,0]=1 contributes (since kernel sees only
        // one non-padded element at position (2,2) of the 3x3 receptive field)
        // Actually with padding=1, the 3x3 kernel at output (0,0) sees:
        // rows -1..1, cols -1..1 in the original input
        // Only (0,0), (0,1), (1,0), (1,1) are valid => 1+2+4+5 = 12
        assert!((out_data[0] - 12.0).abs() < 1e-5);

        Ok(())
    }

    #[test]
    fn test_conv2d_dilation() -> Result<()> {
        // 1x1x5x5 input, 1x1x3x3 kernel, stride=1, no padding, dilation=2
        // Effective kernel size: 2*(3-1)+1 = 5
        // Output should be 1x1x1x1 (floor((5-5)/1)+1 = 1)
        let input_data: Vec<f32> = (1..=25).map(|x| x as f32).collect();
        let weight_data = vec![1.0f32; 9];

        let conv = make_conv2d(
            1,
            1,
            (3, 3),
            (1, 1),
            (0, 0),
            (2, 2),
            1,
            false,
            weight_data,
            None,
        )?;

        let input = Tensor::from_vec(input_data, &[1, 1, 5, 5])?;
        let output = conv.forward(input)?;

        assert_eq!(output.shape(), vec![1, 1, 1, 1]);

        let out_data = output.data()?;
        // With dilation=2, the kernel samples at positions (0,0),(0,2),(0,4),(2,0),(2,2),(2,4),(4,0),(4,2),(4,4)
        // Values: 1,3,5,11,13,15,21,23,25 = 117
        assert!((out_data[0] - 117.0).abs() < 1e-5);

        Ok(())
    }

    #[test]
    fn test_conv2d_groups() -> Result<()> {
        // 2 input channels, 2 output channels, groups=2 (depthwise-like)
        // Each group processes 1 input channel and produces 1 output channel
        // Weight shape: [2, 1, 3, 3]
        let input_data: Vec<f32> = {
            let mut v = Vec::with_capacity(2 * 4 * 4);
            // channel 0: all 1.0
            v.extend(vec![1.0f32; 16]);
            // channel 1: all 2.0
            v.extend(vec![2.0f32; 16]);
            v
        };

        // Group 0 weight (filter 0): all 1.0
        // Group 1 weight (filter 1): all 1.0
        let weight_data = vec![1.0f32; 2 * 1 * 3 * 3];

        let conv = make_conv2d(
            2,
            2,
            (3, 3),
            (1, 1),
            (0, 0),
            (1, 1),
            2,
            false,
            weight_data,
            None,
        )?;

        let input = Tensor::from_vec(input_data, &[1, 2, 4, 4])?;
        let output = conv.forward(input)?;

        assert_eq!(output.shape(), vec![1, 2, 2, 2]);

        let out_data = output.data()?;
        // Group 0: conv of all-1.0 channel with all-1.0 kernel => 9.0 everywhere
        assert!((out_data[0] - 9.0).abs() < 1e-5);
        assert!((out_data[1] - 9.0).abs() < 1e-5);
        assert!((out_data[2] - 9.0).abs() < 1e-5);
        assert!((out_data[3] - 9.0).abs() < 1e-5);
        // Group 1: conv of all-2.0 channel with all-1.0 kernel => 18.0 everywhere
        assert!((out_data[4] - 18.0).abs() < 1e-5);
        assert!((out_data[5] - 18.0).abs() < 1e-5);
        assert!((out_data[6] - 18.0).abs() < 1e-5);
        assert!((out_data[7] - 18.0).abs() < 1e-5);

        Ok(())
    }

    #[test]
    fn test_conv2d_multi_channel() -> Result<()> {
        // 2 input channels, 3 output channels, 1x1 kernel, no padding, stride=1
        // This is essentially a pointwise convolution (like a linear layer per pixel)
        // Weight shape: [3, 2, 1, 1]
        let input_data = vec![
            // batch=1, c=0, h=2, w=2
            1.0, 2.0, 3.0, 4.0, // batch=1, c=1, h=2, w=2
            5.0, 6.0, 7.0, 8.0,
        ];
        // Weight: [3, 2, 1, 1]
        // out_channel 0: [1.0, 0.0] => copies input channel 0
        // out_channel 1: [0.0, 1.0] => copies input channel 1
        // out_channel 2: [1.0, 1.0] => sums both channels
        let weight_data = vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0];

        let conv = make_conv2d(
            2,
            3,
            (1, 1),
            (1, 1),
            (0, 0),
            (1, 1),
            1,
            false,
            weight_data,
            None,
        )?;

        let input = Tensor::from_vec(input_data, &[1, 2, 2, 2])?;
        let output = conv.forward(input)?;

        assert_eq!(output.shape(), vec![1, 3, 2, 2]);

        let out_data = output.data()?;
        // out_channel 0 = input channel 0: [1,2,3,4]
        assert!((out_data[0] - 1.0).abs() < 1e-5);
        assert!((out_data[1] - 2.0).abs() < 1e-5);
        assert!((out_data[2] - 3.0).abs() < 1e-5);
        assert!((out_data[3] - 4.0).abs() < 1e-5);
        // out_channel 1 = input channel 1: [5,6,7,8]
        assert!((out_data[4] - 5.0).abs() < 1e-5);
        assert!((out_data[5] - 6.0).abs() < 1e-5);
        assert!((out_data[6] - 7.0).abs() < 1e-5);
        assert!((out_data[7] - 8.0).abs() < 1e-5);
        // out_channel 2 = sum: [6,8,10,12]
        assert!((out_data[8] - 6.0).abs() < 1e-5);
        assert!((out_data[9] - 8.0).abs() < 1e-5);
        assert!((out_data[10] - 10.0).abs() < 1e-5);
        assert!((out_data[11] - 12.0).abs() < 1e-5);

        Ok(())
    }

    #[test]
    fn test_conv2d_batch() -> Result<()> {
        // Batch of 2 with 1x1 input channels, 3x3 kernel, identity-like
        let input_data: Vec<f32> = {
            let mut v = Vec::new();
            // sample 0: 4x4 of ones
            v.extend(vec![1.0f32; 16]);
            // sample 1: 4x4 of twos
            v.extend(vec![2.0f32; 16]);
            v
        };
        let weight_data = vec![1.0f32; 9];

        let conv = make_conv2d(
            1,
            1,
            (3, 3),
            (1, 1),
            (0, 0),
            (1, 1),
            1,
            false,
            weight_data,
            None,
        )?;

        let input = Tensor::from_vec(input_data, &[2, 1, 4, 4])?;
        let output = conv.forward(input)?;

        assert_eq!(output.shape(), vec![2, 1, 2, 2]);

        let out_data = output.data()?;
        // Sample 0: 3x3 sum of 1.0 = 9.0 everywhere
        assert!((out_data[0] - 9.0).abs() < 1e-5);
        // Sample 1: 3x3 sum of 2.0 = 18.0 everywhere
        assert!((out_data[4] - 18.0).abs() < 1e-5);

        Ok(())
    }

    #[test]
    fn test_conv2d_no_weights_error() -> Result<()> {
        let conv = Conv2d::new_simple(1, 1, 3, false)?;
        let input = Tensor::from_vec(vec![0.0; 16], &[1, 1, 4, 4])?;
        let result = conv.forward(input);
        assert!(result.is_err());
        Ok(())
    }

    #[test]
    fn test_conv2d_wrong_input_dim_error() -> Result<()> {
        let mut conv = Conv2d::new_simple(1, 1, 3, false)?;
        let weight = Tensor::from_vec(vec![1.0; 9], &[1, 1, 3, 3])?;
        conv.init_weights(weight, None)?;

        let input = Tensor::from_vec(vec![0.0; 16], &[4, 4])?;
        let result = conv.forward(input);
        assert!(result.is_err());
        Ok(())
    }

    #[test]
    fn test_conv2d_invalid_groups() {
        // groups=0 should fail
        let result = Conv2d::new_full(4, 4, (3, 3), (1, 1), (0, 0), (1, 1), 0, false);
        assert!(result.is_err());

        // in_channels not divisible by groups
        let result = Conv2d::new_full(3, 4, (3, 3), (1, 1), (0, 0), (1, 1), 2, false);
        assert!(result.is_err());

        // out_channels not divisible by groups
        let result = Conv2d::new_full(4, 3, (3, 3), (1, 1), (0, 0), (1, 1), 2, false);
        assert!(result.is_err());
    }

    #[test]
    fn test_conv2d_new_simple() -> Result<()> {
        let conv = Conv2d::new_simple(3, 16, 3, true)?;
        assert_eq!(conv.in_channels, 3);
        assert_eq!(conv.out_channels, 16);
        assert_eq!(conv.kernel_size, (3, 3));
        assert_eq!(conv.stride, (1, 1));
        assert_eq!(conv.padding, (0, 0));
        assert_eq!(conv.dilation, (1, 1));
        assert_eq!(conv.groups, 1);
        assert!(conv.bias);
        Ok(())
    }
}
