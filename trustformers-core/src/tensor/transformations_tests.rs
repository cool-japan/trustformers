// Copyright (c) 2025-2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Tests for tensor transformation operations.
//!
//! These are regression tests for defects that used to live in
//! [`crate::tensor::transformations`]: `transpose` deep-copying the whole tensor
//! twice, `slice_ranges` cloning the entire source before slicing, and `gather`
//! panicking on oversized index tensors instead of returning an error.

#[cfg(test)]
mod tests {
    use crate::tensor::Tensor;

    /// Build a `[2, 3, 4]` tensor whose element at `[i, j, k]` is `i*100+j*10+k`,
    /// so any misplaced element is immediately identifiable.
    fn coded_3d() -> Tensor {
        let mut values = Vec::with_capacity(24);
        for i in 0..2 {
            for j in 0..3 {
                for k in 0..4 {
                    values.push((i * 100 + j * 10 + k) as f32);
                }
            }
        }
        Tensor::from_vec(values, &[2, 3, 4]).expect("build coded tensor")
    }

    // ------------------------------------------------------------------
    // transpose
    // ------------------------------------------------------------------

    /// `transpose` used to `clone()` the whole tensor and then copy it again
    /// through `as_standard_layout().to_owned()`. The values must be unchanged
    /// by the rewrite that removed the first copy.
    #[test]
    fn transpose_moves_the_right_elements() {
        let t = coded_3d();
        let swapped = t.transpose(0, 2).expect("transpose 0<->2");
        assert_eq!(swapped.shape(), vec![4, 3, 2]);

        let values = swapped.to_vec_f32().expect("values");
        // swapped[k, j, i] == original[i, j, k] == i*100 + j*10 + k
        for k in 0..4usize {
            for j in 0..3usize {
                for i in 0..2usize {
                    let flat = k * 6 + j * 2 + i;
                    let expected = (i * 100 + j * 10 + k) as f32;
                    assert_eq!(
                        values[flat], expected,
                        "swapped[{k},{j},{i}] should be {expected}"
                    );
                }
            }
        }
    }

    /// The result must be a standard-layout owned tensor, not a view carrying
    /// the source's permuted strides -- downstream BLAS paths call `as_slice()`.
    #[test]
    fn transpose_result_is_contiguous() {
        let t = coded_3d();
        let swapped = t.transpose(1, 2).expect("transpose 1<->2");
        match swapped {
            Tensor::F32(arr) => {
                assert!(
                    arr.is_standard_layout(),
                    "transpose must return row-major data"
                );
                assert!(
                    arr.as_slice().is_some(),
                    "transpose result must be sliceable"
                );
            },
            _ => panic!("F32 expected"),
        }
    }

    /// Transposing twice over the same pair of axes is the identity.
    #[test]
    fn transpose_is_an_involution() {
        let t = coded_3d();
        let round_trip = t
            .transpose(0, 2)
            .expect("first transpose")
            .transpose(0, 2)
            .expect("second transpose");
        assert_eq!(round_trip.shape(), t.shape());
        assert_eq!(
            round_trip.to_vec_f32().expect("round trip values"),
            t.to_vec_f32().expect("original values")
        );
    }

    #[test]
    fn transpose_rejects_out_of_range_axes() {
        let t = coded_3d();
        assert!(t.transpose(0, 3).is_err());
        assert!(t.transpose(5, 1).is_err());
    }

    #[test]
    fn transpose_i64_accepts_negative_axes() {
        let t = coded_3d();
        let via_negative = t.transpose_i64(-3, -1).expect("negative axes");
        let via_positive = t.transpose(0, 2).expect("positive axes");
        assert_eq!(via_negative.shape(), via_positive.shape());
        assert_eq!(
            via_negative.to_vec_f32().expect("values"),
            via_positive.to_vec_f32().expect("values")
        );
    }

    // ------------------------------------------------------------------
    // slice_ranges
    // ------------------------------------------------------------------

    /// `slice_ranges` used to `clone()` the entire source tensor before
    /// narrowing it, then produce another owned copy per axis. The rewrite
    /// narrows a view and materialises once -- the elements must be identical.
    #[test]
    fn slice_ranges_extracts_the_right_block() {
        let t = coded_3d();
        let block = t.slice_ranges(&[(1, 2), (0, 2), (1, 3)]).expect("slice");
        assert_eq!(block.shape(), vec![1, 2, 2]);

        // i = 1, j in {0,1}, k in {1,2}
        assert_eq!(
            block.to_vec_f32().expect("values"),
            vec![101.0, 102.0, 111.0, 112.0]
        );
    }

    /// A single-element slice from the far corner: the old "clone everything
    /// first" path made this cost the whole tensor.
    #[test]
    fn slice_ranges_extracts_a_single_corner_element() {
        let t = coded_3d();
        let corner = t.slice_ranges(&[(1, 2), (2, 3), (3, 4)]).expect("slice");
        assert_eq!(corner.shape(), vec![1, 1, 1]);
        assert_eq!(corner.to_vec_f32().expect("values"), vec![123.0]);
    }

    /// The result must be contiguous and must not alias the source.
    #[test]
    fn slice_ranges_result_is_contiguous_and_owned() {
        let t = coded_3d();
        let block = t.slice_ranges(&[(0, 2), (1, 3), (0, 2)]).expect("slice");
        match block {
            Tensor::F32(arr) => {
                assert!(arr.is_standard_layout());
                assert!(arr.as_slice().is_some());
                assert_eq!(arr.len(), 2 * 2 * 2);
            },
            _ => panic!("F32 expected"),
        }
    }

    /// Slicing the full extent of every axis reproduces the input exactly.
    #[test]
    fn slice_ranges_full_extent_is_the_identity() {
        let t = coded_3d();
        let full = t.slice_ranges(&[(0, 2), (0, 3), (0, 4)]).expect("slice");
        assert_eq!(full.shape(), t.shape());
        assert_eq!(
            full.to_vec_f32().expect("values"),
            t.to_vec_f32().expect("values")
        );
    }

    #[test]
    fn slice_ranges_rejects_invalid_ranges() {
        let t = coded_3d();
        // end past the axis length
        assert!(t.slice_ranges(&[(0, 2), (0, 3), (0, 5)]).is_err());
        // start after end
        assert!(t.slice_ranges(&[(0, 2), (2, 1), (0, 4)]).is_err());
        // more ranges than the tensor has axes
        assert!(t.slice_ranges(&[(0, 2), (0, 3), (0, 4), (0, 1)]).is_err());
    }

    /// Fewer ranges than axes is legal: the listed leading axes are narrowed and
    /// every trailing axis is kept whole.
    #[test]
    fn slice_ranges_keeps_unlisted_trailing_axes_whole() {
        let t = coded_3d();
        let partial = t.slice_ranges(&[(1, 2)]).expect("slice leading axis only");
        assert_eq!(partial.shape(), vec![1, 3, 4]);

        let full = t.slice_ranges(&[(1, 2), (0, 3), (0, 4)]).expect("explicit full ranges");
        assert_eq!(
            partial.to_vec_f32().expect("values"),
            full.to_vec_f32().expect("values")
        );
    }

    // ------------------------------------------------------------------
    // slice_multi
    // ------------------------------------------------------------------

    /// `slice_multi` had the same "clone the whole tensor first" shape as
    /// `slice_ranges`; the values must survive the rewrite.
    #[test]
    fn slice_multi_extracts_the_right_block() {
        let t = coded_3d();
        let block = t.slice_multi(&[(1, 2), (0, 2), (1, 3)]).expect("slice_multi");
        assert_eq!(block.shape(), vec![1, 2, 2]);
        assert_eq!(
            block.to_vec_f32().expect("values"),
            vec![101.0, 102.0, 111.0, 112.0]
        );
    }

    /// `slice_multi` requires one range per axis, unlike `slice_ranges`.
    #[test]
    fn slice_multi_requires_one_range_per_axis() {
        let t = coded_3d();
        assert!(t.slice_multi(&[(0, 2), (0, 3)]).is_err());
        assert!(t.slice_multi(&[(0, 2), (0, 3), (0, 5)]).is_err());
    }

    #[test]
    fn slice_multi_result_is_contiguous() {
        let t = coded_3d();
        let block = t.slice_multi(&[(0, 2), (1, 3), (0, 2)]).expect("slice_multi");
        match block {
            Tensor::F32(arr) => {
                assert!(arr.is_standard_layout());
                assert!(arr.as_slice().is_some());
            },
            _ => panic!("F32 expected"),
        }
    }

    /// The same slice must work on I64 tensors (the index/position path).
    #[test]
    fn slice_ranges_supports_i64_tensors() {
        let t = Tensor::from_vec_i64((0..12).collect::<Vec<i64>>(), &[3, 4]).expect("i64 tensor");
        let block = t.slice_ranges(&[(1, 3), (2, 4)]).expect("slice");
        assert_eq!(block.shape(), vec![2, 2]);
        match block {
            Tensor::I64(arr) => {
                let values: Vec<i64> = arr.iter().copied().collect();
                assert_eq!(values, vec![6, 7, 10, 11]);
            },
            _ => panic!("I64 expected"),
        }
    }

    // ------------------------------------------------------------------
    // gather
    // ------------------------------------------------------------------

    #[test]
    fn gather_selects_along_the_requested_dimension() {
        let data = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).expect("data");
        // Pick column 2 then column 0 from row 0, and column 1 twice from row 1.
        let index = Tensor::from_vec_i64(vec![2, 0, 1, 1], &[2, 2]).expect("index");

        let gathered = data.gather(1, &index).expect("gather");
        assert_eq!(gathered.shape(), vec![2, 2]);
        assert_eq!(
            gathered.to_vec_f32().expect("values"),
            vec![3.0, 1.0, 5.0, 5.0]
        );
    }

    /// Regression test: an index tensor larger than the source on a *non-gather*
    /// axis used to reach `data[IxDyn(&coords)]` and panic inside a fallible API.
    #[test]
    fn gather_rejects_an_oversized_index_on_a_non_gather_axis() {
        let data = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).expect("data");
        // 4 rows of indices for a 2-row source, gathering along axis 1.
        let index = Tensor::from_vec_i64(vec![0, 1, 0, 1, 0, 1, 0, 1], &[4, 2]).expect("index");

        let result = data.gather(1, &index);
        assert!(
            result.is_err(),
            "an index tensor with more rows than the source must be refused"
        );
    }

    /// Rank mismatches are refused too (PyTorch requires equal rank).
    #[test]
    fn gather_rejects_a_rank_mismatch() {
        let data = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).expect("data");
        let index = Tensor::from_vec_i64(vec![0, 1, 2], &[3]).expect("index");
        assert!(data.gather(1, &index).is_err());
    }

    /// An out-of-range index value on the gather axis is an error, not a panic.
    #[test]
    fn gather_rejects_an_out_of_range_index_value() {
        let data = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).expect("data");
        let index = Tensor::from_vec_i64(vec![0, 9], &[1, 2]).expect("index");
        assert!(data.gather(1, &index).is_err());
    }

    /// Gathering along axis 0 with a smaller index tensor is legal and must
    /// select rows, not columns.
    #[test]
    fn gather_along_axis_zero_selects_rows() {
        let data = Tensor::from_vec(
            vec![1.0f32, 2.0, 3.0, 10.0, 20.0, 30.0, 100.0, 200.0, 300.0],
            &[3, 3],
        )
        .expect("data");
        let index = Tensor::from_vec_i64(vec![2, 0, 1], &[1, 3]).expect("index");

        let gathered = data.gather(0, &index).expect("gather");
        assert_eq!(gathered.shape(), vec![1, 3]);
        // column 0 from row 2, column 1 from row 0, column 2 from row 1
        assert_eq!(
            gathered.to_vec_f32().expect("values"),
            vec![100.0, 2.0, 30.0]
        );
    }
}
