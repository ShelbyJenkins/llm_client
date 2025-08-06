use ggus::{GGmlType, GGufFileType};
const EPS: f64 = 1e-6;

#[derive(Debug, thiserror::Error, PartialEq)]
pub enum GgmlFileTypeError {
    // ───────────────────────── conversion failures ──────────────────────────
    #[error("unrecognised GGuf `general.file_type`: {0:?}")]
    UnknownFileType(GGufFileType),

    #[error("unrecognised GGuf `general.file_type` numeric value: {0}")]
    UnknownFileTypeValue(u32),

    #[error("unrecognised `GGmlType`: {0:?}")]
    UnknownGGmlType(GGmlType),

    #[error("unrecognised `GGmlType` numeric value: {0}")]
    UnknownGGmlTypeValue(u32),

    #[error("unrecognised weight-block type")]
    UnknownBlockType(&'static str),

    #[error("no format uses exactly {0} bits per weight")]
    UnknownBitsPerWeight(f64),

    #[error("no supported quantisation tag found in `{0}`")]
    UnknownQuantisationTag(String),
}

macro_rules! ggml_file_type_enum {
    (
        $(
            $value:expr, $level:expr, $variant:ident, $tag:expr, $block:path, $ggml:path
        ),* $(,)?
    ) => {
        /// Compact mirror of `enum llama_ftype` in `llama.h`
        /// – commented-out variants and the `LLAMA_FTYPE_MOSTLY_` prefix removed.
        /// Upstream ref: <https://github.com/ggml-org/llama.cpp/blob/e434e69/include/llama.h>
        #[repr(u16)]
        #[allow(non_camel_case_types)]
        #[derive(Copy, Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
        #[non_exhaustive]
        pub enum GgmlFileType {
            $( $variant = $value ),*
        }

        impl GgmlFileType {
            /// Numeric value identical to `llama.h`.
            #[inline]
            pub const fn value(self) -> u16 {
                self as u16
            }

            /// Text tag such as `"Q4_K_M"`.
            #[inline]
            pub const fn as_str(self) -> &'static str {
                match self { $( GgmlFileType::$variant => $tag ),* }
            }

            /// Exact bits each weight occupies.
             #[inline]
            pub const fn bits_per_weight(self) -> f64 {
                match self {
                    $(
                        Self::$variant => {
                            (core::mem::size_of::<$block>() as f64 * 8.0)
                            / <$block as ggml_quants::DataBlock>::COUNT as f64
                        }
                    ),*
                }
            }

            /// Number of weights contained in one block (rarely needed).
            #[inline]
            pub const fn block_len(self) -> usize {
                match self {
                    $(Self::$variant => <$block as ggml_quants::DataBlock>::COUNT,)*
                }
            }

            #[inline]
            pub const fn level(self) -> u8 {
                match self {
                    $( Self::$variant => $level ),+
                }
            }

            /// Convert raw upstream value → enum.
            /// general.file_type: uint32: An enumerated value describing the type of the majority of the tensors in the file. Optional; can be inferred from the tensor types. From: https://github.com/ggml-org/ggml/blob/master/docs/gguf.md
            #[inline]
            pub fn from_file_type(ft: GGufFileType) -> Result<Self, GgmlFileTypeError> {
                $(
                    if ft as u32 == $ggml as u32 {
                        return Ok(Self::$variant);
                    }
                )*
               Err(GgmlFileTypeError::UnknownFileType(ft))
            }

            /// Convert raw upstream value → enum.
            /// general.file_type: uint32: An enumerated value describing the type of the majority of the tensors in the file. Optional; can be inferred from the tensor types. From: https://github.com/ggml-org/ggml/blob/master/docs/gguf.md
            #[inline]
            pub fn from_file_type_value<I: Into<u32>>(value: I) -> Result<Self, GgmlFileTypeError> {
                match value.into() {
                    $( $value => Ok(GgmlFileType::$variant), )*
                    other => Err(GgmlFileTypeError::UnknownFileTypeValue(other)),
                }
            }


            /// Convert an upstream **`GGmlType`** → enum.
            ///
            /// If several `GgmlFileType` variants share the same `GGmlType`
            /// (e.g. all `Q3K`-based formats), the **first** one listed
            /// in the macro invocation wins.
            #[inline]
            pub fn from_ggml_type(gg: GGmlType) -> Result<Self, GgmlFileTypeError> {
                $(
                    if gg as u32 == $ggml as u32 {
                        return Ok(Self::$variant);
                    }
                )*
                Err(GgmlFileTypeError::UnknownGGmlType(gg))
            }

            /// Convert an upstream **`GGmlType`** → enum.
            ///
            /// If several `GgmlFileType` variants share the same `GGmlType`
            /// (e.g. all `Q3K`-based formats), the **first** one listed
            /// in the macro invocation wins.
            #[inline]
            pub fn from_ggml_type_value<I: Into<u32>>(value: I) -> Result<Self, GgmlFileTypeError> {
                let value = value.into();
                $(
                    if value as u32 == $ggml as u32 {
                        return Ok(Self::$variant);
                    }
                )*
                Err(GgmlFileTypeError::UnknownGGmlTypeValue(value))
            }

            /// Convert a **block type** → enum.
            #[inline]
            pub fn from_block<B: 'static>() -> Result<Self, GgmlFileTypeError> {
                let tid = core::any::TypeId::of::<B>();
                $(
                    if tid == core::any::TypeId::of::<$block>() {
                        return Ok(Self::$variant);
                    }
                )*
                Err(GgmlFileTypeError::UnknownBlockType(
                    core::any::type_name::<B>(),
                ))
            }


            #[inline]
            pub fn from_bits_per_weight(bits: f64) -> Result<Self, GgmlFileTypeError> {
                $(
                    if (((core::mem::size_of::<$block>() as f64 * 8.0)
                        / <$block as ggml_quants::DataBlock>::COUNT as f64) - bits).abs() < EPS {
                        return Ok(Self::$variant);
                    }
                )*
                Err(GgmlFileTypeError::UnknownBitsPerWeight(bits))
            }
        }

        /// Compile-time table: tag → enum variant.
        pub const FTYPES: &[(&str, GgmlFileType)] = &[
            $( ($tag, GgmlFileType::$variant) ),*
        ];
    };
}

ggml_file_type_enum! {
     0, 32, AllF32  , "F32"    , f32               , GGmlType::F32  ,
     1, 16, F16     , "F16"    , ggml_quants::f16  , GGmlType::F16  ,
     2,  4, Q4_0    , "Q4_0"   , ggml_quants::Q4_0 , GGmlType::Q4_0 ,
     3,  4, Q4_1    , "Q4_1"   , ggml_quants::Q4_1 , GGmlType::Q4_1 ,
     7,  8, Q8_0    , "Q8_0"   , ggml_quants::Q8_0 , GGmlType::Q8_0 ,
     8,  5, Q5_0    , "Q5_0"   , ggml_quants::Q5_0 , GGmlType::Q5_0 ,
     9,  5, Q5_1    , "Q5_1"   , ggml_quants::Q5_1 , GGmlType::Q5_1 ,
    10,  2, Q2_K    , "Q2_K"   , ggml_quants::Q2K  , GGmlType::Q2K  ,
    11,  3, Q3_K_S  , "Q3_K_S" , ggml_quants::Q3K  , GGmlType::Q3K  ,
    12,  3, Q3_K_M  , "Q3_K_M" , ggml_quants::Q3K  , GGmlType::Q3K  ,
    13,  3, Q3_K_L  , "Q3_K_L" , ggml_quants::Q3K  , GGmlType::Q3K  ,
    14,  4, Q4_K_S  , "Q4_K_S" , ggml_quants::Q4K  , GGmlType::Q4K  ,
    15,  4, Q4_K_M  , "Q4_K_M" , ggml_quants::Q4K  , GGmlType::Q4K  ,
    16,  5, Q5_K_S  , "Q5_K_S" , ggml_quants::Q5K  , GGmlType::Q5K  ,
    17,  5, Q5_K_M  , "Q5_K_M" , ggml_quants::Q5K  , GGmlType::Q5K  ,
    18,  6, Q6_K    , "Q6_K"   , ggml_quants::Q6K  , GGmlType::Q6K  ,
    19,  2, IQ2_XXS , "IQ2_XXS", ggml_quants::Q2K  , GGmlType::IQ2XXS,
    20,  2, IQ2_XS  , "IQ2_XS" , ggml_quants::Q2K  , GGmlType::IQ2XS ,
    21,  2, Q2_K_S  , "Q2_K_S" , ggml_quants::Q2K  , GGmlType::Q2K  ,
    22,  2, IQ3_XS  , "IQ3_XS" , ggml_quants::Q3K  , GGmlType::IQ3XXS,
    23,  2, IQ3_XXS , "IQ3_XXS", ggml_quants::Q3K  , GGmlType::IQ3XXS,
    24,  1, IQ1_S   , "IQ1_S"  , ggml_quants::IQ1S , GGmlType::IQ1S ,
    25,  4, IQ4_NL  , "IQ4_NL" , ggml_quants::IQ4NL, GGmlType::IQ4NL,
    26,  3, IQ3_S   , "IQ3_S"  , ggml_quants::Q3K  , GGmlType::IQ3S ,
    27,  3, IQ3_M   , "IQ3_M"  , ggml_quants::Q3K  , GGmlType::IQ3S ,
    28,  2, IQ2_S   , "IQ2_S"  , ggml_quants::Q2K  , GGmlType::IQ2S ,
    29,  2, IQ2_M   , "IQ2_M"  , ggml_quants::Q2K  , GGmlType::IQ2S ,
    30,  4, IQ4_XS  , "IQ4_XS" , ggml_quants::IQ4XS, GGmlType::IQ4XS,
    31,  1, IQ1_M   , "IQ1_M"  , ggml_quants::IQ1M , GGmlType::IQ1M ,
    32, 16, BF16    , "BF16"   , ggml_quants::bf16 , GGmlType::BF16 ,
    // ─ experimental placeholders ─
    // 36, ?, TQ1_0, "TQ1_0", GgmlDType::TQ1_0 , GGmlType::TQ1_0 ,
    // 37, ?, TQ2_0, "TQ2_0", GgmlDType::TQ2_0 , GGmlType::TQ2_0 ,
}

static TAG_SET: std::sync::OnceLock<(regex::RegexSet, Vec<GgmlFileType>, Vec<&'static str>)> =
    std::sync::OnceLock::new();

impl GgmlFileType {
    fn build_pat(tag: &str) -> String {
        const DELIM: &str = r"[-_\.\u{2010}-\u{2015}]";
        let mut p = String::from("(?i)(?:^|");
        p.push_str(DELIM);
        p.push(')');
        let mut toks = tag.split('_');
        if let Some(first) = toks.next() {
            p.push_str(first);
            for t in toks {
                p.push_str("(?:");
                p.push_str(DELIM);
                p.push_str(")+");
                p.push_str(t);
            }
        }
        p.push_str("(?:$|");
        p.push_str(DELIM);
        p.push(')');
        p
    }

    fn match_ftype(stem: &str) -> Option<GgmlFileType> {
        let stem_uc = stem.to_ascii_uppercase();
        let (set, kinds, tags) = TAG_SET.get_or_init(|| {
            let mut pats = Vec::new();
            let mut kinds = Vec::new();
            let mut tags = Vec::new();
            for &(tag, ft) in FTYPES {
                pats.push(Self::build_pat(tag));
                kinds.push(ft);
                tags.push(tag);
            }
            let set = regex::RegexSet::new(&pats).expect("compile quant-tag RegexSet");
            (set, kinds, tags)
        });

        let m = set.matches(&stem_uc);
        if !m.matched_any() {
            return None;
        }
        let best = m.iter().max_by_key(|&i| tags[i].len()).unwrap();
        Some(kinds[best])
    }
}

impl TryFrom<&str> for GgmlFileType {
    type Error = GgmlFileTypeError;

    /// Accepts “Q4_K_M”, “q4-k-m”, etc.
    fn try_from(tag: &str) -> Result<Self, Self::Error> {
        Self::match_ftype(tag)
            .ok_or_else(|| GgmlFileTypeError::UnknownQuantisationTag(tag.to_string()))
    }
}

impl core::fmt::Display for GgmlFileType {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{} ({}-bit)", self.as_str(), self.level())
    }
}

#[cfg(test)]
mod tests {
    use core::mem::size_of;

    use ggml_quants::DataBlock;

    use super::*;

    #[test]
    fn tags_are_unique() {
        for i in 0..FTYPES.len() {
            for j in i + 1..FTYPES.len() {
                assert_ne!(FTYPES[i].0, FTYPES[j].0, "duplicate tag detected");
            }
        }
    }

    #[test]
    fn bits_per_weight_is_positive() {
        for &(_, ft) in FTYPES {
            assert!(
                ft.bits_per_weight() > 0.0,
                "{ft:?} produced non-positive b/W"
            );
        }
    }

    #[test]
    fn block_struct_consistency() {
        for &(_, ft) in FTYPES {
            match ft {
                /* ── scalar formats ───────────────────────────────────── */
                GgmlFileType::AllF32 | GgmlFileType::F16 | GgmlFileType::BF16 => {
                    assert_eq!(ft.block_len(), 1, "{ft:?} block_len");
                    let expected_bits = ft.level() as f64;
                    assert!(
                        (ft.bits_per_weight() - expected_bits).abs() < EPS,
                        "{ft:?} bits_per_weight"
                    );
                }

                /* classic 4/5/8-bit blocks */
                GgmlFileType::Q4_0 => check::<ggml_quants::Q4_0>(ft),
                GgmlFileType::Q4_1 => check::<ggml_quants::Q4_1>(ft),
                GgmlFileType::Q5_0 => check::<ggml_quants::Q5_0>(ft),
                GgmlFileType::Q5_1 => check::<ggml_quants::Q5_1>(ft),
                GgmlFileType::Q8_0 => check::<ggml_quants::Q8_0>(ft),

                /* 2-bit family */
                GgmlFileType::Q2_K
                | GgmlFileType::Q2_K_S
                | GgmlFileType::IQ2_XXS
                | GgmlFileType::IQ2_XS
                | GgmlFileType::IQ2_S
                | GgmlFileType::IQ2_M => check::<ggml_quants::Q2K>(ft),

                /* 3-bit family */
                GgmlFileType::Q3_K_S
                | GgmlFileType::Q3_K_M
                | GgmlFileType::Q3_K_L
                | GgmlFileType::IQ3_S
                | GgmlFileType::IQ3_M
                | GgmlFileType::IQ3_XXS
                | GgmlFileType::IQ3_XS => check::<ggml_quants::Q3K>(ft),

                /* 4-bit family */
                GgmlFileType::Q4_K_S | GgmlFileType::Q4_K_M => check::<ggml_quants::Q4K>(ft),
                GgmlFileType::IQ4_NL => check::<ggml_quants::IQ4NL>(ft),
                GgmlFileType::IQ4_XS => check::<ggml_quants::IQ4XS>(ft),

                /* 5-bit & 6-bit */
                GgmlFileType::Q5_K_S | GgmlFileType::Q5_K_M => check::<ggml_quants::Q5K>(ft),
                GgmlFileType::Q6_K => check::<ggml_quants::Q6K>(ft),

                /* one-bit family */
                GgmlFileType::IQ1_S => check::<ggml_quants::IQ1S>(ft),
                GgmlFileType::IQ1_M => check::<ggml_quants::IQ1M>(ft),
            }
        }

        fn check<B: DataBlock>(ft: GgmlFileType) {
            assert_eq!(ft.block_len(), B::COUNT, "{ft:?} block_len");
            let bits = (size_of::<B>() as f64 * 8.0) / B::COUNT as f64;
            assert!(
                (ft.bits_per_weight() - bits).abs() < EPS,
                "{ft:?} bits_per_weight mismatch: got {}, expected {bits}",
                ft.bits_per_weight()
            );
        }
    }
}
