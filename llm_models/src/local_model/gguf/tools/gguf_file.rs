//! Support for the GGUF file format.
//!
//! Spec: https://github.com/philpax/ggml/blob/gguf-spec/docs/gguf.md
//! Adapted from: https://github.com/huggingface/candle/blob/main/candle-core/src/quantized/gguf_file.rs

use super::gguf_tensors::{GgmlDType, TensorInfo};
use byteorder::{LittleEndian, ReadBytesExt};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

pub const DEFAULT_ALIGNMENT: u32 = 32;

pub struct GgufFile {
    pub magic: VersionedMagic,
    pub metadata: HashMap<String, Value>,
    pub tensors: Vec<TensorInfo>,
    pub tensor_data_offset: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Magic {
    Gguf,
}

impl TryFrom<u32> for Magic {
    type Error = crate::Error;
    fn try_from(value: u32) -> crate::Result<Self> {
        let magic = match value {
            0x46554747 | 0x47475546 => Self::Gguf,
            _ => crate::bail!("unknown magic 0x{value:08x}"),
        };
        Ok(magic)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VersionedMagic {
    GgufV1,
    GgufV2,
    GgufV3,
}

impl VersionedMagic {
    fn read<R: std::io::Read>(reader: &mut R) -> crate::Result<Self> {
        let magic = reader.read_u32::<LittleEndian>()?;
        let magic = Magic::try_from(magic)?;
        let version = reader.read_u32::<LittleEndian>()?;
        let versioned_magic = match (magic, version) {
            (Magic::Gguf, 1) => Self::GgufV1,
            (Magic::Gguf, 2) => Self::GgufV2,
            (Magic::Gguf, 3) => Self::GgufV3,
            _ => crate::bail!("gguf: unsupported magic/version {magic:?}/{version}"),
        };
        Ok(versioned_magic)
    }
}

impl GgufFile {
    pub fn read<R: std::io::Seek + std::io::Read>(reader: &mut R) -> crate::Result<Self> {
        let magic = VersionedMagic::read(reader)?;

        let tensor_count = match magic {
            VersionedMagic::GgufV1 => reader.read_u32::<LittleEndian>()? as usize,
            VersionedMagic::GgufV2 | VersionedMagic::GgufV3 => {
                reader.read_u64::<LittleEndian>()? as usize
            }
        };
        let metadata_kv_count = match magic {
            VersionedMagic::GgufV1 => reader.read_u32::<LittleEndian>()? as usize,
            VersionedMagic::GgufV2 | VersionedMagic::GgufV3 => {
                reader.read_u64::<LittleEndian>()? as usize
            }
        };

        let mut metadata = HashMap::new();
        for _idx in 0..metadata_kv_count {
            let key = read_string(reader, &magic)?;
            let value_type = reader.read_u32::<LittleEndian>()?;
            let value_type = ValueType::from_u32(value_type)?;
            let value = Value::read(reader, value_type, &magic)?;
            metadata.insert(key, value);
        }

        let mut tensor_infos = vec![];
        for _idx in 0..tensor_count {
            let tensor_name = read_string(reader, &magic)?;
            let n_dimensions = reader.read_u32::<LittleEndian>()?;

            let mut dimensions: Vec<usize> = match magic {
                VersionedMagic::GgufV1 => {
                    let mut dimensions = vec![0; n_dimensions as usize];
                    reader.read_u32_into::<LittleEndian>(&mut dimensions)?;
                    dimensions.into_iter().map(|c| c as usize).collect()
                }
                VersionedMagic::GgufV2 | VersionedMagic::GgufV3 => {
                    let mut dimensions = vec![0; n_dimensions as usize];
                    reader.read_u64_into::<LittleEndian>(&mut dimensions)?;
                    dimensions.into_iter().map(|c| c as usize).collect()
                }
            };
            dimensions.reverse();

            let ggml_dtype = reader.read_u32::<LittleEndian>()?;
            let ggml_dtype = GgmlDType::from_u32(ggml_dtype)?;

            let offset = reader.read_u64::<LittleEndian>()?;
            tensor_infos.push(TensorInfo {
                name: tensor_name,
                shape: dimensions,
                offset,
                ggml_dtype,
            });
        }
        let position = reader.stream_position()?;
        let alignment = match metadata.get("general.alignment") {
            Some(Value::U8(v)) => *v as u32,
            Some(Value::U16(v)) => *v as u32,
            Some(Value::U32(v)) => *v as u32,
            Some(Value::I8(v)) if *v >= 0 => *v as u32,
            Some(Value::I16(v)) if *v >= 0 => *v as u32,
            Some(Value::I32(v)) if *v >= 0 => *v as u32,
            _ => DEFAULT_ALIGNMENT,
        };
        metadata.insert("general.alignment".to_string(), Value::U32(alignment));
        let alignment = alignment as u64;
        let tensor_data_offset = (position + alignment - 1) / alignment * alignment;
        Ok(Self {
            magic,
            metadata,
            tensors: tensor_infos,
            tensor_data_offset,
        })
    }

    pub fn get_value<T: FromValue>(&self, key: &str) -> crate::Result<T> {
        match self.metadata.get(key) {
            Some(value) => T::from_value(value),
            None => T::from_none(key),
        }
    }

    pub fn get_pathed_value<T: FromValue>(
        &self,
        path_prefixes: &[&str],
        field_name: &str,
    ) -> crate::Result<T> {
        let prop_key = if path_prefixes.is_empty() {
            field_name.to_string()
        } else {
            let prefix = path_prefixes.join(".");
            format!("{}.{}", prefix, field_name)
        };
        self.get_value(&prop_key)
    }

    pub fn size(&self) -> u64 {
        self.tensors.iter().map(|t| t.size()).sum()
    }
}

fn read_string<R: std::io::Read>(reader: &mut R, magic: &VersionedMagic) -> crate::Result<String> {
    let len = match magic {
        VersionedMagic::GgufV1 => reader.read_u32::<LittleEndian>()? as usize,
        VersionedMagic::GgufV2 | VersionedMagic::GgufV3 => {
            reader.read_u64::<LittleEndian>()? as usize
        }
    };
    let mut v = vec![0u8; len];
    reader.read_exact(&mut v)?;
    // GGUF strings are supposed to be non-null terminated but in practice this happens.
    while let Some(0) = v.last() {
        v.pop();
    }
    // GGUF strings are utf8 encoded but there are cases that don't seem to be valid.
    Ok(String::from_utf8_lossy(&v).into_owned())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ValueType {
    // The value is a 8-bit unsigned integer.
    U8,
    // The value is a 8-bit signed integer.
    I8,
    // The value is a 16-bit unsigned little-endian integer.
    U16,
    // The value is a 16-bit signed little-endian integer.
    I16,
    // The value is a 32-bit unsigned little-endian integer.
    U32,
    // The value is a 32-bit signed little-endian integer.
    I32,
    // The value is a 64-bit unsigned little-endian integer.
    U64,
    // The value is a 64-bit signed little-endian integer.
    I64,
    // The value is a 32-bit IEEE754 floating point number.
    F32,
    // The value is a 64-bit IEEE754 floating point number.
    F64,
    // The value is a boolean.
    // 1-byte value where 0 is false and 1 is true.
    // Anything else is invalid, and should be treated as either the model being invalid or the reader being buggy.
    Bool,
    // The value is a UTF-8 non-null-terminated string, with length prepended.
    String,
    // The value is an array of other values, with the length and type prepended.
    // Arrays can be nested, and the length of the array is the number of elements in the array, not the number of bytes.
    Array,
}

impl ValueType {
    fn from_u32(v: u32) -> crate::Result<Self> {
        let v = match v {
            0 => Self::U8,
            1 => Self::I8,
            2 => Self::U16,
            3 => Self::I16,
            4 => Self::U32,
            5 => Self::I32,
            6 => Self::F32,
            7 => Self::Bool,
            8 => Self::String,
            9 => Self::Array,
            10 => Self::U64,
            11 => Self::I64,
            12 => Self::F64,
            v => {
                let bytes = v.to_le_bytes();
                let as_le = u32::from_le_bytes(bytes);
                let as_be = u32::from_be_bytes(bytes);
                let ascii_le = String::from_utf8_lossy(&bytes).to_string();
                let ascii_be =
                    String::from_utf8_lossy(&bytes.iter().rev().cloned().collect::<Vec<u8>>())
                        .to_string();

                crate::bail!(format!(
                    "Unrecognized value-type: {v} (0x{v:08x})\n\
                    As little-endian: {as_le} (0x{as_le:08x})\n\
                    As big-endian: {as_be} (0x{as_be:08x})\n\
                    ASCII (LE): {ascii_le}\n\
                    ASCII (BE): {ascii_be}"
                ))
            }
        };
        Ok(v)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Value {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    U64(u64),
    I64(i64),
    F32(f32),
    F64(f64),
    Bool(bool),
    String(String),
    Array(Vec<Value>),
}

impl Value {
    pub fn value_type(&self) -> ValueType {
        match self {
            Self::U8(_) => ValueType::U8,
            Self::I8(_) => ValueType::I8,
            Self::U16(_) => ValueType::U16,
            Self::I16(_) => ValueType::I16,
            Self::U32(_) => ValueType::U32,
            Self::I32(_) => ValueType::I32,
            Self::U64(_) => ValueType::U64,
            Self::I64(_) => ValueType::I64,
            Self::F32(_) => ValueType::F32,
            Self::F64(_) => ValueType::F64,
            Self::Bool(_) => ValueType::Bool,
            Self::String(_) => ValueType::String,
            Self::Array(_) => ValueType::Array,
        }
    }

    pub fn to_u8(&self) -> crate::Result<u8> {
        match self {
            Self::U8(v) => Ok(*v),
            v => crate::bail!("not a u8 {v:?}"),
        }
    }

    pub fn to_i8(&self) -> crate::Result<i8> {
        match self {
            Self::I8(v) => Ok(*v),
            v => crate::bail!("not a i8 {v:?}"),
        }
    }

    pub fn to_u16(&self) -> crate::Result<u16> {
        match self {
            Self::U16(v) => Ok(*v),
            v => crate::bail!("not a u16 {v:?}"),
        }
    }

    pub fn to_i16(&self) -> crate::Result<i16> {
        match self {
            Self::I16(v) => Ok(*v),
            v => crate::bail!("not a i16 {v:?}"),
        }
    }

    pub fn to_u32(&self) -> crate::Result<u32> {
        match self {
            Self::U32(v) => Ok(*v),
            v => crate::bail!("not a u32 {v:?}"),
        }
    }

    pub fn to_i32(&self) -> crate::Result<i32> {
        match self {
            Self::I32(v) => Ok(*v),
            v => crate::bail!("not a i32 {v:?}"),
        }
    }

    /// This will also automatically upcast any integral types which will not truncate.
    pub fn to_u64(&self) -> crate::Result<u64> {
        match self {
            Self::U64(v) => Ok(*v),
            // Autoupcast cases here
            Self::U8(v) => Ok(*v as u64),
            Self::U16(v) => Ok(*v as u64),
            Self::U32(v) => Ok(*v as u64),
            Self::Bool(v) => Ok(*v as u64),
            v => crate::bail!("not a u64 or upcastable to u64 {v:?}"),
        }
    }

    pub fn to_i64(&self) -> crate::Result<i64> {
        match self {
            Self::I64(v) => Ok(*v),
            v => crate::bail!("not a i64 {v:?}"),
        }
    }

    pub fn to_f32(&self) -> crate::Result<f32> {
        match self {
            Self::F32(v) => Ok(*v),
            v => crate::bail!("not a f32 {v:?}"),
        }
    }

    pub fn to_f64(&self) -> crate::Result<f64> {
        match self {
            Self::F64(v) => Ok(*v),
            v => crate::bail!("not a f64 {v:?}"),
        }
    }

    pub fn to_bool(&self) -> crate::Result<bool> {
        match self {
            Self::Bool(v) => Ok(*v),
            v => crate::bail!("not a bool {v:?}"),
        }
    }

    pub fn to_vec(&self) -> crate::Result<&Vec<Value>> {
        match self {
            Self::Array(v) => Ok(v),
            v => crate::bail!("not a vec {v:?}"),
        }
    }

    pub fn to_string(&self) -> crate::Result<&String> {
        match self {
            Self::String(v) => Ok(v),
            v => crate::bail!("not a string {v:?}"),
        }
    }

    fn read<R: std::io::Read>(
        reader: &mut R,
        value_type: ValueType,
        magic: &VersionedMagic,
    ) -> crate::Result<Self> {
        let v = match value_type {
            ValueType::U8 => Self::U8(reader.read_u8()?),
            ValueType::I8 => Self::I8(reader.read_i8()?),
            ValueType::U16 => Self::U16(reader.read_u16::<LittleEndian>()?),
            ValueType::I16 => Self::I16(reader.read_i16::<LittleEndian>()?),
            ValueType::U32 => Self::U32(reader.read_u32::<LittleEndian>()?),
            ValueType::I32 => Self::I32(reader.read_i32::<LittleEndian>()?),
            ValueType::U64 => Self::U64(reader.read_u64::<LittleEndian>()?),
            ValueType::I64 => Self::I64(reader.read_i64::<LittleEndian>()?),
            ValueType::F32 => Self::F32(reader.read_f32::<LittleEndian>()?),
            ValueType::F64 => Self::F64(reader.read_f64::<LittleEndian>()?),
            ValueType::Bool => match reader.read_u8()? {
                0 => Self::Bool(false),
                1 => Self::Bool(true),
                b => crate::bail!("unexpected bool value {b}"),
            },
            ValueType::String => Self::String(read_string(reader, magic)?),
            ValueType::Array => {
                let value_type = reader.read_u32::<LittleEndian>()?;
                let value_type = ValueType::from_u32(value_type)?;
                let len = match magic {
                    VersionedMagic::GgufV1 => reader.read_u32::<LittleEndian>()? as usize,
                    VersionedMagic::GgufV2 | VersionedMagic::GgufV3 => {
                        reader.read_u64::<LittleEndian>()? as usize
                    }
                };
                let mut vs = Vec::with_capacity(len);
                for _ in 0..len {
                    vs.push(Value::read(reader, value_type, magic)?)
                }
                Self::Array(vs)
            }
        };
        Ok(v)
    }
}

pub trait FromValue: Sized {
    fn from_value(value: &Value) -> crate::Result<Self>;

    fn from_none(key: &str) -> crate::Result<Self> {
        crate::bail!("missing key {key}")
    }
}

impl FromValue for String {
    fn from_value(value: &Value) -> crate::Result<Self> {
        value.to_string().cloned()
    }
}

impl FromValue for u64 {
    fn from_value(value: &Value) -> crate::Result<Self> {
        value.to_u64()
    }
}

impl FromValue for u32 {
    fn from_value(value: &Value) -> crate::Result<Self> {
        value.to_u32()
    }
}

impl FromValue for f32 {
    fn from_value(value: &Value) -> crate::Result<Self> {
        value.to_f32()
    }
}

impl FromValue for bool {
    fn from_value(value: &Value) -> crate::Result<Self> {
        value.to_bool()
    }
}

impl<T: FromValue> FromValue for Vec<T> {
    fn from_value(value: &Value) -> crate::Result<Self> {
        match value {
            Value::Array(arr) => arr.iter().map(T::from_value).collect(),
            _ => crate::bail!("not an array"),
        }
    }
}

impl<T: FromValue> FromValue for Option<T> {
    fn from_value(value: &Value) -> crate::Result<Self> {
        Ok(Some(T::from_value(value)?))
    }

    fn from_none(_key: &str) -> crate::Result<Self> {
        Ok(None)
    }
}
