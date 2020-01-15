#[inline]
pub fn serialize<W: std::io::Write, T: ?Sized>(mut w: W, value: &T) -> Result<(), bincode::Error>
where
    T: serde::Serialize,
{
    let bytes: Vec<u8> = bincode::serialize(value)?;
    let _ = w.write(&bytes)?;
    Ok(())
}

#[inline]
pub fn deserialize<R, T>(reader: R) -> bincode::Result<T>
where
    R: std::io::Read,
    T: serde::de::DeserializeOwned,
{
    bincode::deserialize_from(reader)
}
