use io_utils::imux::IMuxSync;

#[inline]
pub fn serialize<W: std::io::Write + Send, T: ?Sized>(
    writer: &mut IMuxSync<W>,
    value: &T,
) -> Result<(), bincode::Error>
where
    T: serde::Serialize,
{
    let bytes: Vec<u8> = bincode::serialize(value)?;
    let _ = writer.write(&bytes)?;
    writer.flush()?;
    Ok(())
}

#[inline]
pub fn deserialize<R, T>(reader: &mut IMuxSync<R>) -> bincode::Result<T>
where
    R: std::io::Read + Send,
    T: serde::de::DeserializeOwned,
{
    let bytes: Vec<u8> = reader.read()?;
    bincode::deserialize(&bytes[..])
}
