use crate::errors::Result;
use crate::traits::Config;
use std::fs::File;
use std::io::BufReader;
use std::path::Path;

pub fn load_config<T: Config>(path: &Path) -> Result<T> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let config: T = serde_json::from_reader(reader)?;
    config.validate()?;
    Ok(config)
}
