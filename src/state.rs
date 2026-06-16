use std::path::{Component, Path, PathBuf};

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct AgentState {
    read_roots: Vec<PathBuf>,
    write_roots: Vec<PathBuf>,
}

impl AgentState {
    pub fn new(
        read_roots: impl IntoIterator<Item = PathBuf>,
        write_roots: impl IntoIterator<Item = PathBuf>,
    ) -> Self {
        let mut read_roots = normalize_many(read_roots);
        let write_roots = normalize_many(write_roots);

        for root in &write_roots {
            push_unique(&mut read_roots, root.clone());
        }

        Self {
            read_roots,
            write_roots,
        }
    }

    pub fn read_roots(&self) -> &[PathBuf] {
        &self.read_roots
    }

    pub fn write_roots(&self) -> &[PathBuf] {
        &self.write_roots
    }

    pub fn can_read(&self, path: impl AsRef<Path>) -> bool {
        is_under_any_root(path.as_ref(), &self.read_roots)
    }

    pub fn can_write(&self, path: impl AsRef<Path>) -> bool {
        is_under_any_root(path.as_ref(), &self.write_roots)
    }
}

fn is_under_any_root(path: &Path, roots: &[PathBuf]) -> bool {
    let path = normalize_path(path);
    roots.iter().any(|root| path.starts_with(root))
}

fn normalize_many(paths: impl IntoIterator<Item = PathBuf>) -> Vec<PathBuf> {
    let mut normalized = Vec::new();
    for path in paths {
        push_unique(&mut normalized, normalize_path(&path));
    }
    normalized
}

fn push_unique(paths: &mut Vec<PathBuf>, path: PathBuf) {
    if !paths.iter().any(|existing| existing == &path) {
        paths.push(path);
    }
}

fn normalize_path(path: &Path) -> PathBuf {
    let absolute = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()
            .unwrap_or_else(|_| PathBuf::from("."))
            .join(path)
    };

    match absolute.canonicalize() {
        Ok(canonical) => canonical,
        Err(_) => normalize_with_existing_ancestor(&absolute),
    }
}

fn normalize_with_existing_ancestor(path: &Path) -> PathBuf {
    let mut missing = Vec::new();
    let mut cursor = path;

    while !cursor.exists() {
        match (cursor.parent(), cursor.file_name()) {
            (Some(parent), Some(file_name)) => {
                missing.push(file_name.to_os_string());
                cursor = parent;
            }
            _ => return normalize_lexically(path),
        }
    }

    let mut normalized = cursor
        .canonicalize()
        .unwrap_or_else(|_| normalize_lexically(cursor));
    for component in missing.iter().rev() {
        normalized.push(component);
    }
    normalized
}

fn normalize_lexically(path: &Path) -> PathBuf {
    let mut normalized = PathBuf::new();
    for component in path.components() {
        match component {
            Component::CurDir => {}
            Component::ParentDir => {
                normalized.pop();
            }
            other => normalized.push(other.as_os_str()),
        }
    }
    normalized
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn write_roots_are_also_readable() {
        let temp = tempfile::tempdir().unwrap();
        let read = temp.path().join("read");
        let write = temp.path().join("write");
        std::fs::create_dir_all(&read).unwrap();
        std::fs::create_dir_all(&write).unwrap();

        let state = AgentState::new([read.clone()], [write.clone()]);

        assert!(state.can_read(read.join("file.txt")));
        assert!(state.can_read(write.join("file.txt")));
        assert!(state.can_write(write.join("file.txt")));
        assert!(!state.can_write(read.join("file.txt")));
    }

    #[test]
    fn sibling_paths_do_not_match_by_string_prefix() {
        let temp = tempfile::tempdir().unwrap();
        let allowed = temp.path().join("allowed");
        let sibling = temp.path().join("allowed-but-not-really");
        std::fs::create_dir_all(&allowed).unwrap();
        std::fs::create_dir_all(&sibling).unwrap();

        let state = AgentState::new([allowed], []);

        assert!(!state.can_read(sibling.join("secret.txt")));
    }
}
